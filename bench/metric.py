"""Composite story-quality metric for ``dspy.Evaluate`` and the optimizer.

Folds the deterministic canon metrics (and, optionally, cached LM-judge verdicts)
into a single score in ``[0, 1]`` so prompts/demos can be tuned to reduce the
real continuity failures. Higher is better; each failing check subtracts its
weight.

The metric is deterministic by default (LM judges off), so it runs in CI without
an API key. ``make_story_metric`` returns a callable with the
``metric(example, prediction, trace=None) -> float`` shape DSPy optimizers and
``dspy.Evaluate`` expect.
"""
from __future__ import annotations

from pathlib import Path

import canon as canon_mod
import canon_guard
import canon_metrics

# Penalty weight per failing check. Hard continuity gates cost the most.
WEIGHTS: dict[str, float] = {
    "scaffolding_leak": 0.20,
    "name_arc": 0.20,
    "canon_fact": 0.20,
    "required_artifact": 0.15,
    "timeline_age": 0.15,
    # LM-judge checks (only applied when verdicts are supplied).
    "appearance_described": 0.15,
    "appearance_vs_spec": 0.15,
    "azazel_maw": 0.10,
    "leash_fray": 0.10,
    "kira_not_strawman": 0.10,
    "dangling_payoff": 0.05,
}
DEFAULT_WEIGHT = 0.10


def _prediction_text(prediction) -> str:
    """Extract the manuscript/prose text from a variety of prediction shapes."""
    if isinstance(prediction, str):
        return prediction
    for attr in ("story", "manuscript", "prose", "text"):
        value = getattr(prediction, attr, None)
        if isinstance(value, str) and value:
            return value
    return str(prediction)


def score_text(text: str, canon, judge_findings=None, weights: dict[str, float] | None = None) -> float:
    """Score one manuscript: ``1.0`` minus the weight of every failing check.

    ``judge_findings`` is an optional pre-computed list of ``qa.Finding`` from the
    LM judges (so the LM is never called from inside the metric). Result is
    clamped to ``[0, 1]``.
    """
    weights = weights or WEIGHTS
    findings = list(canon_metrics.run_all(text, canon))
    if judge_findings:
        findings.extend(judge_findings)

    penalty = 0.0
    for f in findings:
        if f.severity == "fail":
            penalty += weights.get(f.check, DEFAULT_WEIGHT)
        elif f.severity == "warn":
            penalty += 0.5 * weights.get(f.check, DEFAULT_WEIGHT)
    return max(0.0, min(1.0, 1.0 - penalty))


def make_story_metric(canon=None, *, threshold: float | None = None, weights=None):
    """Build a DSPy-compatible metric closure bound to a ``canon``.

    When ``threshold`` is given the metric returns a ``bool`` (score >= threshold)
    — the shape bootstrapping optimizers expect; otherwise it returns the float
    score for ``dspy.Evaluate``.
    """
    canon = canon if canon is not None else canon_mod.load_canon()

    def metric(example, prediction, trace=None) -> float | bool:
        text = _prediction_text(prediction)
        # An example may pin its own canon; fall back to the bound one.
        ex_canon = getattr(example, "canon", None) or canon
        judge_findings = getattr(example, "judge_findings", None)
        score = score_text(text, ex_canon, judge_findings=judge_findings, weights=weights)
        if threshold is not None:
            return score >= threshold
        # During optimization (trace is not None) DSPy wants a hard pass/fail.
        if trace is not None:
            return score >= 1.0
        return score

    return metric


def make_chapter_metric(canon=None, *, threshold: float | None = None, weights=None):
    """Build a per-chapter metric for optimizing the chapter drafter.

    Unlike :func:`make_story_metric` (whole manuscript), this scores one chapter's
    prose against the canon using the same deterministic critic the generation
    guard uses, so the optimizer is tuned on exactly the constraints the pipeline
    enforces. The DSPy ``example`` must carry ``chapter_number`` (and optionally
    ``chapter_title``); it may pin its own ``canon``.
    """
    canon = canon if canon is not None else canon_mod.load_canon()
    weights = weights or WEIGHTS

    def metric(example, prediction, trace=None) -> float | bool:
        prose = _prediction_text(prediction)
        number = int(getattr(example, "chapter_number", 1))
        title = getattr(example, "chapter_title", "Untitled")
        ex_canon = getattr(example, "canon", None) or canon
        violations = canon_guard.chapter_violations(prose, ex_canon, number, title)
        penalty = sum(weights.get(f.check, DEFAULT_WEIGHT) for f in violations)
        score = max(0.0, min(1.0, 1.0 - penalty))
        if threshold is not None:
            return score >= threshold
        # During optimization (trace is not None) DSPy wants a hard pass/fail.
        if trace is not None:
            return score >= 1.0
        return score

    return metric


def score_path(story_md_path: Path, canon=None) -> float:
    """Convenience: score a story markdown file on disk (deterministic only)."""
    canon = canon if canon is not None else canon_mod.load_canon()
    return score_text(Path(story_md_path).read_text(encoding="utf-8"), canon)
