"""Unified story scorecard — the executable Definition of Done.

Composes the deterministic Tier-1 checks (:mod:`qa`) with the LLM-backed
Tier-2 / Tier-3 judges into a single machine-readable scorecard per draft, per
``bench/eval-spec.md``:

- Tier-2: POV consistency (:mod:`pov_check`), prose lint (:mod:`story_linter`,
  detection mode), continuity / contradictions and premise fidelity
  (:mod:`bench.judge`).
- Tier-3: the six-axis qualitative rubric (:mod:`bench.judge`).

Tier-1 runs with no model. The LLM gates need a configured DSPy LM: set
``JudgeInputs.run_llm`` (the ``scripts/evaluate.py --with-llm`` path) or inject
precomputed results (used by tests). A draft only reports ``ship=True`` when
every evaluated gate is clean *and* every required gate actually ran
(``complete=True``) — a deterministic-only pass reports ``tier1_clean=True``
but ``ship=False`` because the judges could not run.

When the rubric runs, the "real ending" Definition-of-Done item is judged by
it; otherwise that item is reported as a ``manual`` check.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import qa
from qa import Finding

PASS, FAIL, WARN, SKIPPED, MANUAL = "pass", "fail", "warn", "skipped", "manual"

# Soft budgets from bench/eval-spec.md (T1.3 / T1.4).
NAME_DRIFT_BUDGET = 2
PHRASE_REUSE_BUDGET = 5

# Tier-1 deterministic gates, in report order.
TIER1_GATES = (
    "structure",
    "chapter_length",
    "chapter_band",
    "placeholder_protagonist",
    "character_presence",
    "name_drift",
    "cross_chapter_phrase_reuse",
)


@dataclass
class JudgeInputs:
    """Configuration + injection seams for the LLM-backed gates.

    ``run_llm`` turns the Tier-2/3 judges on; ``idea`` is required for the
    premise-fidelity gate. The ``*_classifications`` / ``*_replacements`` /
    ``contradictions`` / ``premise_verdict`` / ``rubric_scores`` fields inject
    precomputed results so tests (and cached runs) can skip the model.
    """

    idea: str | None = None
    run_llm: bool = False
    rubric_samples: int = 1
    pov_classifications: list | None = None
    linter_replacements: list | None = None
    contradictions: list | None = None
    premise_verdict: object | None = None
    rubric_scores: list | None = None


@dataclass
class Gate:
    """One evaluated gate: its tier, status, and any fail/warn messages."""

    gate: str
    tier: int
    status: str  # pass | fail | warn | skipped
    messages: list[str] = field(default_factory=list)


@dataclass
class FullScorecard:
    """The unified verdict for one draft."""

    ship: bool
    complete: bool
    tier1_clean: bool
    gates: list[Gate]
    budgets: dict[str, dict]
    definition_of_done: list[dict]
    manual_checks: list[str]
    not_evaluated: list[str]
    fails: list[dict]
    warns: list[dict]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _gate_from_findings(name: str, tier: int, findings: list[Finding]) -> Gate:
    """Collapse the findings for one check into a single gate status."""
    relevant = [f for f in findings if f.check == name]
    if any(f.severity == "fail" for f in relevant):
        status = FAIL
    elif any(f.severity == "warn" for f in relevant):
        status = WARN
    else:
        status = PASS
    messages = [f.message for f in relevant if f.severity in ("fail", "warn")]
    return Gate(gate=name, tier=tier, status=status, messages=messages)


def _tier1_findings(text: str) -> list[Finding]:
    """Run every deterministic check that backs a Tier-1 gate."""
    return [
        *qa.check_structure(text),
        *qa.check_chapter_length(text),
        *qa.check_chapter_band(text),
        *qa.check_placeholder_protagonist(text),
        *qa.check_character_presence(text),
        *qa.check_name_drift(text),
        *qa.check_cross_chapter_phrase_reuse(text),
    ]


def _budgets(findings: list[Finding]) -> dict[str, dict]:
    """Count warn-level drift / reuse findings against their soft budgets."""
    drift = sum(
        1 for f in findings if f.check == "name_drift" and f.severity == "warn"
    )
    reuse = sum(
        1
        for f in findings
        if f.check == "cross_chapter_phrase_reuse" and f.severity == "warn"
    )
    return {
        "name_drift": {
            "count": drift, "budget": NAME_DRIFT_BUDGET, "over": drift > NAME_DRIFT_BUDGET,
        },
        "phrase_reuse": {
            "count": reuse, "budget": PHRASE_REUSE_BUDGET, "over": reuse > PHRASE_REUSE_BUDGET,
        },
    }


def _pov_gate(
    chapters: dict[str, str],
    pov_classifications: list | None,
    run_llm: bool,
) -> Gate:
    """Tier-2 POV consistency, or a skipped gate when no model is available."""
    if pov_classifications is None and not run_llm:
        return Gate("pov_consistency", 2, SKIPPED, ["needs a model (run with --with-llm)"])
    import pov_check

    findings = pov_check.check_pov_consistency(chapters, pov_classifications)
    return _gate_from_findings("pov_consistency", 2, findings)


def _linter_gate(
    text: str,
    linter_replacements: list | None,
    run_llm: bool,
) -> Gate:
    """Tier-2 prose lint in *detection* mode — advisory (warn), never mutating."""
    if linter_replacements is None and not run_llm:
        return Gate("prose_linter", 2, SKIPPED, ["needs a model (run with --with-llm)"])
    replacements = linter_replacements
    if replacements is None:
        import story_linter

        proposed = story_linter.lint(text)
        _, joined = story_linter._extract_inputs(text)
        replacements = story_linter.validate(proposed, joined)
    if not replacements:
        return Gate("prose_linter", 2, PASS, [])
    messages = [
        f"{r.find!r} -> {r.replace!r}" + (f" ({r.reason})" if r.reason else "")
        for r in replacements
    ]
    return Gate("prose_linter", 2, WARN, messages)


def _continuity_gate(
    chapters: dict[str, str],
    cast: str,
    contradictions: list | None,
    run_llm: bool,
) -> Gate:
    """Tier-2 continuity / contradiction judge."""
    if contradictions is None and not run_llm:
        return Gate("continuity", 2, SKIPPED, ["needs a model (run with --with-llm)"])
    import bench.judge as judge

    findings = judge.check_continuity(chapters, cast, contradictions)
    return _gate_from_findings("continuity", 2, findings)


def _premise_gate(
    idea: str | None,
    chapters: dict[str, str],
    verdict: object | None,
    run_llm: bool,
) -> Gate:
    """Tier-2 premise-fidelity judge (needs the original idea)."""
    if verdict is None and not run_llm:
        return Gate("premise_fidelity", 2, SKIPPED, ["needs a model (run with --with-llm)"])
    if verdict is None and not idea:
        return Gate("premise_fidelity", 2, SKIPPED, ["needs the original idea (pass --idea)"])
    import bench.judge as judge

    findings = judge.check_premise_fidelity(idea or "", chapters, verdict)
    return _gate_from_findings("premise_fidelity", 2, findings)


def _rubric_gate(
    chapters: dict[str, str],
    scores: list | None,
    samples: int,
    run_llm: bool,
) -> Gate:
    """Tier-3 six-axis qualitative rubric judge."""
    if scores is None and not run_llm:
        return Gate("rubric", 3, SKIPPED, ["needs a model (run with --with-llm)"])
    import bench.judge as judge

    findings = judge.check_rubric(chapters, scores, samples=samples)
    return _gate_from_findings("rubric", 3, findings)


def _definition_of_done(by_name: dict[str, Gate], tier1_clean: bool) -> list[dict]:
    """Map docs/07 Definition-of-Done items onto the gates that enforce them."""

    def status(name: str) -> str:
        gate = by_name.get(name)
        return gate.status if gate is not None else SKIPPED

    # "Real ending" is judged by the Tier-3 rubric when it ran; otherwise manual.
    rubric = by_name.get("rubric")
    ending = rubric.status if rubric is not None and rubric.status != SKIPPED else MANUAL

    return [
        {"item": "All required sections present", "status": status("structure"), "gate": "structure"},
        {"item": "N non-empty chapters (>= hard floor)", "status": status("chapter_length"), "gate": "chapter_length"},
        {"item": "Protagonist named in prose (no placeholders)", "status": status("placeholder_protagonist"), "gate": "placeholder_protagonist"},
        {"item": "Real ending dramatizes the spine's final beats", "status": ending, "gate": "rubric"},
        {"item": "No real QA fails", "status": PASS if tier1_clean else FAIL, "gate": "tier1"},
        {"item": "POV consistent across chapters", "status": status("pov_consistency"), "gate": "pov_consistency"},
        {"item": "Story dramatizes the premise", "status": status("premise_fidelity"), "gate": "premise_fidelity"},
    ]


def evaluate(text: str, judge: JudgeInputs | None = None) -> FullScorecard:
    """Score one story markdown string against the full Definition of Done.

    ``judge`` carries the LLM-gate config + injection seams (see
    :class:`JudgeInputs`). With its defaults (no model, nothing injected) the
    Tier-2/3 gates report ``skipped`` and only Tier-1 is decided.
    """
    judge = judge or JudgeInputs()
    chapters = qa.split_chapters(text)
    cast = qa._section(text, 3, "Characters")
    findings = _tier1_findings(text)
    gates = [_gate_from_findings(name, 1, findings) for name in TIER1_GATES]
    gates.append(_pov_gate(chapters, judge.pov_classifications, judge.run_llm))
    gates.append(_linter_gate(text, judge.linter_replacements, judge.run_llm))
    gates.append(_continuity_gate(chapters, cast, judge.contradictions, judge.run_llm))
    gates.append(_premise_gate(judge.idea, chapters, judge.premise_verdict, judge.run_llm))
    gates.append(_rubric_gate(chapters, judge.rubric_scores, judge.rubric_samples, judge.run_llm))
    return _summarize(gates, _budgets(findings))


def _summarize(gates: list[Gate], budgets: dict[str, dict]) -> FullScorecard:
    """Derive the ship / complete / tier1 verdict from the evaluated gates."""
    over_budget = any(b["over"] for b in budgets.values())
    fails = [_row(g) for g in gates if g.status == FAIL]
    warns = [_row(g) for g in gates if g.status == WARN]
    not_evaluated = [g.gate for g in gates if g.status == SKIPPED]
    tier1_clean = not any(g.status == FAIL for g in gates if g.tier == 1) and not over_budget
    complete = not not_evaluated
    dod = _definition_of_done({g.gate: g for g in gates}, tier1_clean)
    return FullScorecard(
        ship=not fails and not over_budget and complete,
        complete=complete,
        tier1_clean=tier1_clean,
        gates=gates,
        budgets=budgets,
        definition_of_done=dod,
        manual_checks=[d["item"] for d in dod if d["status"] == MANUAL],
        not_evaluated=not_evaluated,
        fails=fails,
        warns=warns,
    )


def _row(gate: Gate) -> dict:
    return {"gate": gate.gate, "tier": gate.tier, "messages": gate.messages}


def evaluate_path(path: Path, judge: JudgeInputs | None = None) -> FullScorecard:
    """Read a story markdown file and evaluate it. See :func:`evaluate`."""
    return evaluate(Path(path).read_text(encoding="utf-8"), judge)


def main(argv: list[str] | None = None) -> int:
    """Deterministic (Tier-1) scorecard CLI. For Tier-2 use scripts/evaluate.py."""
    parser = argparse.ArgumentParser(
        description="Deterministic Tier-1 story scorecard. "
        "For the LLM-backed Tier-2 gates run scripts/evaluate.py --with-llm.",
    )
    parser.add_argument("story", type=Path, help="story markdown file")
    args = parser.parse_args(argv)
    card = evaluate_path(args.story)
    print(card.to_json())
    return 0 if card.tier1_clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
