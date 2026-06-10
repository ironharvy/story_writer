"""Tier-2/Tier-3 LLM-judge checks for the bench harness.

`bench/criteria.py` covers the deterministic Tier-1 gates; this is the
LLM-judge layer the eval-spec deferred until "the prompt + judge model are
chosen". It reuses the gates in ``evals/`` over a story artifact loaded by
``evals.loader`` and returns plain finding dicts so the scorecard can fold them
in without importing dspy itself.

The caller must have configured a DSPy LM first (e.g. via
``dspy_runtime.configure_dspy``); this module never configures the model, so
the deterministic bench path stays model-free unless judging is explicitly
requested.
"""
from __future__ import annotations

from pathlib import Path

from evals import bible_consistency, bible_internal, comprehension, continuity, premise_fidelity
from evals.loader import load_story
from evals.types import EvalFinding


def _as_dict(f: EvalFinding) -> dict:
    return {
        "check": f.check,
        "severity": f.severity,
        "message": f.message,
        "fix": f.fix,
        "locus": f.locus,
    }


def judge(story_md_path: Path, *, per_chapter: bool = False, idea: str = "") -> list[dict]:
    """Run the LLM-judge gates over one story artifact; return finding dicts.

    Manuscript-level gates (bible self-consistency, continuity, premise
    fidelity) always run. ``per_chapter`` additionally runs the bible and
    comprehension gates on every chapter — accurate but N× slower.
    """
    story = load_story(story_md_path)
    findings: list[EvalFinding] = []

    internal_report, _clar = bible_internal.check(story.world_bible)
    findings.extend(internal_report.findings)
    findings.extend(continuity.check(story.chapters).findings)

    summary = "\n\n".join(
        f"### {t}\n{' '.join(p.split()[:140])}" for t, p in story.chapters.items()
    )
    findings.extend(
        premise_fidelity.check(idea or story.premise, story.premise, story.spine, summary).findings
    )

    if per_chapter:
        for title, prose in story.chapters.items():
            beats = story.plan.get(title, "")
            findings.extend(bible_consistency.check(story.world_bible, title, prose).findings)
            findings.extend(comprehension.check(title, beats, story.world_bible, prose).findings)

    return [_as_dict(f) for f in findings]
