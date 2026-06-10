"""Premise-fidelity gate.

A coherent, well-written manuscript can still quietly drift off the premise the
user actually asked for. This gate asks: does the finished story *dramatize*
the core idea — not mention it, but build scenes and stakes on it? For the
"mages die young" premise that means the magic-as-blood-transaction, the
aging-as-cost, and the old-start-wars-the-young-fight theme should be load
bearing, not set dressing.

It runs once, post-generation, over the spine, premise, and the rolling
summary (not the full prose — the summary already distils what happened), and
returns gaps as warnings: this is an advisory quality signal, never a hard
gate that would auto-reject a finished novel on one judge's call.
"""
from __future__ import annotations

import logging

import dspy
from pydantic import BaseModel, Field

from evals.types import EvalFinding, GateReport

logger = logging.getLogger(__name__)


class PremiseGap(BaseModel):
    element: str = Field(description="The premise element that is under-dramatized")
    severity: str = Field(description="missing (never realized) or weak (mentioned but not dramatized)")
    explanation: str
    fix: str = Field(description="Where in the arc to strengthen it")


class JudgePremiseFidelity(dspy.Signature):
    """Decide whether the manuscript actually dramatizes the user's premise.

    Break the premise into its load-bearing elements. For each, judge whether
    the story builds real scenes, choices, and consequences on it (dramatized)
    or merely references it (weak) or never delivers it (missing). Reward
    dramatization, not lip service. Return gaps only; an empty list means the
    premise is fully realized. Also return a 1-5 fidelity score.
    """

    story_idea: str = dspy.InputField()
    core_premise: str = dspy.InputField()
    spine: str = dspy.InputField()
    story_summary: str = dspy.InputField(
        desc="Representative prose excerpts from across the chapters (openings/key "
        "scenes) — judge what the PROSE dramatizes, not how terse this digest is"
    )
    fidelity_score: int = dspy.OutputField(desc="1 (off-premise) to 5 (fully dramatizes the idea)")
    gaps: list[PremiseGap] = dspy.OutputField()


def _judge(idea: str, premise: str, spine: str, summary: str):
    j = dspy.ChainOfThought(JudgePremiseFidelity)
    return j(story_idea=idea, core_premise=premise, spine=spine, story_summary=summary)


def check(idea: str, premise: str, spine: str, summary: str) -> GateReport:
    """Run premise fidelity once over the finished story's summary."""
    report = GateReport(gate="premise_fidelity")
    try:
        result = _judge(idea, premise, spine, summary)
    except Exception as exc:
        logger.warning("premise_fidelity judge failed: %s", exc)
        return report
    score = getattr(result, "fidelity_score", None)
    for g in (result.gaps or []):
        severity = "warn" if g.severity.strip().lower() == "missing" else "info"
        report.findings.append(
            EvalFinding(
                check="premise_fidelity",
                severity=severity,
                message=f"{g.element} ({g.severity}): {g.explanation}",
                fix=g.fix,
                locus=g.element,
            )
        )
    report.findings.insert(
        0,
        EvalFinding(
            check="premise_fidelity",
            severity="info",
            message=f"Premise fidelity score: {score}/5",
        ),
    )
    logger.info("premise_fidelity: score=%s gaps=%d", score, len(report.findings) - 1)
    return report
