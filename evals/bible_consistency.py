"""Bible-consistency gate.

Asks the model, chapter by chapter, whether the prose honours the world bible:
are the named locations described the way the bible defines them, do characters
behave in line with who the bible says they are, and — most importantly for a
hard-magic premise — does every act of magic pay the cost the rules demand?

The check is deliberately conservative: it only flags an issue when the prose
*contradicts* canon or *omits a cost the bible makes mandatory*, not when it
merely adds new colour. Adding fresh detail is good writing; contradicting the
bible is a continuity bug.
"""
from __future__ import annotations

import logging

import dspy
from pydantic import BaseModel, Field

from core.types import WorldBible
from evals.types import EvalFinding, GateReport

logger = logging.getLogger(__name__)

# Map the judge's own severity vocabulary onto the qa.py info/warn/fail scale.
_SEVERITY = {"contradiction": "fail", "drift": "warn", "omission": "warn", "minor": "info"}


class BibleIssue(BaseModel):
    entity: str = Field(description="The location, character, or rule involved")
    kind: str = Field(description="One of: location, character, rule, timeline")
    severity: str = Field(
        description="contradiction (prose flatly contradicts canon), "
        "drift (subtly inconsistent), omission (a mandatory cost/rule is "
        "ignored where the scene clearly invokes it), or minor"
    )
    explanation: str = Field(description="What the prose does vs. what the bible says")
    fix: str = Field(description="Concrete revision the drafter should make")


class JudgeBibleConsistency(dspy.Signature):
    """Compare one chapter's prose against the world bible and report only
    genuine inconsistencies.

    Flag an issue ONLY when the prose contradicts the bible, makes a named
    location/character behave against its canonical definition, or stages an
    act of magic without paying the cost the rules require. Do NOT flag the
    prose for adding new, non-contradictory detail, for paraphrasing the bible
    in fresh language, or for leaving canon facts unmentioned when the scene
    has no reason to mention them. Absence is not contradiction.

    If everything is consistent, return an empty list.
    """

    rules_of_the_world: list[str] = dspy.InputField(
        desc="Canonical rules — especially any cost/price the magic system exacts"
    )
    characters: list[str] = dspy.InputField(desc="Canonical character definitions")
    locations: list[str] = dspy.InputField(desc="Canonical location descriptions")
    chapter_title: str = dspy.InputField()
    chapter_prose: str = dspy.InputField()
    issues: list[BibleIssue] = dspy.OutputField(
        desc="Only genuine contradictions/omissions; empty if consistent"
    )


def _judge(world_bible: WorldBible, chapter_title: str, prose: str) -> list[BibleIssue]:
    """Isolated model call. Unit tests monkeypatch this."""
    judge = dspy.ChainOfThought(JudgeBibleConsistency)
    result = judge(
        rules_of_the_world=world_bible.rules_of_the_world,
        characters=world_bible.characters,
        locations=world_bible.locations,
        chapter_title=chapter_title,
        chapter_prose=prose,
    )
    return list(result.issues or [])


def check(world_bible: WorldBible, chapter_title: str, prose: str) -> GateReport:
    """Run the bible-consistency judge over one chapter and return findings."""
    report = GateReport(gate="bible_consistency")
    try:
        issues = _judge(world_bible, chapter_title, prose)
    except Exception as exc:  # judge failure must not crash drafting
        logger.warning("bible_consistency judge failed for %r: %s", chapter_title, exc)
        return report
    for issue in issues:
        severity = _SEVERITY.get(issue.severity.strip().lower(), "warn")
        report.findings.append(
            EvalFinding(
                check="bible_consistency",
                severity=severity,
                message=f"{issue.kind}/{issue.entity}: {issue.explanation}",
                fix=issue.fix,
                locus=issue.entity,
            )
        )
    logger.info(
        "bible_consistency %r: %d issue(s) (%d blocking)",
        chapter_title, len(report.findings), len(report.blocking),
    )
    return report
