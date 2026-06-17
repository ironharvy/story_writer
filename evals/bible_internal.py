"""Bible self-consistency gate.

The world bible is assembled piecemeal — rules, then locations, then a
timeline, then characters, each character/location *enhanced* in its own
isolated model call. Nothing forces those calls to agree with one another, so
the bible can contradict itself before a single chapter is written: two
siblings given different surnames, a character whose stated age can't fit the
timeline, a location that violates a rule, a role assigned to two people.

This gate reads the finished bible and reports those internal contradictions.
Crucially it also emits *canon clarifications* — short authoritative rulings
("The siblings share the surname Thorne") — that the drafter is handed so every
chapter inherits one consistent canon instead of re-litigating the conflict.
Running this once, before drafting, is far cheaper than catching the same drift
chapter by chapter.
"""
from __future__ import annotations

import logging

import dspy
from pydantic import BaseModel, Field

from core.types import WorldBible
from evals.types import EvalFinding, GateReport

logger = logging.getLogger(__name__)

_SEVERITY = {"contradiction": "fail", "tension": "warn", "minor": "info"}


class InternalIssue(BaseModel):
    entities: list[str] = Field(description="The bible entries in conflict")
    kind: str = Field(description="naming, age, role, timeline, location, rule, or relationship")
    severity: str = Field(description="contradiction, tension, or minor")
    explanation: str
    resolution: str = Field(
        description="The single authoritative ruling that resolves the conflict, "
        "stated as canon (e.g. 'Both siblings use the surname Thorne')"
    )


class CheckBibleInternalConsistency(dspy.Signature):
    """Read a world bible and find places where it contradicts ITSELF.

    Look across the rules, characters, locations, and timeline for: characters
    related as kin but given mismatched surnames; ages or appearances that
    can't be reconciled with the timeline; a role or title assigned to two
    different characters; a location whose description breaks one of the rules;
    relationships described inconsistently between two characters' entries.

    Report only genuine internal contradictions, not stylistic variation or a
    detail simply being absent from one entry. For each, give a single
    authoritative resolution stated as canon. Empty list if the bible is
    internally consistent.
    """

    rules_of_the_world: list[str] = dspy.InputField()
    characters: list[str] = dspy.InputField()
    locations: list[str] = dspy.InputField()
    timeline: list[str] = dspy.InputField()
    issues: list[InternalIssue] = dspy.OutputField()


def _judge(world_bible: WorldBible) -> list[InternalIssue]:
    judge = dspy.ChainOfThought(CheckBibleInternalConsistency)
    result = judge(
        rules_of_the_world=world_bible.rules_of_the_world,
        characters=world_bible.characters,
        locations=world_bible.locations,
        timeline=world_bible.timeline,
    )
    return list(result.issues or [])


def check(world_bible: WorldBible) -> tuple[GateReport, list[str]]:
    """Run the bible self-consistency judge.

    Returns the report plus the list of canon clarifications (deduped, in
    order) to hand the drafter so chapters inherit one resolved canon.
    """
    report = GateReport(gate="bible_internal")
    clarifications: list[str] = []
    try:
        issues = _judge(world_bible)
    except Exception as exc:
        logger.warning("bible_internal judge failed: %s", exc)
        return report, clarifications
    for issue in issues:
        severity = _SEVERITY.get(issue.severity.strip().lower(), "warn")
        report.findings.append(
            EvalFinding(
                check="bible_internal",
                severity=severity,
                message=f"{issue.kind} between {', '.join(issue.entities)}: {issue.explanation}",
                fix=issue.resolution,
                locus=", ".join(issue.entities),
            )
        )
        res = issue.resolution.strip()
        if res and res not in clarifications:
            clarifications.append(res)
    logger.info(
        "bible_internal: %d issue(s), %d clarification(s)",
        len(report.findings), len(clarifications),
    )
    return report, clarifications
