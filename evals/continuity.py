"""Cross-chapter continuity gate.

A 50k-word manuscript does not fit in one context window, so we can't hand the
whole thing to a judge and ask "what contradicts what". Instead we extract a
compact, structured *digest* per chapter (who was present, who died or left,
what changed, where it happened, any hard factual claims) and then judge the
*sequence of digests* for contradictions: a character dead in chapter 3 acting
in chapter 5, a trait or relationship that flips, a timeline that doesn't add
up, two distinct characters collapsed into one name, a location that teleports.

Digests are small, so the contradiction judge sees the whole spine of facts at
once even when the prose never could.
"""
from __future__ import annotations

import logging

import dspy
from pydantic import BaseModel, Field

from evals.types import EvalFinding, GateReport

logger = logging.getLogger(__name__)

_SEVERITY = {"hard": "fail", "contradiction": "fail", "minor": "warn", "wobble": "warn"}


class ChapterDigest(BaseModel):
    characters_present: list[str] = Field(default_factory=list)
    deaths_or_exits: list[str] = Field(
        default_factory=list, description="Characters who die, leave, or are incapacitated this chapter"
    )
    key_events: list[str] = Field(default_factory=list, description="Plot-load-bearing events, terse")
    locations: list[str] = Field(default_factory=list)
    hard_facts: list[str] = Field(
        default_factory=list,
        description="Stated facts that must stay true later: ranks, kinships, who-owns-what. "
        "For ages and any life-force/time quantity, label which quantity it is — these are "
        "DISTINCT and must never be merged: chronological age, apparent age (how old someone "
        "looks), hoarded/remaining life-force reserve, and years already spent. E.g. "
        "'Kael: chronological 19, looks 17, hoarded reserve ~40 years'. A reserve that "
        "shrinks as it is spent is expected, not a fact that flipped.",
    )
    reconciled: list[str] = Field(
        default_factory=list,
        description="Apparent discrepancies the prose itself explicitly explains (e.g. 'the "
        "3000-year figure is the Ledger's lie; his true age is ~300'), so later checks don't "
        "re-flag them as contradictions.",
    )


class DigestChapter(dspy.Signature):
    """Extract a terse, factual digest of one chapter for continuity tracking.
    Record only what the prose states; do not invent. Keep each item short."""

    chapter_title: str = dspy.InputField()
    chapter_prose: str = dspy.InputField()
    digest: ChapterDigest = dspy.OutputField()


class Contradiction(BaseModel):
    chapters: list[str] = Field(description="Titles of the chapters in tension")
    kind: str = Field(description="death, trait, relationship, timeline, identity, location, or other")
    severity: str = Field(description="hard (flat contradiction) or minor (wobble)")
    explanation: str
    fix: str = Field(description="Which chapter to change and how")


class FindContradictions(dspy.Signature):
    """Given an ordered list of per-chapter digests, list factual
    contradictions ACROSS chapters. Look for: a character who died/left
    reappearing as if present; an age, rank, kinship, or possession that flips;
    a timeline that can't be reconciled; two distinct characters sharing one
    name; a location that moves impossibly. Report only real cross-chapter
    contradictions — not within-chapter style issues, not mere absence. Empty
    list if the manuscript is internally consistent.

    Do NOT flag a difference the world's own rules explain. Apply these
    precision rules before reporting anything:
    - Distinguish the kinds of "years". Chronological age, apparent age,
      hoarded/remaining life-force reserve, and years-spent are DIFFERENT
      quantities. "Kael has forty years" (a reserve he can spend) does NOT
      contradict "Kael is nineteen" (his age). Only flag when the SAME kind is
      given two irreconcilable values.
    - A reserve or count that decreases over later chapters is expected
      (he spent it), not a flipped fact.
    - If a digest's ``reconciled`` notes show the prose itself explains an
      apparent discrepancy (e.g. an inflated age that is openly "the Ledger's
      lie" vs a true age), do NOT flag it — the text already resolved it.
    - Two clearly-distinct characters who merely have similar or shared names
      (e.g. Joren vs Jaren) are NOT an identity contradiction; at most note a
      'minor' naming-collision risk. Do not infer one died because a
      similarly-named character did.
    Reserve hard FAILs for unambiguous, unreconciled contradictions of the same
    quantity or fact.
    """

    digests: list[str] = dspy.InputField(desc="Ordered 'Chapter — digest' strings")
    contradictions: list[Contradiction] = dspy.OutputField()


def _digest_chapter(title: str, prose: str) -> ChapterDigest:
    d = dspy.ChainOfThought(DigestChapter)
    return d(chapter_title=title, chapter_prose=prose).digest


def _find(digests: list[str]) -> list[Contradiction]:
    f = dspy.ChainOfThought(FindContradictions)
    return list(f(digests=digests).contradictions or [])


def _render_digest(title: str, d: ChapterDigest) -> str:
    parts = [f"### {title}"]
    if d.characters_present:
        parts.append("present: " + ", ".join(d.characters_present))
    if d.deaths_or_exits:
        parts.append("died/left: " + ", ".join(d.deaths_or_exits))
    if d.locations:
        parts.append("where: " + ", ".join(d.locations))
    if d.key_events:
        parts.append("events: " + "; ".join(d.key_events))
    if d.hard_facts:
        parts.append("facts: " + "; ".join(d.hard_facts))
    if d.reconciled:
        parts.append("reconciled (do not re-flag): " + "; ".join(d.reconciled))
    return "\n".join(parts)


def check(chapters: dict[str, str]) -> GateReport:
    """Run continuity over an ordered ``{title: prose}`` mapping."""
    report = GateReport(gate="continuity")
    if len(chapters) < 2:
        return report
    rendered: list[str] = []
    for title, prose in chapters.items():
        try:
            digest = _digest_chapter(title, prose)
            rendered.append(_render_digest(title, digest))
        except Exception as exc:
            logger.warning("continuity digest failed for %r: %s", title, exc)
    if len(rendered) < 2:
        return report
    try:
        contradictions = _find(rendered)
    except Exception as exc:
        logger.warning("continuity judge failed: %s", exc)
        return report
    for c in contradictions:
        severity = _SEVERITY.get(c.severity.strip().lower(), "warn")
        report.findings.append(
            EvalFinding(
                check="continuity",
                severity=severity,
                message=f"{c.kind} across {', '.join(c.chapters)}: {c.explanation}",
                fix=c.fix,
                locus=", ".join(c.chapters),
            )
        )
    logger.info("continuity: %d contradiction(s) (%d blocking)", len(report.findings), len(report.blocking))
    return report
