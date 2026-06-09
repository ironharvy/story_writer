"""Generation-time self-correction for the cheap deterministic canon constraints.

DSPy 3.x removed ``dspy.Assert`` / ``dspy.Suggest``, so this is the brief's
fallback: a small critic→revise submodule. The **critic**
(:func:`chapter_violations`) is a pure function — it runs the deterministic
canon metrics over a single drafted chapter and returns the hard violations
(scaffolding leak, a not-yet-allowed name, a locked-appearance contradiction, a
missing required artifact, an impossible year-span). The **revise** step
(:func:`revise_chapter`) feeds those violations back to the LM as feedback and
asks for a corrected draft.

:func:`draft_with_canon_guard` ties them together: draft, critique, and if the
draft violates a constraint, revise with the violation as feedback — up to
``max_revisions`` times. The critic is fully testable without an LM.
"""
from __future__ import annotations

import logging
from typing import Callable

from canon_metrics import (
    check_canon_fact,
    check_name_arc,
    check_required_artifact,
    check_scaffolding_leak,
    check_timeline_age,
)
from qa import Finding

logger = logging.getLogger(__name__)

# The subset of canon metrics cheap and deterministic enough to self-correct on.
_GUARD_CHECKS = (
    check_scaffolding_leak,
    check_name_arc,
    check_canon_fact,
    check_required_artifact,
    check_timeline_age,
)


def _wrap_chapter(chapter_number: int, chapter_title: str, prose: str) -> str:
    """Wrap a single chapter's prose in the minimal markdown the metrics parse."""
    return f"## Final Story\n\n### Chapter {chapter_number}: {chapter_title}\n\n{prose}\n"


def chapter_violations(
    prose: str, canon, chapter_number: int, chapter_title: str = "Untitled"
) -> list[Finding]:
    """Return the hard (severity=fail) canon violations in one chapter's prose."""
    text = _wrap_chapter(chapter_number, chapter_title, prose)
    out: list[Finding] = []
    for check in _GUARD_CHECKS:
        out.extend(f for f in check(text, canon) if f.severity == "fail")
    return out


def format_feedback(violations: list[Finding]) -> str:
    """Render violations as revise-step feedback."""
    lines = ["The draft violates these canon constraints; fix every one:"]
    lines += [f"- {f.message}" for f in violations]
    return "\n".join(lines)


def revise_chapter(prose: str, feedback: str, canon_directives: str = "") -> str:
    """Ask the configured LM to rewrite the prose so it satisfies the constraints.

    Kept verbatim where possible: the revise must only fix the listed violations,
    not rewrite the chapter wholesale.
    """
    import dspy

    class ReviseChapterForCanon(dspy.Signature):
        """Rewrite the chapter prose so it satisfies the canon constraints.

        You are given a drafted chapter, the canon directives in force, and a
        list of specific constraint violations. Fix every listed violation while
        preserving the chapter's events, tone, and length as much as possible —
        change only what is needed (a contradicting detail, a not-yet-allowed
        name, leaked planning vocabulary, a missing required element). Return the
        full corrected chapter prose, nothing else.
        """

        canon_directives: str = dspy.InputField(desc="Canon rules in force this chapter")
        violations: str = dspy.InputField(desc="The specific violations to fix")
        prose: str = dspy.InputField(desc="The drafted chapter prose")
        revised_prose: str = dspy.OutputField(desc="Corrected chapter prose")

    revise = dspy.ChainOfThought(ReviseChapterForCanon)
    return revise(
        canon_directives=canon_directives,
        violations=feedback,
        prose=prose,
    ).revised_prose


def draft_with_canon_guard(
    draft_fn: Callable[[], str],
    canon,
    chapter_number: int,
    chapter_title: str = "Untitled",
    canon_directives: str = "",
    max_revisions: int = 2,
) -> str:
    """Draft a chapter, then critique-and-revise until it satisfies the cheap
    deterministic canon constraints (or ``max_revisions`` is exhausted).

    ``draft_fn`` is a zero-arg callable returning the initial prose. With an
    empty canon there are no constraints, so the draft is returned unchanged.
    """
    prose = draft_fn()
    for attempt in range(1, max_revisions + 1):
        violations = chapter_violations(prose, canon, chapter_number, chapter_title)
        if not violations:
            return prose
        logger.info(
            "canon_guard ch.%d attempt %d: %d violation(s); revising",
            chapter_number, attempt, len(violations),
        )
        try:
            prose = revise_chapter(prose, format_feedback(violations), canon_directives)
        except Exception as exc:  # an LM/revise failure must not kill the run
            logger.warning("canon_guard ch.%d revise failed: %s", chapter_number, exc)
            return prose
    residual = chapter_violations(prose, canon, chapter_number, chapter_title)
    if residual:
        logger.warning(
            "canon_guard ch.%d: %d violation(s) remain after %d revision(s)",
            chapter_number, len(residual), max_revisions,
        )
    return prose
