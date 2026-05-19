"""LLM-backed point-of-view consistency check.

The heuristic checks in :mod:`qa` work on the text alone. Deciding whether a
chapter is narrated in first or third person reliably needs a model: quoted
dialogue ("She said, 'I am tired.'"), letters/journals embedded in the prose,
and close internal monologue all confound a plain pronoun count.

A story that is *entirely* first-person — or entirely third-person — is fine.
The failure mode this check looks for is chapters whose narration disagrees
with the rest of the book, or chapters that shift POV within themselves. (In
the 2026-05-10 bake-off, variant C at 27B narrated chapters 1–4 in third
person, chapter 5 fully in first person, then slipped to "the protagonist" in
chapter 6 — none of which `qa.check_name_drift` could see.)

Findings reuse :class:`qa.Finding`, so this check can be folded into a unified
QA report later; for now it ships with its own CLI (:mod:`scripts.check_pov`),
which keeps :mod:`qa` free of any DSPy dependency.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from qa import Finding, split_chapters

logger = logging.getLogger(__name__)

DECISIVE_POVS = ("first_person", "third_person")
POV_LABELS = (*DECISIVE_POVS, "mixed", "other")


@dataclass
class ChapterPOV:
    """A chapter's classified dominant narrative point of view."""

    chapter: str
    pov: str  # one of POV_LABELS
    note: str = ""


def _format_chapters(chapters: dict[str, str]) -> str:
    return "\n\n".join(f"### {title}\n\n{body}" for title, body in chapters.items())


def _coerce(item: object) -> ChapterPOV | None:
    """Coerce one DSPy output item into a ``ChapterPOV``."""
    if isinstance(item, ChapterPOV):
        chapter, pov, note = item.chapter, item.pov, item.note
    elif isinstance(item, dict):
        chapter = item.get("chapter") or ""
        pov = item.get("pov") or ""
        note = item.get("note") or ""
    else:
        chapter = getattr(item, "chapter", None) or ""
        pov = getattr(item, "pov", None) or ""
        note = getattr(item, "note", "") or ""
    chapter = str(chapter).strip()
    pov = str(pov).strip().lower().replace(" ", "_").replace("-", "_")
    if not chapter:
        return None
    if pov not in POV_LABELS:
        logger.warning("pov_check: unknown POV label %r for %r; treating as 'other'", pov, chapter)
        pov = "other"
    return ChapterPOV(chapter=chapter, pov=pov, note=str(note).strip())


def classify(chapters: dict[str, str]) -> list[ChapterPOV]:
    """Ask the configured DSPy LM to classify each chapter's dominant POV.

    Raises whatever the underlying DSPy call raises — failures stay visible.
    """
    import dspy

    if not chapters:
        return []

    class ClassifyChapterPOV(dspy.Signature):
        """Classify the dominant narrative point of view of each chapter.

        You are given the chapters of one story, each introduced by a
        ``### <label>`` heading followed by its prose. For each chapter,
        determine the dominant point of view of the *narration only* — ignore
        the POV inside quoted dialogue, letters, journal entries, or other
        characters' reported speech; only the narrator's own voice counts:

        - ``first_person``: the narrator refers to themselves as "I" / "we".
        - ``third_person``: the narrator refers to the characters by name and
          as "he" / "she" / "they".
        - ``mixed``: the chapter's narration is itself inconsistent — it
          shifts between first and third person within the chapter.
        - ``other``: second-person narration, a framing that fits none of the
          above, or a chapter too short or empty to tell.

        Return exactly one classification per input chapter, echoing the
        chapter's label verbatim. ``note`` is a short justification (one
        phrase) and may be empty.
        """

        chapters_text: str = dspy.InputField(
            desc="The story's chapters, each as '### <label>' then its prose."
        )
        classifications: list[ChapterPOV] = dspy.OutputField(
            desc="One {chapter, pov, note} per input chapter; pov in "
            "{first_person, third_person, mixed, other}."
        )

    module = dspy.ChainOfThought(ClassifyChapterPOV)
    result = module(chapters_text=_format_chapters(chapters))
    raw = result.classifications or []
    out: list[ChapterPOV] = []
    for item in raw:
        coerced = _coerce(item)
        if coerced is None:
            logger.warning("pov_check: dropping malformed classification: %r", item)
            continue
        out.append(coerced)
    return out


def check_pov_consistency(
    chapters: dict[str, str],
    classifications: Iterable[ChapterPOV] | None = None,
) -> list[Finding]:
    """Flag chapters whose narrative POV disagrees with the dominant POV.

    ``classifications`` is exposed for testing; when ``None`` it is obtained
    via :func:`classify` (which calls the LM).
    """
    if not chapters:
        return []
    classifications = (
        list(classify(chapters)) if classifications is None else list(classifications)
    )
    if not classifications:
        return [Finding(
            check="pov_consistency",
            severity="warn",
            message="POV check returned no classifications",
        )]

    by_chapter = {c.chapter: c for c in classifications}
    decisive = [c.pov for c in classifications if c.pov in DECISIVE_POVS]
    if not decisive:
        return [Finding(
            check="pov_consistency",
            severity="info",
            message="POV undetermined for all chapters",
        )]

    dominant, _ = Counter(decisive).most_common(1)[0]

    findings: list[Finding] = []
    outliers = 0
    for title in chapters:
        c = by_chapter.get(title)
        if c is None:
            findings.append(Finding(
                check="pov_consistency",
                severity="warn",
                message=f"chapter '{title}' was not classified by the POV check",
            ))
            continue
        suffix = f" ({c.note})" if c.note else ""
        if c.pov == "mixed":
            outliers += 1
            findings.append(Finding(
                check="pov_consistency",
                severity="fail",
                message=f"chapter '{title}' shifts narrative POV within the chapter{suffix}",
            ))
        elif c.pov in DECISIVE_POVS and c.pov != dominant:
            outliers += 1
            findings.append(Finding(
                check="pov_consistency",
                severity="fail",
                message=(
                    f"chapter '{title}' is narrated {c.pov} but the story is "
                    f"dominantly {dominant}{suffix}"
                ),
            ))
        # "other" chapters (too short, second person, …) are not flagged.

    if outliers == 0:
        findings.append(Finding(
            check="pov_consistency",
            severity="info",
            message=f"all classifiable chapters are {dominant}",
        ))
    return findings


def run(path: Path) -> list[Finding]:
    """Run the POV consistency check on a story markdown file."""
    text = Path(path).read_text(encoding="utf-8")
    return check_pov_consistency(split_chapters(text))
