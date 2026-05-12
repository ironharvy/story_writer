"""Post-generation lint pass for chapter prose.

The drafter sometimes leaves token-level errors in chapter prose that the world
bible already disambiguates:

- Placeholder phrases such as "the protagonist" / "the child" used before the
  protagonist's name has been established (variant A, chapter 1).
- Misspellings of canonical character names (e.g. "Cindar" for "Cinder") that
  some models produce inconsistently.

This module asks the configured DSPy LM to read the world bible's Characters
section together with the chapter prose, and to return a structured list of
verbatim find/replace pairs. The replacements are then applied only to chapter
prose (the body under the ``## Final Story`` heading); the world bible, plan,
and spine are left untouched.

Higher-level callers should drive this via :mod:`scripts.lint_story`, which
writes a ``.replacements.json`` sidecar next to the rewritten story.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from qa import _section, split_chapters

logger = logging.getLogger(__name__)


@dataclass
class Replacement:
    """A single verbatim find/replace edit proposed by the linter."""

    find: str
    replace: str
    reason: str = ""


def _extract_inputs(story_md: str) -> tuple[str, str]:
    """Return (characters_section, joined_chapter_prose)."""
    characters = _section(story_md, 3, "Characters")
    chapters = split_chapters(story_md)
    joined = "\n\n".join(
        f"### {title}\n\n{body}" for title, body in chapters.items()
    )
    return characters, joined


def _coerce(item: object) -> Replacement | None:
    """Coerce one DSPy output item into a ``Replacement``.

    DSPy may return dicts or pydantic-like objects depending on how the LM
    structured-output adapter resolves ``list[Replacement]``.
    """
    if isinstance(item, Replacement):
        return item
    if isinstance(item, dict):
        find = item.get("find") or ""
        replace = item.get("replace") or ""
        reason = item.get("reason") or ""
        if find and replace:
            return Replacement(find=find, replace=replace, reason=reason)
        return None
    find = getattr(item, "find", None)
    replace = getattr(item, "replace", None)
    reason = getattr(item, "reason", "") or ""
    if find and replace:
        return Replacement(find=find, replace=replace, reason=reason)
    return None


def lint(story_md: str) -> list[Replacement]:
    """Ask the configured DSPy LM for token-level replacements.

    Raises whatever the underlying DSPy call raises — we deliberately do not
    swallow errors so model misbehaviour stays visible.
    """
    import dspy

    characters, joined_prose = _extract_inputs(story_md)
    if not characters.strip():
        logger.warning("lint: '### Characters' section is empty; skipping")
        return []
    if not joined_prose.strip():
        logger.warning("lint: no chapter prose found; skipping")
        return []

    class LintReplacements(dspy.Signature):
        """Identify token-level errors in chapter prose that should be replaced.

        You are given the ``### Characters`` section of the world bible (the
        canonical names + descriptions of every character) and the full chapter
        prose of a story. Return a list of verbatim find/replace pairs to apply
        to the chapter prose.

        Propose a replacement for:
        - Placeholder phrases such as "the protagonist", "The protagonist",
          "the child", "the hunter" used where a specific established character
          from the Characters section is clearly meant. Use the character's
          canonical name as the replacement.
        - Misspellings or inconsistent spellings of canonical character names
          (e.g. "Cindar" instead of "Cinder").
        - Inconsistent capitalisation of canonical names where the canonical
          form is clearly intended.

        Do NOT propose a replacement when:
        - The change would be stylistic or tonal rather than a token-level
          error.
        - The placeholder genuinely refers to an unnamed minor character.
        - The ``find`` string does not appear verbatim in the chapter prose.
        - The canonical and the alternate spelling are both intentional
          aliases established in the world bible.

        Rules for each replacement:
        - ``find`` must appear verbatim (case-sensitive) in the chapter prose.
          If the same placeholder occurs in two cases ("the protagonist" and
          "The protagonist"), emit a separate entry for each.
        - ``replace`` must be the correct canonical form.
        - ``reason`` is a short human-readable justification (one phrase).

        Return an empty list if no token-level errors are found.
        """

        world_bible_characters: str = dspy.InputField(
            desc="The '### Characters' section of the world bible, listing canonical names."
        )
        chapter_prose: str = dspy.InputField(
            desc="The full chapter prose, with '### Chapter N: Title' headings preserved."
        )
        replacements: list[Replacement] = dspy.OutputField(
            desc="Verbatim find/replace pairs to apply to chapter prose only."
        )

    module = dspy.ChainOfThought(LintReplacements)
    result = module(
        world_bible_characters=characters,
        chapter_prose=joined_prose,
    )
    raw = result.replacements or []
    out: list[Replacement] = []
    for item in raw:
        coerced = _coerce(item)
        if coerced is None:
            logger.warning("lint: dropping malformed replacement: %r", item)
            continue
        out.append(coerced)
    return out


def validate(
    replacements: Iterable[Replacement],
    chapter_prose: str,
) -> list[Replacement]:
    """Drop replacements that don't appear in chapter prose or are trivial."""
    out: list[Replacement] = []
    seen: set[tuple[str, str]] = set()
    for r in replacements:
        if not r.find or not r.replace:
            logger.warning("validate: dropping empty replacement: %r", r)
            continue
        if r.find == r.replace:
            continue
        if r.find not in chapter_prose:
            logger.warning(
                "validate: dropping replacement (find not in prose): %r", r
            )
            continue
        key = (r.find, r.replace)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _chapter_region(story_md: str) -> tuple[int, int] | None:
    """Locate the chapter prose region in ``story_md``.

    The region starts after the ``## Final Story`` line and ends at the next
    ``## `` heading (or end of file). Returns ``None`` when no such region
    exists.
    """
    final_heading_re = re.compile(r"^## Final Story\s*$", re.MULTILINE)
    m = final_heading_re.search(story_md)
    if not m:
        return None
    start = m.end()
    next_h2 = re.compile(r"^## (?!#)", re.MULTILINE)
    nxt = next_h2.search(story_md, start)
    end = nxt.start() if nxt else len(story_md)
    return start, end


def _replacement_pattern(find: str) -> re.Pattern[str]:
    """Word-boundary aware pattern for ``find``.

    Anchors with ``\\b`` only on sides whose terminal character is alphanumeric
    (so phrases like ``the protagonist`` get word-bounded matches but a leading
    quote or punctuation isn't required for matching).
    """
    escaped = re.escape(find)
    if find and find[0].isalnum():
        escaped = r"\b" + escaped
    if find and find[-1].isalnum():
        escaped = escaped + r"\b"
    return re.compile(escaped)


def apply(
    story_md: str,
    replacements: Iterable[Replacement],
) -> tuple[str, list[tuple[Replacement, int]]]:
    """Apply ``replacements`` to chapter prose only.

    Returns the rewritten markdown and per-replacement substitution counts.
    Replacements whose ``find`` doesn't occur in the chapter region produce a
    count of 0 (they are left in the returned list for the sidecar).
    """
    replacements = list(replacements)
    if not replacements:
        return story_md, []

    region = _chapter_region(story_md)
    if region is None:
        logger.warning("apply: '## Final Story' heading not found; nothing rewritten")
        return story_md, [(r, 0) for r in replacements]

    start, end = region
    before = story_md[:start]
    middle = story_md[start:end]
    after = story_md[end:]

    counts: list[tuple[Replacement, int]] = []
    for r in replacements:
        pat = _replacement_pattern(r.find)
        middle, n = pat.subn(r.replace, middle)
        counts.append((r, n))

    return before + middle + after, counts


def lint_and_apply(
    story_md: str,
) -> tuple[str, list[Replacement], list[tuple[Replacement, int]]]:
    """Run the linter, validate the output, and apply it.

    Returns ``(rewritten_md, validated_replacements, counts)``.
    """
    proposed = lint(story_md)
    _, joined_prose = _extract_inputs(story_md)
    validated = validate(proposed, joined_prose)
    rewritten, counts = apply(story_md, validated)
    return rewritten, validated, counts


def replacements_to_json(
    replacements: Iterable[Replacement],
    counts: Iterable[tuple[Replacement, int]] | None = None,
) -> str:
    """Render replacements as JSON. Includes ``applied`` count when provided."""
    if counts is None:
        payload = [asdict(r) for r in replacements]
    else:
        payload = [{**asdict(r), "applied": n} for r, n in counts]
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
