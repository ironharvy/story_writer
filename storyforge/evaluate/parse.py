"""Parse a manuscript Markdown file into title / premise / cast / chapters.

Follows the conventions in STORY_EVAL_SPEC.md ("Expected manuscript shape"):
chapters split on the chapter-heading level; canonical cast read from the list
under the characters heading; canonical name = text before the first
`, : ; — – -` delimiter with surrounding * / _ emphasis stripped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Delimiters that terminate a canonical name (spec-defined, order-independent).
_NAME_DELIMS = (",", ":", ";", "—", "–", "-")
_EMPH = re.compile(r"^(?:\*\*|\*|__|_)+|(?:\*\*|\*|__|_)+$")
_FULLY_EMPH = re.compile(r"^(?:\*\*|\*|__|_).+?(?:\*\*|\*|__|_)$")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_CHAPTER_NAME = re.compile(r"\b(chapter|prologue|epilogue|interlude)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9']+")


@dataclass
class CastEntry:
    canonical: str
    first_token: str
    raw: str


@dataclass
class Chapter:
    label: str
    prose: str

    @property
    def word_count(self) -> int:
        return len(_WORD.findall(self.prose))


@dataclass
class Manuscript:
    title: str
    premise: str
    characters: list[CastEntry] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    world_text: str = ""  # premise + world/cast section, for 5-gram exclusion
    raw: str = ""

    @property
    def prose(self) -> str:
        """All chapter prose concatenated."""
        return "\n\n".join(c.prose for c in self.chapters)


def canonical_name(item: str) -> str | None:
    """Extract the canonical name from a cast list item, or None if it's not a name.

    Skips bullets whose entire content is bold/italic with no trailing description
    (e.g. ``**Reclaim Identity**`` — a motivation in the cast slot, not a name).
    """
    s = _LIST_MARKER.sub("", item.strip()).strip()
    if not s:
        return None
    idx = min((s.find(d) for d in _NAME_DELIMS if d in s), default=-1)
    if idx == -1:
        name_part, has_desc = s, False
    else:
        name_part, has_desc = s[:idx], bool(s[idx + 1:].strip())
    name_part = name_part.strip()
    if not has_desc and _FULLY_EMPH.match(name_part):
        return None  # motivation-as-name guard
    name = _EMPH.sub("", name_part).strip(" *_.\"'")
    return name or None


def _headings(lines: list[str]) -> list[tuple[int, int, str]]:
    """Return (line_index, level, text) for each ATX heading."""
    out = []
    for i, ln in enumerate(lines):
        m = _HEADING.match(ln)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


def _section_body(lines: list[str], start: int, headings: list[tuple[int, int, str]],
                  max_level: int) -> tuple[str, int]:
    """Body text from line start until the next heading of level <= max_level."""
    end = len(lines)
    for (idx, level, _t) in headings:
        if idx > start and level <= max_level:
            end = idx
            break
    return "\n".join(lines[start + 1:end]).strip(), end


def parse_manuscript(md: str) -> Manuscript:
    lines = md.splitlines()
    headings = _headings(lines)

    title = ""
    for (_i, level, text) in headings:
        if level == 1:
            title = text
            break

    # Locate level-2 sections by keyword.
    premise_text = ""
    world_text = ""
    cast_lines: list[str] = []
    manuscript_start: int | None = None
    for (idx, level, text) in headings:
        if level != 2:
            continue
        low = text.lower()
        if "premise" in low and not premise_text:
            premise_text, _ = _section_body(lines, idx, headings, 2)
        elif any(k in low for k in ("world", "character", "cast", "dramatis")):
            body, end = _section_body(lines, idx, headings, 2)
            world_text += "\n" + body
            cast_lines = lines[idx + 1:end]
        elif any(k in low for k in ("manuscript", "story", "the novel", "novel", "text")):
            if manuscript_start is None:
                manuscript_start = idx

    if not premise_text:
        # Fallback: first non-empty paragraph after the title.
        for ln in lines:
            s = ln.strip()
            if s and not s.startswith("#"):
                premise_text = s
                break

    # Parse cast entries from list items in the characters section.
    characters: list[CastEntry] = []
    seen = set()
    for ln in cast_lines:
        if not _LIST_MARKER.match(ln):
            continue
        name = canonical_name(ln)
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        characters.append(CastEntry(canonical=name, first_token=name.split()[0], raw=ln.strip()))

    # Chapters: prefer headings naming a chapter/prologue/epilogue; else level-3
    # headings after the manuscript section.
    named = [(idx, lvl, t) for (idx, lvl, t) in headings if _CHAPTER_NAME.search(t)]
    if named:
        chap_heads = named
    else:
        floor = manuscript_start if manuscript_start is not None else -1
        chap_heads = [(idx, lvl, t) for (idx, lvl, t) in headings if lvl == 3 and idx > floor]

    chapters: list[Chapter] = []
    for n, (idx, lvl, text) in enumerate(chap_heads):
        end = len(lines)
        for (idx2, lvl2, _t2) in headings:
            if idx2 > idx and lvl2 <= lvl:
                end = idx2
                break
        prose = "\n".join(lines[idx + 1:end]).strip()
        chapters.append(Chapter(label=text, prose=prose))

    world_text = (premise_text + "\n" + world_text).strip()
    return Manuscript(title=title, premise=premise_text, characters=characters,
                      chapters=chapters, world_text=world_text, raw=md)
