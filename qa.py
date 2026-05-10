"""Post-generation quality checks for story markdown output.

Each check is a pure function: takes the path of a story `.md` artifact and
returns a list of `Finding` records. The CLI in `scripts/run_qa.py` runs all
checks and prints a comparative report.

The story markdown is expected to follow the layout produced by `mymain.py`:

    # Story
    ## ...                # generation params, idea, premise, spine
    ## World bible
    ### Rules of the World
    ### Locations
        - <bullet>, <description>
        ...
    #### Location 1
    #### Location 2
    ...
    ### Characters
        1. <name>, <description>
        ...
    #### Character 1
    ...
    ### Timeline
    ## Final Story
    ### Chapter 1: <title>
    ...
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

NGRAM = 5
WORD_RE = re.compile(r"[A-Za-z']+")
PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+(?:the\s+)?[A-Z][a-z]+){0,2})\b")


@dataclass
class Finding:
    check: str
    severity: str  # "info" | "warn" | "fail"
    message: str


# --- section parsing ---------------------------------------------------------

def _read(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _section(text: str, heading_level: int, heading: str) -> str:
    """Return the body under `<level> heading`, up to the next heading at the
    same level (`## ...` for level 2, `### ...` for level 3)."""
    prefix = "#" * heading_level + " "
    same_level_re = re.compile(rf"^#{{{heading_level}}} (?!#)")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == f"{prefix}{heading}":
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if same_level_re.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end])


def split_chapters(text: str) -> dict[str, str]:
    final = _section(text, 2, "Final Story")
    chapters: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in final.splitlines():
        if line.startswith("### "):
            if current is not None:
                chapters[current] = "\n".join(buf).strip()
            current = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    if current is not None:
        chapters[current] = "\n".join(buf).strip()
    return chapters


def character_bullets(text: str) -> list[str]:
    """Return the list of `<name>, <desc>` bullets from `### Characters`.
    Only the first numbered list is read; enhanced `#### Character N` blocks
    are skipped."""
    section = _section(text, 3, "Characters")
    bullets: list[str] = []
    for line in section.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)", line)
        if m:
            bullets.append(m.group(1).strip())
    return bullets


_NAME_DELIM_RE = re.compile(r"[,:;—–\-]")
_GENERIC_FIRST_TOKENS = {"The", "A", "An"}


def canonical_name(bullet: str) -> str:
    """Return the head of a `Name <delim> description` bullet.

    Tries the first comma, colon, semicolon, em-dash, en-dash, or hyphen — whichever
    comes first."""
    m = _NAME_DELIM_RE.search(bullet)
    head = bullet[: m.start()] if m else bullet
    return head.strip()


# --- check 1: cross-chapter phrase reuse -------------------------------------

def _words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def check_cross_chapter_phrase_reuse(text: str, n: int = NGRAM, min_chapters: int = 2) -> list[Finding]:
    """Detect n-gram phrases reused verbatim across multiple chapters.

    Excludes phrases that appear in the World bible (those are caught by the
    bible-leakage check)."""
    chapters = split_chapters(text)
    bible_grams = _ngrams(_words(_section(text, 2, "World bible")), n)

    by_chapter: dict[str, set[tuple[str, ...]]] = {
        title: _ngrams(_words(body), n) for title, body in chapters.items()
    }

    counter: Counter[tuple[str, ...]] = Counter()
    for grams in by_chapter.values():
        for g in grams:
            counter[g] += 1

    findings: list[Finding] = []
    repeated = [
        (g, c) for g, c in counter.items()
        if c >= min_chapters and g not in bible_grams
    ]
    repeated.sort(key=lambda x: -x[1])
    for gram, count in repeated[:10]:
        findings.append(Finding(
            check="cross_chapter_phrase_reuse",
            severity="warn",
            message=f"x{count} chapters: '{' '.join(gram)}'",
        ))
    findings.append(Finding(
        check="cross_chapter_phrase_reuse",
        severity="info",
        message=f"total repeated {n}-grams across {len(chapters)} chapters: {len(repeated)}",
    ))
    return findings


# --- check 2: name drift -----------------------------------------------------

def _proper_nouns(body: str) -> Counter[str]:
    """Return counts of capitalized 1-3 word sequences in `body`."""
    counts: Counter[str] = Counter()
    for m in PROPER_NOUN_RE.finditer(body):
        counts[m.group(1)] += 1
    return counts


def check_name_drift(text: str) -> list[Finding]:
    """Flag chapter-prose names that look like drifted variants of canonical
    character names (same first token, different full name)."""
    bullets = character_bullets(text)
    canonical = [canonical_name(b) for b in bullets]
    canonical_set = set(canonical)
    # Skip generic first tokens like "The"/"A"/"An" so that a canonical
    # like "The Young Mage" doesn't mark every "The X" proper noun in
    # chapters as drift.
    canonical_first_tokens = {
        n.split()[0] for n in canonical
        if n and n.split()[0] not in _GENERIC_FIRST_TOKENS
    }

    chapters = split_chapters(text)
    chapter_prose = "\n\n".join(chapters.values())
    seen = _proper_nouns(chapter_prose)

    findings: list[Finding] = []
    drifts: dict[str, set[str]] = {}
    for proper, count in seen.items():
        first = proper.split()[0]
        if first not in canonical_first_tokens:
            continue
        if proper in canonical_set:
            continue  # exact canonical match — fine
        # Same first token, different full form — possible drift
        # Skip cases where the chapter form is a strict prefix of the canonical
        # (e.g. "Kaelen" alone when canonical is "Kaelen Vey") — that's a
        # legitimate first-name reference, not drift.
        canonical_with_same_first = [n for n in canonical if n.split()[0] == first]
        is_strict_prefix = any(
            c == proper or c.startswith(proper + " ") for c in canonical_with_same_first
        )
        if is_strict_prefix:
            continue
        drifts.setdefault(first, set()).add(proper)

    for first, variants in drifts.items():
        canonical_form = next(n for n in canonical if n.split()[0] == first)
        findings.append(Finding(
            check="name_drift",
            severity="fail",
            message=f"canonical='{canonical_form}' but chapters also use: {sorted(variants)}",
        ))

    if not drifts:
        findings.append(Finding(
            check="name_drift",
            severity="info",
            message=f"no drift detected ({len(canonical)} canonical names)",
        ))
    return findings


# --- check 3: character presence --------------------------------------------

def check_character_presence(text: str) -> list[Finding]:
    """Each canonical character must appear in chapter prose at least once."""
    bullets = character_bullets(text)
    canonical = [canonical_name(b) for b in bullets]
    chapter_prose = "\n\n".join(split_chapters(text).values())

    findings: list[Finding] = []
    missing: list[str] = []
    for name in canonical:
        # Match full name OR first token (e.g. "Kaelen" alone is OK for
        # "Kaelen Vey").
        first = name.split()[0]
        if name in chapter_prose or re.search(rf"\b{re.escape(first)}\b", chapter_prose):
            continue
        missing.append(name)

    for name in missing:
        findings.append(Finding(
            check="character_presence",
            severity="fail",
            message=f"character never appears in chapters: '{name}'",
        ))
    if not missing:
        findings.append(Finding(
            check="character_presence",
            severity="info",
            message=f"all {len(canonical)} characters appear in chapters",
        ))
    return findings


# --- runner ------------------------------------------------------------------

CHECKS = [
    check_cross_chapter_phrase_reuse,
    check_name_drift,
    check_character_presence,
]


def run_all(path: Path) -> list[Finding]:
    text = _read(path)
    out: list[Finding] = []
    for check in CHECKS:
        out.extend(check(text))
    return out
