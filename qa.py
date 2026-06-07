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
CHAPTER_MIN_WORDS = 80
# Target word band for a chapter. Below the hard floor (CHAPTER_MIN_WORDS) a
# chapter FAILs as broken/truncated; within the floor but outside this band it
# WARNs as thin/bloated. This is the single source of truth for the length
# thresholds — see docs/07-acceptance-and-quality.md and bench/eval-spec.md.
CHAPTER_TARGET_BAND = (300, 2500)
WORD_RE = re.compile(r"[A-Za-z']+")
# Proper-noun matcher: 1–3 capitalised tokens joined by spaces/tabs only.
# Newlines are excluded so a paragraph break can't merge two separate names
# (e.g. "Renn\n\nThe Sanctum" was previously parsed as one entity).
PROPER_NOUN_RE = re.compile(r"\b([A-Z][a-z]+(?:[ \t]+(?:the[ \t]+)?[A-Z][a-z]+){0,2})\b")
# Bullets whose entire content is wrapped in markdown emphasis with no
# trailing description — some models use this slot for motivations/goals
# instead of character names.
_BOLD_ONLY_BULLET_RE = re.compile(r"\s*\*+[^*]+\*+\s*")


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
    are skipped. Bullets whose entire content is bold/italic with no
    trailing description are skipped — some models emit motivation lists
    ("**Reclaim Identity**") in this slot, which would otherwise be reported
    as missing characters."""
    section = _section(text, 3, "Characters")
    bullets: list[str] = []
    for line in section.splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+)", line)
        if not m:
            continue
        content = m.group(1).strip()
        if _BOLD_ONLY_BULLET_RE.fullmatch(content):
            continue
        bullets.append(content)
    return bullets


_NAME_DELIM_RE = re.compile(r"[,:;—–\-]")
_GENERIC_FIRST_TOKENS = {"The", "A", "An"}


def canonical_name(bullet: str) -> str:
    """Return the head of a `Name <delim> description` bullet.

    Tries the first comma, colon, semicolon, em-dash, en-dash, or hyphen — whichever
    comes first. Surrounding markdown emphasis (``**Cinder**`` → ``Cinder``) is
    stripped so a model that bolds the name doesn't produce a literal-asterisk
    canonical."""
    m = _NAME_DELIM_RE.search(bullet)
    head = bullet[: m.start()] if m else bullet
    head = head.strip()
    head = re.sub(r"^\*+|\*+$", "", head).strip()
    return head


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
        # Severity is "warn" rather than "fail": same-first-token matches catch
        # real misspellings (Cinder/Cindar) but also harmless paraphrases
        # ("High Cardinal Vane" ↔ "High Clergy") and WorldState parse
        # artifacts. The post-generation linter (scripts/lint_story.py) is the
        # tool that fixes the real misspellings.
        findings.append(Finding(
            check="name_drift",
            severity="warn",
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


# --- check 4: chapter length -------------------------------------------------

def check_chapter_length(text: str, min_words: int = CHAPTER_MIN_WORDS) -> list[Finding]:
    """Fail chapters whose prose is below `min_words`.

    Catches empty / truncated chapters that previously passed every check
    (the `runs/demo-hollow` regression had an empty chapter 3 that passed
    R1–R6 silently)."""
    chapters = split_chapters(text)
    findings: list[Finding] = []
    short: list[tuple[str, int]] = []
    for title, body in chapters.items():
        n = len(_words(body))
        if n < min_words:
            short.append((title, n))
    for title, n in short:
        findings.append(Finding(
            check="chapter_length",
            severity="fail",
            message=f"chapter '{title}' has only {n} words (min={min_words})",
        ))
    if not short and chapters:
        findings.append(Finding(
            check="chapter_length",
            severity="info",
            message=f"all {len(chapters)} chapters >= {min_words} words",
        ))
    return findings


# --- check 5: structural completeness ---------------------------------------

# (level, heading) sections every finished artifact must contain. Missing any
# of these means the heading-coupled parsers above are mis-reading the file
# (risk R-8), so the other checks' results can't be trusted.
REQUIRED_SECTIONS = (
    (2, "World bible"),
    (3, "Characters"),
    (2, "Final Story"),
)


def _heading_present(text: str, level: int, heading: str) -> bool:
    """True if a ``<level> heading`` line exists in ``text``."""
    target = f"{'#' * level} {heading}"
    return any(line.strip() == target for line in text.splitlines())


def check_structure(text: str) -> list[Finding]:
    """Fail when a required artifact section is missing or empty of chapters.

    Guards the heading-coupled parsers: a layout drift (renamed or omitted
    section) otherwise makes the other checks silently mis-parse."""
    findings: list[Finding] = []
    missing = [
        f"{'#' * level} {heading}"
        for level, heading in REQUIRED_SECTIONS
        if not _heading_present(text, level, heading)
    ]
    for heading in missing:
        findings.append(Finding(
            check="structure",
            severity="fail",
            message=f"required section missing: '{heading}'",
        ))
    chapters = split_chapters(text)
    if _heading_present(text, 2, "Final Story") and not chapters:
        findings.append(Finding(
            check="structure",
            severity="fail",
            message="'## Final Story' present but contains no '### Chapter' sections",
        ))
    if not findings and chapters:
        findings.append(Finding(
            check="structure",
            severity="info",
            message=f"all required sections present ({len(chapters)} chapters)",
        ))
    return findings


# --- check 6: protagonist placeholder ---------------------------------------

# Only high-precision literals are gated deterministically: the bare word
# "protagonist" essentially never appears in real narrative prose, so its
# presence is the unnamed-cold-open defect. Subtler placeholders ("the child",
# "the boy") are left to the LLM linter (story_linter.py), which can tell a
# placeholder from a legitimate unnamed minor character.
_PROTAGONIST_PLACEHOLDERS = ("protagonist",)


def check_placeholder_protagonist(
    text: str,
    placeholders: tuple[str, ...] = _PROTAGONIST_PLACEHOLDERS,
) -> list[Finding]:
    """Fail chapters whose prose uses a placeholder term instead of a name."""
    chapters = split_chapters(text)
    patterns = {
        p: re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE) for p in placeholders
    }
    findings: list[Finding] = []
    for title, body in chapters.items():
        hits = sorted(p for p, pat in patterns.items() if pat.search(body))
        if hits:
            findings.append(Finding(
                check="placeholder_protagonist",
                severity="fail",
                message=f"chapter '{title}' uses placeholder term(s) {hits} instead of a name",
            ))
    if not findings and chapters:
        findings.append(Finding(
            check="placeholder_protagonist",
            severity="info",
            message="no protagonist-placeholder terms in chapter prose",
        ))
    return findings


# --- check 7: chapter word band ---------------------------------------------

def check_chapter_band(
    text: str,
    lo: int = CHAPTER_TARGET_BAND[0],
    hi: int = CHAPTER_TARGET_BAND[1],
) -> list[Finding]:
    """Warn chapters outside the target word band (thin / bloated).

    Chapters below the hard floor are already a FAIL from
    :func:`check_chapter_length`, so they're skipped here rather than
    double-reported as "thin"."""
    chapters = split_chapters(text)
    findings: list[Finding] = []
    for title, body in chapters.items():
        n = len(_words(body))
        if n < CHAPTER_MIN_WORDS:
            continue
        if n < lo:
            findings.append(Finding(
                check="chapter_band",
                severity="warn",
                message=f"chapter '{title}' is thin: {n} words (target >= {lo})",
            ))
        elif n > hi:
            findings.append(Finding(
                check="chapter_band",
                severity="warn",
                message=f"chapter '{title}' is bloated: {n} words (target <= {hi})",
            ))
    if not findings and chapters:
        findings.append(Finding(
            check="chapter_band",
            severity="info",
            message=f"all chapters within target band {lo}-{hi} words",
        ))
    return findings


# --- runner ------------------------------------------------------------------

CHECKS = [
    check_cross_chapter_phrase_reuse,
    check_name_drift,
    check_character_presence,
    check_chapter_length,
]


def run_all(path: Path) -> list[Finding]:
    text = _read(path)
    out: list[Finding] = []
    for check in CHECKS:
        out.extend(check(text))
    return out
