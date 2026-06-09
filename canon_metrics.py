"""Deterministic canon/continuity metrics.

Pure functions over a story's markdown text plus a :class:`canon.Canon`. They
mirror :mod:`qa`'s contract exactly — each returns ``list[qa.Finding]`` with
``severity`` in ``{"info", "warn", "fail"}`` — so they fold into the same
info/warn/fail report and the same post-generation gate. Unlike :mod:`qa`'s
checks (which read the manuscript alone), these are parameterised by the canon
config, which encodes the rules that span chapters.

Each metric here is the GUARD half of an intervention whose PREVENT half lives in
:func:`canon.render_chapter_slice` (see ``docs/canon-schema.md``):

* :func:`check_scaffolding_leak`   — planning/production vocab in prose.
* :func:`check_name_arc`           — a name used before its allowed chapter.
* :func:`check_canon_fact`         — appearance contradicting locked_appearance.
* :func:`check_required_artifact`  — a chapter missing its required artifact.
* :func:`check_timeline_age`       — a year-span that can't fit the timeline.
"""
from __future__ import annotations

import re

from qa import Finding, split_chapters

# Number words 0-20 plus the round tens, enough to read narrated year-spans.
_NUMBER_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50,
}

_CHAPTER_NUM_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)
# A year-span phrase: "<number> years" optionally lead by next/another/over/etc.
_SPAN_RE = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+years\b",
    re.IGNORECASE,
)


def _chapter_number(title: str) -> int | None:
    """Extract the integer N from a ``Chapter N: ...`` heading label."""
    m = _CHAPTER_NUM_RE.search(title)
    return int(m.group(1)) if m else None


def _numbered_chapters(text: str) -> list[tuple[int, str, str]]:
    """Return ``(number, title, body)`` for chapters with a parseable number."""
    out: list[tuple[int, str, str]] = []
    for title, body in split_chapters(text).items():
        n = _chapter_number(title)
        if n is not None:
            out.append((n, title, body))
    return out


def _mask_devices(body: str, device_allowlist: list[str]) -> str:
    """Blank out any sanctioned device phrase so a blocklist hit inside it is not
    counted as a scaffolding leak."""
    masked = body
    for phrase in device_allowlist:
        masked = re.sub(re.escape(phrase), " ", masked, flags=re.IGNORECASE)
    return masked


# --- metric 1: scaffolding leak ---------------------------------------------

def check_scaffolding_leak(text: str, canon) -> list[Finding]:
    """Fail prose that contains planning/production vocabulary.

    Device-allowlist phrases are masked first, so an intentional meta-narrative
    device ("The reader sees her stand.") is not flagged even though it contains
    a blocked token.
    """
    patterns = [re.compile(p, re.IGNORECASE) for p in canon.scaffolding_blocklist]
    if not patterns:
        return []

    findings: list[Finding] = []
    leaks = 0
    for title, body in split_chapters(text).items():
        masked = _mask_devices(body, canon.device_allowlist)
        for pat in patterns:
            for m in pat.finditer(masked):
                leaks += 1
                findings.append(Finding(
                    check="scaffolding_leak",
                    severity="fail",
                    message=f"chapter '{title}' leaks scaffolding vocabulary: {m.group(0)!r}",
                ))
    if leaks == 0:
        findings.append(Finding(
            check="scaffolding_leak",
            severity="info",
            message="no scaffolding vocabulary in chapter prose",
        ))
    return findings


# --- metric 2: name-arc ordering --------------------------------------------

def _quoted_spans(body: str) -> list[str]:
    """Return the contents of quoted spans (straight or curly double quotes)."""
    return re.findall(r"[\"“]([^\"”]+)[\"”]", body)


def check_name_arc(text: str, canon) -> list[Finding]:
    """Fail a name that appears before the chapter it is allowed in.

    Two rules, both from ``canon.name_arc``:

    * ``earliest_narration``: a name may not appear anywhere in a chapter whose
      number is below its earliest chapter.
    * ``reveal_rules``: a specific speaker may only utter a name from a given
      chapter onward (detected as the name inside a quoted span in a chapter
      where that speaker is present).
    """
    arc = canon.name_arc or {}
    earliest = arc.get("earliest_narration", {})
    reveal_rules = arc.get("reveal_rules", [])
    if not earliest and not reveal_rules:
        return []

    chapters = _numbered_chapters(text)
    findings: list[Finding] = []

    for number, title, body in chapters:
        for name, first_ch in earliest.items():
            if number >= int(first_ch):
                continue
            if re.search(rf"\b{re.escape(name)}\b", body):
                findings.append(Finding(
                    check="name_arc",
                    severity="fail",
                    message=(
                        f"chapter '{title}' uses the name '{name}' but it is not "
                        f"allowed until chapter {first_ch}"
                    ),
                ))

    for rule in reveal_rules:
        speaker = rule.get("speaker", "")
        name = rule.get("name", "")
        from_ch = int(rule.get("from_chapter", 0))
        if not speaker or not name:
            continue
        for number, title, body in chapters:
            if number >= from_ch:
                continue
            if not re.search(rf"\b{re.escape(speaker)}\b", body):
                continue
            said = any(re.search(rf"\b{re.escape(name)}\b", q) for q in _quoted_spans(body))
            if said:
                findings.append(Finding(
                    check="name_arc",
                    severity="fail",
                    message=(
                        f"chapter '{title}': {speaker} says '{name}' but may not "
                        f"until chapter {from_ch}"
                    ),
                ))

    if not findings:
        findings.append(Finding(
            check="name_arc",
            severity="info",
            message="name-arc ordering respected",
        ))
    return findings


# --- metric 3: canon-fact contradiction (locked appearance) -----------------

_NEGATION_RE = re.compile(r"\b(no|not|never|n't|without)\b", re.IGNORECASE)


def check_canon_fact(text: str, canon) -> list[Finding]:
    """Fail prose whose protagonist appearance contradicts ``locked_appearance``.

    For each locked feature (e.g. eyes=grey) the canon lists contradicting
    adjectives (blue, green, ...). A contradiction within a single clause of the
    feature word ("his blue eyes", "eyes were blue") is a FAIL — unless negated
    ("not blue eyes"), which is consistent.
    """
    proto = (canon.locked_appearance or {}).get("protagonist", {})
    contradictions = proto.get("contradictions", {}) if proto else {}
    if not contradictions:
        return []

    prose = "\n\n".join(split_chapters(text).values())
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()

    for feature, bad_values in contradictions.items():
        correct = proto.get(feature, "")
        for bad in bad_values:
            # Two unambiguous shapes, both kept inside a single clause so an
            # adjacent feature doesn't borrow another feature's colour
            # ("grey eyes ... brown hair"):
            #   - attributive: "blue eyes", "blue glittering eyes"
            #   - predicate:   "eyes were (a deep) green"
            patterns = [
                rf"\b{re.escape(bad)}\b(?:\s+\w+){{0,2}}\s+\b{re.escape(feature)}\b",
                rf"\b{re.escape(feature)}\b\s+"
                rf"(?:were|was|are|is|looked|seemed|appeared|turned|shone|"
                rf"gleamed|glowed|remained|stayed|gone)\b"
                rf"[^.;,\n]{{0,15}}\b{re.escape(bad)}\b",
            ]
            for pat in patterns:
                for m in re.finditer(pat, prose, re.IGNORECASE):
                    window = prose[max(0, m.start() - 12): m.start()]
                    if _NEGATION_RE.search(window):
                        continue
                    key = (feature, bad)
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(Finding(
                        check="canon_fact",
                        severity="fail",
                        message=(
                            f"appearance contradicts locked {feature}='{correct}': "
                            f"prose says {feature} are/were '{bad}' ({m.group(0).strip()!r})"
                        ),
                    ))

    if not findings:
        findings.append(Finding(
            check="canon_fact",
            severity="info",
            message="no locked-appearance contradictions detected",
        ))
    return findings


# --- metric 4: required-artifact placement ----------------------------------

def check_required_artifact(text: str, canon) -> list[Finding]:
    """Fail a chapter that is missing its required artifact.

    Only chapters that are present in the manuscript are checked (a not-yet-drafted
    chapter is not a failure here)."""
    required = canon.required_artifacts or {}
    if not required:
        return []

    by_number = {n: (title, body) for n, title, body in _numbered_chapters(text)}
    findings: list[Finding] = []
    checked = 0
    for chapter_n, spec in required.items():
        if chapter_n not in by_number:
            continue
        checked += 1
        title, body = by_number[chapter_n]
        patterns = spec.get("patterns", []) or [spec.get("name", "")]
        if any(re.search(re.escape(p), body, re.IGNORECASE) for p in patterns if p):
            continue
        findings.append(Finding(
            check="required_artifact",
            severity="fail",
            message=(
                f"chapter '{title}' is missing its required artifact: "
                f"{spec.get('name', 'artifact')}"
            ),
        ))
    if not findings and checked:
        findings.append(Finding(
            check="required_artifact",
            severity="info",
            message=f"all {checked} present required artifacts found",
        ))
    return findings


# --- metric 5: timeline / age span ------------------------------------------

def _word_to_int(token: str) -> int | None:
    token = token.lower()
    if token.isdigit():
        return int(token)
    return _NUMBER_WORDS.get(token)


def check_timeline_age(text: str, canon) -> list[Finding]:
    """Fail a narrated year-span that cannot fit the canonical timeline.

    The canon's ``timeline.max_total_span_years`` is the longest span the story
    can express (age 16 -> 21 = 5 years). Any "<N> years" phrase with N greater
    than that is impossible (e.g. "the next eight years" in a 5-year story)."""
    timeline = canon.timeline or {}
    max_span = timeline.get("max_total_span_years")
    if max_span is None:
        return []
    max_span = int(max_span)

    prose = "\n\n".join(split_chapters(text).values())
    findings: list[Finding] = []
    for m in _SPAN_RE.finditer(prose):
        value = _word_to_int(m.group(1))
        if value is None or value <= max_span:
            continue
        findings.append(Finding(
            check="timeline_age",
            severity="fail",
            message=(
                f"timeline: prose says '{m.group(0)}' but the story spans at most "
                f"{max_span} years"
            ),
        ))
    if not findings:
        findings.append(Finding(
            check="timeline_age",
            severity="info",
            message=f"no year-span exceeds the {max_span}-year timeline",
        ))
    return findings


# --- runner ------------------------------------------------------------------

CHECKS = [
    check_scaffolding_leak,
    check_name_arc,
    check_canon_fact,
    check_required_artifact,
    check_timeline_age,
]


def run_all(text: str, canon) -> list[Finding]:
    """Run every deterministic canon metric and concatenate the findings."""
    out: list[Finding] = []
    for check in CHECKS:
        out.extend(check(text, canon))
    return out
