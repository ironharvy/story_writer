#!/usr/bin/env python3
"""Compile an external bible-style story repo into a single manuscript and run QA metrics.

The story_writer QA tools (``qa.py``, ``scripts/word_count.py``) expect a
single ``.md`` file in the layout produced by ``mymain.py``:

    # Story
    ## ...
    ## World bible
    ### Rules of the World
    ### Locations
    ### Characters         (numbered: ``1. Name, description``)
    ### Timeline
    ## Final Story
    ### Chapter N: Title
    ...

A "bible repo" — like ``ironharvy/The-Nameless`` — stores those pieces as
separate cross-referenced markdown files: ``*-characters.md``,
``*-locations.md``, ``*-world-bible.md``, and a ``chapters/`` directory
containing one prose file per chapter. This script reads that layout, emits
the single-file manuscript, and runs the heuristic QA checks
(``cross_chapter_phrase_reuse``, ``name_drift``, ``character_presence``,
``chapter_length``) plus a word-frequency pass on the chapter prose.

The DSPy-backed checks (``scripts/check_pov.py``, ``scripts/lint_story.py``)
are not invoked here — run them separately against the emitted manuscript if
a model is available.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# Add repo root so top-level modules (qa) import when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Add scripts/ so sibling helpers (word_count) import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from qa import Finding, run_all  # noqa: E402
from word_count import count_words, drop_top_words  # noqa: E402


SEVERITY_RANK = {"fail": 0, "warn": 1, "info": 2}

# Honorifics/titles to strip from dossier headings when building canonical
# character names. The shorter, single-word forms ("Marta", "Grigori", "Azazel")
# match how the chapter prose refers to these characters and keep the QA
# name-drift check usable. Order matters: longest prefix wins.
TITLE_PREFIXES = (
    "Clerical Acolyte",
    "Demon Lord",
    "Archivist",
    "Auditor",
    "Master",
    "Sister",
)

_DOSSIER_HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)\s*$")
_CHAPTER_HEADING_RE = re.compile(r"^#\s+(Chapter\s+\S+:\s+.+?)\s*$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- world bible sections ---------------------------------------------------


def _extract_h2_section(text: str, needle: str) -> str:
    """Return body lines under the first H2 heading containing ``needle``.

    The body ends at the next H2 or EOF. Horizontal-rule dividers (``---``)
    used between sections in the world bible are stripped.
    """
    lines = text.splitlines()
    start: int | None = None
    for i, line in enumerate(lines):
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m and needle in m.group(1):
            start = i + 1
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        if re.match(r"^##\s+", lines[j]):
            end = j
            break
    body = "\n".join(lines[start:end]).strip()
    body = re.sub(r"^---\s*$", "", body, flags=re.MULTILINE).strip()
    return body


# --- characters --------------------------------------------------------------


def _short_character_name(raw: str) -> str:
    """Reduce a dossier heading like ``Sister Marta`` or ``Vessel 84 (...)``
    to the short canonical name (``Marta``, ``Vessel 84``)."""
    name = re.sub(r"\s*\(.+?\)\s*$", "", raw).strip()
    for prefix in TITLE_PREFIXES:
        if name.startswith(prefix + " "):
            return name[len(prefix) + 1 :].strip()
    return name


def _first_bullet(body: str, label: str) -> str:
    """Return the value of the first ``* **<label>:** <value>`` bullet."""
    pat = re.compile(rf"^\*\s+\*\*{re.escape(label)}:\*\*\s+(.+?)\s*$")
    for line in body.splitlines():
        m = pat.match(line)
        if m:
            return m.group(1).strip()
    return ""


def _parse_dossier(text: str) -> list[tuple[str, str]]:
    """Return raw ``(heading_name, dossier_body)`` pairs for ``## N. ...`` blocks."""
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = _DOSSIER_HEADING_RE.match(lines[i])
        if not m:
            i += 1
            continue
        name = m.group(2).strip()
        j = i + 1
        while j < len(lines) and not _DOSSIER_HEADING_RE.match(lines[j]):
            j += 1
        body = "\n".join(lines[i + 1 : j])
        out.append((name, body))
        i = j
    return out


def compile_characters(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw_name, body in _parse_dossier(text):
        name = _short_character_name(raw_name)
        role = _first_bullet(body, "Role") or raw_name
        out.append((name, role))
    return out


def compile_locations(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw_name, body in _parse_dossier(text):
        desc = _first_bullet(body, "Atmosphere & Lore") or raw_name
        out.append((raw_name, desc))
    return out


# --- chapters ---------------------------------------------------------------


def compile_chapters(chapters_dir: Path) -> list[tuple[str, str]]:
    """Return ``[(chapter_heading, body), ...]`` in filename order.

    Files are expected to be named ``chapter-<n>[<suffix>]-<slug>.md`` and to
    start with a ``# Chapter <n>: <title>`` line. The leading H1 is stripped
    from the body and used as the heading.
    """
    files = sorted(chapters_dir.glob("chapter-*.md"))
    out: list[tuple[str, str]] = []
    for f in files:
        text = _read(f)
        lines = text.splitlines()
        if not lines:
            continue
        m = _CHAPTER_HEADING_RE.match(lines[0])
        if not m:
            # Fall back to a synthesized heading so the chapter still parses.
            heading = "Chapter ?: " + f.stem
        else:
            heading = m.group(1).strip()
        body = "\n".join(lines[1:]).strip()
        out.append((heading, body))
    return out


# --- manuscript writer ------------------------------------------------------


def render_manuscript(
    *,
    title: str,
    rules: str,
    locations: list[tuple[str, str]],
    characters: list[tuple[str, str]],
    timeline: str,
    chapters: list[tuple[str, str]],
) -> str:
    out: list[str] = []
    out.append("# Story")
    out.append("")
    out.append("## Title")
    out.append("")
    out.append(title)
    out.append("")
    out.append("## World bible")
    out.append("")
    out.append("### Rules of the World")
    out.append("")
    out.append(rules or "_(none extracted)_")
    out.append("")
    out.append("### Locations")
    out.append("")
    for i, (name, desc) in enumerate(locations, start=1):
        out.append(f"{i}. {name}, {desc}")
    out.append("")
    out.append("### Characters")
    out.append("")
    for i, (name, desc) in enumerate(characters, start=1):
        out.append(f"{i}. {name}, {desc}")
    out.append("")
    out.append("### Timeline")
    out.append("")
    out.append(timeline or "_(none extracted)_")
    out.append("")
    out.append("## Final Story")
    out.append("")
    for heading, body in chapters:
        out.append(f"### {heading}")
        out.append("")
        out.append(body)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# --- QA + word-count reports ------------------------------------------------


def render_qa_report(manuscript_path: Path, findings: list[Finding]) -> str:
    by_check: dict[str, list[Finding]] = {}
    for f in findings:
        by_check.setdefault(f.check, []).append(f)
    totals = Counter(f.severity for f in findings)

    out: list[str] = []
    out.append("# QA Report")
    out.append("")
    out.append(f"Manuscript: `{manuscript_path.name}`")
    out.append("")
    out.append(
        f"Summary: **{totals.get('fail', 0)} fail**, "
        f"{totals.get('warn', 0)} warn, "
        f"{totals.get('info', 0)} info"
    )
    out.append("")
    out.append(
        "Checks from `story_writer/qa.py`. The DSPy-backed POV and lint "
        "checks (`scripts/check_pov.py`, `scripts/lint_story.py`) need a "
        "running LM and are not invoked here."
    )
    out.append("")
    for check in sorted(by_check):
        out.append(f"## {check}")
        out.append("")
        items = sorted(by_check[check], key=lambda f: SEVERITY_RANK.get(f.severity, 9))
        for f in items:
            tag = {"fail": "FAIL", "warn": "WARN", "info": "info"}.get(f.severity, f.severity)
            out.append(f"- **[{tag}]** {f.message}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def render_word_counts(top: list[tuple[str, int]]) -> str:
    out: list[str] = []
    out.append("# Word Frequencies")
    out.append("")
    out.append(
        "Top content words in the ``## Final Story`` section. Stopwords and "
        "the 20 most-frequent surviving words are dropped (the defaults from "
        "``scripts/word_count.py``) so this surfaces the *characteristic* "
        "vocabulary, not the boilerplate."
    )
    out.append("")
    out.append("| Word | Count |")
    out.append("|---|---|")
    for word, count in top:
        out.append(f"| {word} | {count} |")
    return "\n".join(out) + "\n"


# --- CLI -------------------------------------------------------------------


def _autopath(root: Path, override: Path | None, glob: str) -> Path:
    if override is not None:
        return override
    matches = sorted(root.glob(glob))
    if not matches:
        raise SystemExit(f"no file in {root} matches {glob}")
    if len(matches) > 1:
        raise SystemExit(
            f"multiple files in {root} match {glob}: {[m.name for m in matches]}; "
            "pass an explicit --*-file override"
        )
    return matches[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Bible repo root (containing chapters/ and *-characters.md / *-locations.md / *-world-bible.md).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write manuscript.md, qa-report.md, and word-counts.md into.",
    )
    parser.add_argument("--title", default="", help="Manuscript title; defaults to the root folder name.")
    parser.add_argument("--characters-file", type=Path, default=None)
    parser.add_argument("--locations-file", type=Path, default=None)
    parser.add_argument("--world-bible-file", type=Path, default=None)
    parser.add_argument("--chapters-dir", type=Path, default=None)
    parser.add_argument(
        "--rules-section", default="Rules of the Realm",
        help="Substring identifying the world-bible H2 to use as 'Rules of the World'.",
    )
    parser.add_argument(
        "--timeline-section", default="Timeline",
        help="Substring identifying the world-bible H2 to use as 'Timeline'.",
    )
    parser.add_argument("--word-count-top", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"!! not a directory: {root}", file=sys.stderr)
        return 1

    characters_path = _autopath(root, args.characters_file, "*-characters.md")
    locations_path = _autopath(root, args.locations_file, "*-locations.md")
    world_bible_path = _autopath(root, args.world_bible_file, "*-world-bible.md")
    chapters_dir = (args.chapters_dir or root / "chapters").resolve()
    if not chapters_dir.is_dir():
        print(f"!! not a directory: {chapters_dir}", file=sys.stderr)
        return 1

    world_bible = _read(world_bible_path)
    manuscript = render_manuscript(
        title=args.title or root.name,
        rules=_extract_h2_section(world_bible, args.rules_section),
        locations=compile_locations(_read(locations_path)),
        characters=compile_characters(_read(characters_path)),
        timeline=_extract_h2_section(world_bible, args.timeline_section),
        chapters=compile_chapters(chapters_dir),
    )

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manuscript_path = out_dir / "manuscript.md"
    manuscript_path.write_text(manuscript, encoding="utf-8")

    findings = run_all(manuscript_path)
    (out_dir / "qa-report.md").write_text(render_qa_report(manuscript_path, findings), encoding="utf-8")

    counts = count_words(manuscript_path, min_length=4, min_count=2)
    counts = drop_top_words(counts, 20)
    (out_dir / "word-counts.md").write_text(
        render_word_counts(counts.most_common(args.word_count_top)), encoding="utf-8"
    )

    fails = sum(1 for f in findings if f.severity == "fail")
    warns = sum(1 for f in findings if f.severity == "warn")
    print(f"wrote {manuscript_path}")
    print(f"wrote {out_dir / 'qa-report.md'}")
    print(f"wrote {out_dir / 'word-counts.md'}")
    print(f"QA: {fails} fail, {warns} warn")
    return 2 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
