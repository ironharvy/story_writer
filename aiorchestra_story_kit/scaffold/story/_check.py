#!/usr/bin/env python3
"""Lightweight consistency gate for an aiorchestra story repo.

Run by aiorchestra at the ``validate`` stage. stdlib only. Does NOT judge prose
quality — only that the ``story/`` tree is internally consistent: a downstream
artifact that has real content must not sit on top of empty / placeholder
prerequisites, and an assembled ``story/final.md`` must be well-formed.

Exit 0 = ok; exit 1 = inconsistency (details on stderr).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent  # the story/ directory

PLACEHOLDER_MARKER = "fills this in"
MIN_CONTENT_CHARS = 80
MIN_CHAPTER_BYTES = 200
MIN_FINAL_BYTES = 500

WORLD_FILES = [
    "world/rules.md",
    "world/characters.md",
    "world/locations.md",
    "world/timeline.md",
]

# (artifact, prerequisites): if the artifact has real content, each prereq must too.
DEPS: list[tuple[str, list[str]]] = [
    ("premise.md", ["idea.md"]),
    ("spine.md", ["premise.md"]),
    ("world/rules.md", ["premise.md", "spine.md"]),
    ("world/characters.md", ["premise.md", "spine.md", "world/rules.md"]),
    ("world/locations.md", ["premise.md", "spine.md", "world/rules.md", "world/characters.md"]),
    (
        "world/timeline.md",
        ["premise.md", "spine.md", "world/rules.md", "world/characters.md", "world/locations.md"],
    ),
    ("chapter_plan.md", ["premise.md", "spine.md", *WORLD_FILES]),
    ("enhancers.md", ["chapter_plan.md", *WORLD_FILES]),
]


def content_len(rel: str) -> int:
    """Length of a story file with markdown noise and placeholder text stripped."""
    path = ROOT / rel
    if not path.is_file():
        return 0
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith(("#", "<!--", "|")):
            continue
        if s.startswith("_") and s.endswith("_"):  # italic placeholder note
            continue
        if PLACEHOLDER_MARKER in s:
            continue
        if s.startswith("- **") and s.endswith("—"):  # empty spine beat
            continue
        kept.append(s)
    return len("\n".join(kept))


def has_content(rel: str) -> bool:
    return content_len(rel) >= MIN_CONTENT_CHARS


def main() -> int:
    errors: list[str] = []

    for artifact, prereqs in DEPS:
        if not has_content(artifact):
            continue
        for pre in prereqs:
            if not has_content(pre):
                errors.append(f"story/{artifact} has content but prerequisite story/{pre} is empty/placeholder")

    chapters_dir = ROOT / "chapters"
    chapter_files = sorted(chapters_dir.glob("*.md")) if chapters_dir.is_dir() else []
    if chapter_files:
        if not has_content("chapter_plan.md"):
            errors.append("story/chapters/ has chapters but story/chapter_plan.md is empty/placeholder")
        for cf in chapter_files:
            if cf.stat().st_size < MIN_CHAPTER_BYTES:
                errors.append(f"story/chapters/{cf.name} is suspiciously short (< {MIN_CHAPTER_BYTES} bytes)")

    final = ROOT / "final.md"
    if final.is_file():
        text = final.read_text(encoding="utf-8")
        if len(text) < MIN_FINAL_BYTES:
            errors.append("story/final.md exists but is too short to be the assembled story")
        if "## Final Story" not in text:
            errors.append("story/final.md is missing the `## Final Story` section")

    if errors:
        print("story/_check.py: consistency check failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("story/_check.py: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
