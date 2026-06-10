"""Reconstruct the structured story objects from a generated ``story.md``.

The pipeline writes everything it produces — premise, spine, the enhanced world
bible, every chapter — into one markdown artifact. To run the eval gates over a
*finished* story (rather than only inline during drafting) we parse that
artifact back into a :class:`core.types.WorldBible` plus the chapter map, so a
single ``scripts/run_evals.py`` invocation can grade any past run.

Section layout produced by ``core.foundation`` / ``main`` (see ``qa.py`` header):

    ## Story Title / ## Core Premise / ## Spine
    ## World bible
    ### Rules of the World        -> bullet list
    ### Locations / #### Location N   (enhanced prose blocks)
    ### Timeline                  -> bullet list
    ### Characters / #### Character N (enhanced prose blocks)
    ## Final Story / ### Chapter K: <title>
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import qa
from core.types import WorldBible


@dataclass
class StoryArtifact:
    title: str
    premise: str
    spine: str
    world_bible: WorldBible
    chapters: dict[str, str]
    plan: dict[str, str]  # chapter title -> plan beats, when present in the artifact


def _bullets(section_body: str) -> list[str]:
    """Lines that are markdown bullets or numbered items, stripped of markers."""
    out: list[str] = []
    for line in section_body.splitlines():
        m = re.match(r"^\s*(?:[-*]|\d+\.)\s+(.+)", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _enhanced_blocks(text: str, prefix: str) -> list[str]:
    """Collect the level-4 ``#### <prefix> N`` prose blocks in document order."""
    lines = text.splitlines()
    blocks: list[str] = []
    cur: list[str] | None = None
    head_re = re.compile(rf"^#### {re.escape(prefix)} \d+\b")
    for line in lines:
        if head_re.match(line):
            if cur is not None:
                blocks.append("\n".join(cur).strip())
            cur = []
        elif line.startswith("#### ") or re.match(r"^#{1,3} ", line):
            # any other heading closes the current block
            if cur is not None:
                blocks.append("\n".join(cur).strip())
                cur = None
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def _plan(text: str) -> dict[str, str]:
    """Parse ``### Chapters Plan`` -> ``#### Chapter N: <title>`` beat blocks into a
    ``{"Chapter N: <title>": beats}`` map keyed to match :func:`qa.split_chapters`."""
    body = qa._section(text, 3, "Chapters Plan")
    plan: dict[str, str] = {}
    cur_key: str | None = None
    buf: list[str] = []
    for line in body.splitlines():
        m = re.match(r"^#### (.+)", line)
        if m:
            if cur_key is not None:
                plan[cur_key] = "\n".join(buf).strip()
            cur_key = m.group(1).strip()
            buf = []
        elif cur_key is not None:
            buf.append(line)
    if cur_key is not None:
        plan[cur_key] = "\n".join(buf).strip()
    return plan


def load_story(path: str | Path) -> StoryArtifact:
    text = qa._read(Path(path))
    title = qa._section(text, 2, "Story Title").strip() or "Untitled"
    premise = qa._section(text, 2, "Core Premise").strip()
    spine = qa._section(text, 2, "Spine").strip()

    rules = _bullets(qa._section(text, 3, "Rules of the World"))
    timeline = _bullets(qa._section(text, 3, "Timeline"))

    # Prefer the enhanced #### blocks (what the drafter actually saw); fall back
    # to the short bullets if a run didn't enhance.
    locations = _enhanced_blocks(text, "Location") or _bullets(qa._section(text, 3, "Locations"))
    characters = _enhanced_blocks(text, "Character") or qa.character_bullets(text)

    world_bible = WorldBible(
        story_title=title,
        rules_of_the_world=rules,
        characters=characters,
        locations=locations,
        timeline=timeline,
    )
    return StoryArtifact(
        title=title,
        premise=premise,
        spine=spine,
        world_bible=world_bible,
        chapters=qa.split_chapters(text),
        plan=_plan(text),
    )
