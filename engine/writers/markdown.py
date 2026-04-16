"""Render a :class:`StoryState` to the legacy ``story_output.md`` format.

Extracted verbatim from ``main.py``'s step-11 block so the CLI produces
byte-identical output after the refactor. The web app will not use this
writer — it renders :class:`StoryState` directly in React — but the CLI and
scripts still do.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from engine.stages import _split_story_chapters
from engine.types import StoryState


logger = logging.getLogger(__name__)


def render_markdown(state: StoryState) -> str:
    """Return the markdown representation of the story as a string.

    Pure function of ``state`` — no filesystem side effects. Callers that
    want to write to disk should use :func:`write_markdown`.
    """
    parts: list[str] = ["# Story Output\n\n"]

    parts.append("## Core Premise\n")
    parts.append(f"{state.core_premise}\n\n")
    parts.append("## Spine Template\n")
    parts.append(f"{state.spine_template}\n\n")
    parts.append("## World Bible\n")
    parts.append(f"{state.world_bible}\n\n")

    if state.character_visuals:
        parts.append("## Character Visuals\n\n")
        for cv in state.character_visuals:
            parts.append(f"### {cv.name}\n")
            parts.append(f"**Reference:** {cv.reference_mix}\n\n")
            parts.append(f"**Features:** {cv.distinguishing_features}\n\n")
            portrait = state.character_portrait_paths.get(cv.name)
            if portrait:
                parts.append(f"![{cv.name} portrait]({portrait})\n\n")

    parts.append("## Arc Outline\n")
    parts.append(f"{state.arc_outline}\n\n")
    parts.append("## Chapter Plan\n")
    parts.append(f"{state.chapter_plan}\n\n")
    parts.append("## Enhancers Guide\n")
    parts.append(f"{state.enhancers_guide}\n\n")
    parts.append("## Final Story\n")

    if state.scene_image_paths:
        # Match the legacy chapter-interleaved format: each chapter followed
        # by its scene illustration.
        chapters = _split_story_chapters(state.final_story_text)
        for i, chapter_text in enumerate(chapters, start=1):
            parts.append(f"\n\n### Chapter {chapter_text}")
            scene = state.scene_image_paths.get(i)
            if scene:
                parts.append(f"\n\n![Chapter {i} scene]({scene})\n")
    else:
        parts.append(f"{state.final_story_text}\n")

    return "".join(parts)


def write_markdown(state: StoryState, path: str) -> str:
    """Write the state's markdown representation to ``path``.

    Creates parent directories as needed. Returns the final path written.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    logger.info("Saving story output to %s...", path)
    content = render_markdown(state)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
