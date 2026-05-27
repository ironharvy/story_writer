"""Variant A — rolling free-text continuity (the original baseline).

Each chapter is drafted with :func:`story.run_enhance_chapter` and a
``story_so_far`` summary is updated after every chapter via
:func:`story.run_generate_story_so_far`. The summary is the carry between
chapters; nothing structured is preserved.
"""
from __future__ import annotations

import logging

from core.artifact import update_artifact
from core.types import DraftedChapter, DraftingInput, DraftingOutput
from generators import register
from story import run_enhance_chapter, run_generate_story_so_far

logger = logging.getLogger(__name__)


@register(
    id="baseline",
    status="promoted",
    description="Rolling free-text 'story so far' summary carried across chapters.",
)
class Baseline:
    id = "baseline"
    status = "promoted"
    description = "Rolling free-text 'story so far' summary carried across chapters."

    def draft(self, inp: DraftingInput) -> DraftingOutput:
        update_artifact(inp.output_file, "Final Story", "", level=2)
        chapters: list[DraftedChapter] = []
        story_so_far = ""
        total = len(inp.chapters_plan)
        for i, ch in enumerate(inp.chapters_plan, 1):
            chapter_plan_str = f"{ch.chapter_title}\n{ch.chapter_beats}"
            prose = run_enhance_chapter(
                chapter_plan_str,
                inp.idea,
                inp.title,
                inp.spine,
                inp.world_bible,
                story_so_far,
                chapter_index=i,
                total_chapters=total,
            )
            update_artifact(
                inp.output_file, f"Chapter {i}: {ch.chapter_title}", prose, level=3,
            )
            chapters.append(DraftedChapter(chapter_title=ch.chapter_title, prose=prose))

            plan_so_far_strs = [
                c.chapter_title + "\n" + c.chapter_beats
                for c in inp.chapters_plan[:i]
            ]
            story_so_far = run_generate_story_so_far(plan_so_far_strs, story_so_far, prose)
            logger.info("chapter %d/%d drafted; story_so_far_len=%d", i, total, len(story_so_far))

        return DraftingOutput(chapters=chapters, continuity_artifact=None)
