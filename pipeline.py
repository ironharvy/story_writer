"""Variant A (baseline) pipeline: foundation → chapter plan → drafting loop.

Reuses :func:`core.foundation.build_foundation` for everything up through the
world bible, then walks the chapter plan one beat at a time. Each chapter is
drafted with :func:`story.run_enhance_chapter` and the rolling
``story_so_far`` summary is updated by :func:`story.run_generate_story_so_far`.

Once the registry refactor (Phase 7) lands, the body of :func:`write` becomes
the entry point of ``generators/baseline.py``.
"""
import logging

from _compat import observe
from core.artifact import update_artifact
from core.foundation import (
    _snip,
    build_foundation,
    run_generate_chapters_plan,
    sanity_check,
)
from story import run_enhance_chapter, run_generate_story_so_far

logger = logging.getLogger(__name__)


@observe()
def write(idea: str, title: str, output_file: str, number_of_chapters: int = 7):
    updated_idea, story_title, spine, world_bible = build_foundation(
        idea, title, output_file
    )

    logger.info("STEP sanity_check | running on world_bible (chars=%d, locs=%d)",
                len(world_bible.characters), len(world_bible.locations))
    is_consistent = sanity_check(updated_idea, story_title, spine, world_bible)
    logger.info("STEP sanity_check | output is_consistent=%s", is_consistent)
    if not is_consistent:
        logger.error("Idea, spine, and world bible are not consistent")
        return

    # Chapters plan
    logger.info("STEP chapters_plan | inputs idea=%s | title=%s | spine=%s | world_bible(chars=%d, locs=%d) | number_of_chapters=%d",
                _snip(updated_idea), _snip(story_title, 80), _snip(spine),
                len(world_bible.characters), len(world_bible.locations), number_of_chapters)
    chapters_plan = run_generate_chapters_plan(updated_idea, story_title, spine, world_bible, number_of_chapters)
    logger.info("STEP chapters_plan | output count=%d", len(chapters_plan))
    update_artifact(output_file, "Chapters Plan", "", level=3)

    for i, chapter in enumerate(chapters_plan, 1):
        logger.info("STEP chapters_plan | chapter %d=%s", i, _snip(chapter, 300))
        update_artifact(output_file, f"Chapter {i}: {chapter.chapter_title}", chapter.chapter_beats, level=4)
    logger.info(f"Chapters plan added to artifact {output_file}")

    # Final story
    story_so_far = ""
    update_artifact(output_file, "Final Story", "", level=2)

    for i, chapter in enumerate(chapters_plan, 1):
        chapter_plan_str = f"{chapter.chapter_title}\n{chapter.chapter_beats}"
        logger.info("STEP enhance_chapter[%d/%d] | chapter_outline=%s | story_so_far_len=%d",
                    i, len(chapters_plan), _snip(chapter_plan_str, 200), len(story_so_far))
        enhanced = run_enhance_chapter(
            chapter_plan_str, updated_idea, story_title, spine, world_bible, story_so_far,
            chapter_index=i, total_chapters=len(chapters_plan),
        )
        logger.info("STEP enhance_chapter[%d/%d] | output_chars=%d preview=%s",
                    i, len(chapters_plan), len(enhanced), _snip(enhanced, 240))
        update_artifact(output_file, f"Chapter {i}: {chapter.chapter_title}", enhanced, level=3)
        logger.info(f"Chapter {i} added to artifact {output_file}")
        plan_so_far = chapters_plan[:i]
        logger.info("STEP story_so_far[%d/%d] | summarizing | plan_so_far_count=%d",
                    i, len(chapters_plan), len(plan_so_far))

        chapter_plan_strs = [c.chapter_title + "\n" + c.chapter_beats for c in plan_so_far]
        story_so_far = run_generate_story_so_far(chapter_plan_strs, story_so_far, enhanced)
        logger.info("STEP story_so_far[%d/%d] | new_len=%d", i, len(chapters_plan), len(story_so_far))
