"""Variant B (world-state) pipeline: foundation → world state → drafting loop.

Shares the foundation (idea/premise/spine/world-bible) with variants A and C
via :func:`core.foundation.build_foundation`, then maintains a structured
:class:`~core.types.WorldState` across chapters instead of A's freeform
``story_so_far`` summary.

Once the registry refactor (Phase 7) lands, the body of :func:`write` becomes
the entry point of ``generators/world_state.py``.
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
from core.types import render_world_state
from world_state import (
    run_advance_world_state,
    run_draft_chapter_with_state,
    run_init_world_state,
)

logger = logging.getLogger(__name__)


@observe()
def _plan_chapters(
    updated_idea, story_title, spine, world_bible, output_file, number_of_chapters
):
    """Generate and record the chapter plan."""
    chapters_plan = run_generate_chapters_plan(
        updated_idea,
        story_title,
        spine,
        world_bible,
        number_of_chapters,
    )
    update_artifact(output_file, "Chapters Plan", "", level=3)
    for i, chapter in enumerate(chapters_plan, 1):
        update_artifact(
            output_file,
            f"Chapter {i}: {chapter.chapter_title}",
            chapter.chapter_beats,
            level=4,
        )
    logger.info("STEP chapters_plan | output count=%d", len(chapters_plan))
    return chapters_plan


@observe()
def _draft_chapters(
    updated_idea,
    story_title,
    spine,
    world_bible,
    world_state,
    chapters_plan,
    output_file,
):
    """Draft each chapter, advancing the world state after every one."""
    total = len(chapters_plan)
    update_artifact(output_file, "Final Story", "", level=2)
    for i, chapter in enumerate(chapters_plan, 1):
        chapter_plan_str = f"{chapter.chapter_title}\n{chapter.chapter_beats}"
        logger.info(
            "STEP draft_chapter[%d/%d] | clock=%s",
            i,
            total,
            _snip(world_state.story_clock, 120),
        )
        prose = run_draft_chapter_with_state(
            chapter_plan_str,
            updated_idea,
            story_title,
            spine,
            world_bible,
            world_state,
            chapter_index=i,
            total_chapters=total,
        )
        update_artifact(
            output_file, f"Chapter {i}: {chapter.chapter_title}", prose, level=3
        )
        world_state = run_advance_world_state(
            world_state,
            chapter_plan_str,
            prose,
            updated_idea,
            story_title,
            world_bible,
        )
        update_artifact(
            output_file,
            f"World State after Chapter {i}",
            render_world_state(world_state),
            level=4,
        )
        logger.info(
            "STEP advance_world_state[%d/%d] | clock=%s",
            i,
            total,
            _snip(world_state.story_clock, 120),
        )


@observe()
def write(idea: str, title: str, output_file: str, number_of_chapters: int = 7) -> None:
    """Run the world-state story pipeline end to end."""
    updated_idea, story_title, spine, world_bible = build_foundation(
        idea, title, output_file
    )

    if not sanity_check(updated_idea, story_title, spine, world_bible):
        logger.error("Idea, spine, and world bible are not consistent")
        return

    chapters_plan = _plan_chapters(
        updated_idea,
        story_title,
        spine,
        world_bible,
        output_file,
        number_of_chapters,
    )

    world_state = run_init_world_state(updated_idea, story_title, spine, world_bible)
    update_artifact(
        output_file, "World State (initial)", render_world_state(world_state), level=3
    )

    _draft_chapters(
        updated_idea,
        story_title,
        spine,
        world_bible,
        world_state,
        chapters_plan,
        output_file,
    )
