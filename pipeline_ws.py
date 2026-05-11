"""Story pipeline variant that tracks a structured world state.

Identical to :mod:`pipeline` up to and including :func:`pipeline.build_world_bible`.
From there it maintains a :class:`world_state.WorldState` instead of a freeform
``story_so_far`` summary: the world bible is the static canon, and the world
state is what mutates chapter by chapter.
"""

import logging

from _compat import observe
from artifact import update_artifact
from lm_retry import LMOutputError
from pipeline import _snip, build_world_bible
from story import (
    run_clarify_idea,
    run_generate_chapters_plan,
    run_generate_core_premise,
    run_generate_spine,
    sanity_check,
)
from world_state import (
    render_world_state,
    run_advance_world_state,
    run_draft_chapter_with_state,
    run_init_world_state,
)

logger = logging.getLogger(__name__)


@observe()
def _build_foundation(idea: str, title: str, output_file: str):
    """Run the shared idea → premise → spine → world bible steps."""
    updated_idea, story_title = run_clarify_idea(idea, title)
    update_artifact(output_file, "Story Title", story_title)

    core_premise = run_generate_core_premise(updated_idea)
    update_artifact(output_file, "Core Premise", core_premise)
    logger.info("STEP core_premise | output=%s", _snip(core_premise, 600))

    spine = run_generate_spine(updated_idea, core_premise)
    update_artifact(output_file, "Spine", spine)
    logger.info("STEP spine | output=%s", _snip(spine, 800))

    world_bible = build_world_bible(updated_idea, story_title, spine, output_file)
    return updated_idea, story_title, spine, world_bible


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
        try:
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
        except LMOutputError as exc:
            logger.error("Chapter %d drafting failed after retries: %s", i, exc)
            prose = (
                f"*[Chapter {i} could not be generated — the model returned no "
                f"usable text. {exc}]*"
            )
        update_artifact(
            output_file, f"Chapter {i}: {chapter.chapter_title}", prose, level=3
        )
        try:
            world_state = run_advance_world_state(
                world_state,
                chapter_plan_str,
                prose,
                updated_idea,
                story_title,
                world_bible,
            )
        except LMOutputError as exc:
            logger.error(
                "World-state advance failed after retries at chapter %d: %s; "
                "keeping previous state",
                i,
                exc,
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
    updated_idea, story_title, spine, world_bible = _build_foundation(
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
