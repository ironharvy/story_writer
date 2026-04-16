"""End-to-end pipeline runner.

Composes the functions in :mod:`engine.stages` into the default ordering
used by both the CLI and the (future) web worker. Callers can also invoke
individual stages via :func:`run_stage` when driving the pipeline
interactively (e.g. pausing between stages to persist state).
"""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Optional, Sequence

from engine.prompter import Prompter
from engine.stages import (
    run_character_portraits,
    run_character_visuals,
    run_ideate,
    run_inpaint,
    run_premise,
    run_scene_images,
    run_similarity_check,
    run_spine,
    run_story,
    run_world_bible,
    run_world_bible_questions,
)
from engine.types import PipelineOptions, StageName, StoryState


logger = logging.getLogger(__name__)


STAGE_FUNCS: dict[StageName, Callable] = {
    StageName.IDEATE: run_ideate,
    StageName.PREMISE: run_premise,
    StageName.SPINE: run_spine,
    StageName.WORLD_BIBLE_QUESTIONS: run_world_bible_questions,
    StageName.WORLD_BIBLE: run_world_bible,
    StageName.CHARACTER_VISUALS: run_character_visuals,
    StageName.CHARACTER_PORTRAITS: run_character_portraits,
    StageName.STORY: run_story,
    StageName.INPAINT: run_inpaint,
    StageName.SCENE_IMAGES: run_scene_images,
    StageName.SIMILARITY_CHECK: run_similarity_check,
}


def default_stage_order(options: PipelineOptions) -> list[StageName]:
    """Return the stages to run given the current options.

    The ordering mirrors the legacy CLI flow. Optional stages are appended
    based on :class:`PipelineOptions` flags so the worker and CLI share a
    single source of truth about what a "full run" includes.
    """
    order: list[StageName] = [
        StageName.IDEATE,
        StageName.PREMISE,
        StageName.SPINE,
        StageName.WORLD_BIBLE_QUESTIONS,
        StageName.WORLD_BIBLE,
    ]
    if options.enable_images:
        order += [StageName.CHARACTER_VISUALS, StageName.CHARACTER_PORTRAITS]
    order.append(StageName.STORY)
    if options.inpaint_chapters:
        order.append(StageName.INPAINT)
    if options.enable_images:
        order.append(StageName.SCENE_IMAGES)
    if options.check_similar:
        order.append(StageName.SIMILARITY_CHECK)
    return order


def run_stage(
    stage: StageName,
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Run a single stage by name. Useful for workers that persist state
    between stages (so a crash doesn't lose completed work)."""
    func = STAGE_FUNCS[stage]
    logger.debug("Running stage %s", stage.value)
    return func(state, generators, prompter, options)


def run_all(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
    *,
    stages: Optional[Sequence[StageName]] = None,
    on_stage_complete: Optional[Callable[[StageName, StoryState], None]] = None,
) -> StoryState:
    """Run every stage in ``stages`` (or :func:`default_stage_order`) in order.

    ``on_stage_complete`` fires after each stage with the updated state. The
    eventual web worker will hook this to emit SSE progress events and
    persist state to Postgres.
    """
    options.validate()
    plan = list(stages) if stages is not None else default_stage_order(options)

    for stage in plan:
        state = run_stage(stage, state, generators, prompter, options)
        if on_stage_complete is not None:
            on_stage_complete(stage, state)

    return state
