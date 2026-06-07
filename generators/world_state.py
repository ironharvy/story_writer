"""Variant B — structured :class:`~core.types.WorldState` carried across chapters.

The world bible is the static canon; ``WorldState`` is what mutates (story
clock, character positions and knowledge, location condition, open plot
threads, key objects, recent events). It's initialised before chapter 1
and folded after each chapter by :func:`world_state.run_advance_world_state`.
"""
from __future__ import annotations

import logging

from core.artifact import update_artifact
from core.types import DraftedChapter, DraftingInput, DraftingOutput, render_world_state
from generators import register
from world_state import (
    run_advance_world_state,
    run_draft_chapter_with_state,
    run_init_world_state,
)

logger = logging.getLogger(__name__)


@register(
    id="world_state",
    status="promoted",
    description="Structured WorldState carried across chapters (clock, characters, threads).",
)
class WorldStateGenerator:
    id = "world_state"
    status = "promoted"
    description = "Structured WorldState carried across chapters (clock, characters, threads)."

    def draft(self, inp: DraftingInput) -> DraftingOutput:
        world_state = run_init_world_state(
            inp.idea, inp.title, inp.spine, inp.world_bible, reviewer=inp.reviewer,
        )
        update_artifact(
            inp.output_file, "World State (initial)", render_world_state(world_state), level=3,
        )

        update_artifact(inp.output_file, "Final Story", "", level=2)
        chapters: list[DraftedChapter] = []
        per_chapter_states = []
        total = len(inp.chapters_plan)
        for i, ch in enumerate(inp.chapters_plan, 1):
            chapter_plan_str = f"{ch.chapter_title}\n{ch.chapter_beats}"
            prose = run_draft_chapter_with_state(
                chapter_plan_str,
                inp.idea,
                inp.title,
                inp.spine,
                inp.world_bible,
                world_state,
                chapter_index=i,
                total_chapters=total,
                reviewer=inp.reviewer,
            )
            update_artifact(
                inp.output_file, f"Chapter {i}: {ch.chapter_title}", prose, level=3,
            )
            chapters.append(DraftedChapter(chapter_title=ch.chapter_title, prose=prose))

            world_state = run_advance_world_state(
                world_state, chapter_plan_str, prose,
                inp.idea, inp.title, inp.world_bible,
            )
            per_chapter_states.append(world_state)
            update_artifact(
                inp.output_file,
                f"World State after Chapter {i}",
                render_world_state(world_state),
                level=4,
            )

        return DraftingOutput(
            chapters=chapters,
            continuity_artifact={
                "final_state": world_state,
                "per_chapter": per_chapter_states,
            },
        )
