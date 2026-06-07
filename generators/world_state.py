"""Variant B — structured :class:`~core.types.WorldState` carried across chapters.

The world bible is the static canon; ``WorldState`` is what mutates (story
clock, character positions and knowledge, location condition, open plot
threads, key objects, recent events). It's initialised before chapter 1 by
:func:`run_init_world_state` and folded after each chapter by
:func:`run_advance_world_state`.

The drafting helpers used to live in a top-level ``world_state.py``; they are
now inlined here (this is the only consumer) per the AGENTS.md generator
layout. The ``WorldState`` dataclass and its renderer live in ``core.types``.
"""
from __future__ import annotations

import logging

import dspy

from _compat import observe
from core.artifact import update_artifact
from core.foundation import act_hint_for_chapter
from core.types import (
    DraftedChapter,
    DraftingInput,
    DraftingOutput,
    Reviewer,
    WorldBible,
    WorldState,
    render_world_state,
)
from generators import register

logger = logging.getLogger(__name__)


@observe()
def run_init_world_state(
    story_idea: str,
    story_title: str,
    story_spine: str,
    world_bible: WorldBible,
    reviewer: Reviewer | None = None,
) -> WorldState:
    """Derive the opening world state from the world bible, before Chapter 1."""

    class InitWorldState(dspy.Signature):
        """Derive the world state at the very start of the story.

        Translate the static world bible into a *snapshot at story opening*:
        where each character is and what they know/believe right now, the
        starting condition of each location, the plot threads the premise sets
        up, key objects and where they begin. recent_events starts empty.
        """

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        world_bible: WorldBible = dspy.InputField()
        previous_result: str = dspy.InputField(
            desc="Previous attempt at the world state, if regenerating",
        )
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the world state",
        )
        world_state: WorldState = dspy.OutputField(
            desc="The state of the story world at the opening, before Chapter 1",
        )

    feedback = ""
    previous_result = ""
    init_func = dspy.ChainOfThought(InitWorldState)
    while True:
        result = init_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            world_bible=world_bible,
            previous_result=previous_result,
            feedback=feedback,
        )
        previous_result = render_world_state(result.world_state)
        if reviewer is None:
            return result.world_state
        feedback, is_correct = reviewer("Initial world state:", previous_result)
        if is_correct:
            return result.world_state


@observe()
def run_advance_world_state(
    current_state: WorldState,
    chapter_plan: str,
    chapter_prose: str,
    story_idea: str,
    story_title: str,
    world_bible: WorldBible,
) -> WorldState:
    """Fold a freshly written chapter into a new world state."""

    class AdvanceWorldState(dspy.Signature):
        """Update the world state after a chapter has been written.

        Reflect everything the chapter changed: character positions, knowledge,
        relationships and physical condition; physical or political changes to
        locations; plot threads opened or resolved; the advanced story clock;
        movement of key objects. Append the new chapter to recent_events, and
        condense the oldest entries if that log is getting long.
        """

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        world_bible: WorldBible = dspy.InputField(desc="Static canon, for reference")
        chapter_plan: str = dspy.InputField(
            desc="Plan/beats for the chapter just written"
        )
        chapter_prose: str = dspy.InputField(
            desc="Full prose of the chapter just written"
        )
        current_state: WorldState = dspy.InputField(
            desc="World state at the chapter's start"
        )
        updated_state: WorldState = dspy.OutputField(
            desc="World state reflecting everything that changed in the chapter",
        )

    advance_func = dspy.ChainOfThought(AdvanceWorldState)
    result = advance_func(
        story_idea=story_idea,
        story_title=story_title,
        world_bible=world_bible,
        chapter_plan=chapter_plan,
        chapter_prose=chapter_prose,
        current_state=current_state,
    )
    return result.updated_state


@observe()
def run_draft_chapter_with_state(
    chapter: str,
    story_idea: str,
    story_title: str,
    story_spine: str,
    world_bible: WorldBible,
    world_state: WorldState,
    chapter_index: int = 1,
    total_chapters: int = 1,
    reviewer: Reviewer | None = None,
) -> str:
    """Draft chapter prose from the static world bible plus the current state."""

    class DraftChapterWithState(dspy.Signature):
        """Write the chapter prose.

        The world_bible (locations, characters, timeline, rules) is reference material,
        not source material — describe scenes in fresh language and never reuse
        evocative phrases from the bible verbatim. The world_state tells you where the
        story actually stands right now: honour it (character positions, knowledge,
        injuries, open threads, where key objects are) and do not contradict it.
        Drive the chapter on a concrete scene with at least one moment of friction (a
        request denied, a plan reversed, a cost paid on the page). End on an action or
        image, not a thematic aphorism. Vary sentence rhythm; avoid stacking triadic
        'X and Y' constructions or 'not X but Y' parallelism. Show the world's cost on
        a body or an object rather than restating it.

        Write this chapter in a tone appropriate to {act_hint}; do not foreshadow
        events from later acts. The story_spine you receive only covers beats up to
        and including the current act — treat anything beyond it as not yet decided.
        """

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        act_hint: str = dspy.InputField(
            desc="Which act of the three-act structure this chapter belongs to",
        )
        world_bible: WorldBible = dspy.InputField(desc="Static canon, reference only")
        world_state: WorldState = dspy.InputField(
            desc="State of the world as this chapter opens"
        )
        chapter: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the chapter")
        prose: str = dspy.OutputField(desc="Chapter prose")

    hint = act_hint_for_chapter(chapter_index, total_chapters, story_spine)

    feedback = ""
    draft_func = dspy.ChainOfThought(DraftChapterWithState)
    while True:
        result = draft_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=hint["spine_through_act"],
            act_hint=hint["label"],
            world_bible=world_bible,
            world_state=world_state,
            chapter=chapter,
            feedback=feedback,
        )
        if reviewer is None:
            return result.prose
        feedback, is_correct = reviewer("Drafted Chapter:", result.prose)
        if is_correct:
            return result.prose


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
