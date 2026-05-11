"""Mutable world state that evolves chapter-by-chapter.

The *world bible* (rules, character/location profiles, backstory timeline) is
the static canon produced before drafting begins. ``WorldState`` is the part
that *changes* as the story unfolds: where characters are and what they know,
how locations have changed, which plot threads are open, where we are on the
story clock, and what happened in recent chapters.

This module is a parallel alternative to the freeform ``story_so_far`` summary
used by :mod:`pipeline`; it reuses :func:`story.act_hint_for_chapter` and the
:class:`story.WorldBible` produced by :func:`pipeline.build_world_bible`.
"""

import logging
from dataclasses import dataclass, field

import dspy

import ui
from _compat import observe
from lm_retry import call_with_retry
from story import WorldBible, act_hint_for_chapter

logger = logging.getLogger(__name__)


@dataclass
class WorldState:
    """A snapshot of everything that changes while the story is being told."""

    story_clock: str
    characters: list[str]
    locations: list[str]
    plot_threads: list[str]
    key_objects: list[str]
    recent_events: list[str] = field(default_factory=list)


def render_world_state(state: WorldState) -> str:
    """Render a ``WorldState`` as a human-readable markdown block."""

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- (none)"

    return "\n".join(
        [
            f"**Story clock:** {state.story_clock}",
            "",
            "**Characters (current state):**",
            _bullets(state.characters),
            "",
            "**Locations (current state):**",
            _bullets(state.locations),
            "",
            "**Open plot threads:**",
            _bullets(state.plot_threads),
            "",
            "**Key objects:**",
            _bullets(state.key_objects),
            "",
            "**Recent events:**",
            _bullets(state.recent_events),
        ]
    )


@observe()
def run_init_world_state(
    story_idea: str,
    story_title: str,
    story_spine: str,
    world_bible: WorldBible,
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
        result = call_with_retry(
            init_func,
            fields="world_state",
            label="init_world_state",
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            world_bible=world_bible,
            previous_result=previous_result,
            feedback=feedback,
        )
        previous_result = render_world_state(result.world_state)
        feedback, is_correct = ui.review_answer("Initial world state:", previous_result)
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
    result = call_with_retry(
        advance_func,
        fields="updated_state",
        label="advance_world_state",
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
        result = call_with_retry(
            draft_func,
            fields="prose",
            label=f"draft_chapter_ws[{chapter_index}/{total_chapters}]",
            story_idea=story_idea,
            story_title=story_title,
            story_spine=hint["spine_through_act"],
            act_hint=hint["label"],
            world_bible=world_bible,
            world_state=world_state,
            chapter=chapter,
            feedback=feedback,
        )
        feedback, is_correct = ui.review_answer("Drafted Chapter:", result.prose)
        if is_correct:
            return result.prose
