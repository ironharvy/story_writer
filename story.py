"""Variant A (baseline) chapter drafting.

Drafts each chapter individually from its plan + a rolling free-text
``story_so_far`` summary. Companion to ``pipeline.py``, which threads the
summary across chapters. World-state (variant B) lives in
``world_state.py``; the dspy.Module variant (C) is in ``story_module.py``.

Foundation stages (idea/premise/spine/world-bible/chapter-plan + the
act-slicer + sanity_check) live in ``core.foundation``. Shared types
(``WorldBible``, ``PlanEntry``) live in ``core.types``.

Once the registry refactor (Phase 7) lands, this file becomes
``generators/baseline.py``.
"""
from __future__ import annotations

import random

import dspy

import ui
from _compat import observe
from core.foundation import act_hint_for_chapter
from core.types import WorldBible


@observe()
def run_enhance_chapter(
    chapter: str,
    story_idea: str,
    story_title: str,
    story_spine: str,
    world_bible: WorldBible,
    story_so_far: str,
    chapter_index: int = 1,
    total_chapters: int = 1,
):
    class DraftChapter(dspy.Signature):
        """Write the chapter prose.

The world bible (locations, characters, timeline, rules) is reference material,
not source material — describe scenes in fresh language and never reuse
evocative phrases from the bible verbatim. Drive the chapter on a concrete
scene with at least one moment of friction (a request denied, a plan reversed,
a cost paid on the page). End on an action or image, not a thematic aphorism.
Vary sentence rhythm; avoid stacking triadic 'X and Y' constructions or
'not X but Y' parallelism. Show the world's cost on a body or an object rather
than restating it.

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
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        additional_detail_to_include: str = dspy.InputField(default="")
        chapter: str = dspy.InputField()
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the chapter",
        )
        story_so_far: str = dspy.InputField(desc="Story so far")
        prose: str = dspy.OutputField(desc="Chapter prose")

    class GenerateRandomDetail(dspy.Signature):
        """Generate a random detail for the chapter"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        chapter: str = dspy.InputField()
        random_detail: str = dspy.OutputField(
            desc="Random detail that doesn't influence the chapter but makes it more interesting. Scenry description or quirky item or a meal or something else fitting the setting",
        )

    hint = act_hint_for_chapter(chapter_index, total_chapters, story_spine)
    spine_for_draft = hint["spine_through_act"]

    random_detail = ""
    random_gen = dspy.ChainOfThought(GenerateRandomDetail)
    if random.random() < 0.33:
        random_detail = random_gen(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=spine_for_draft,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            chapter=chapter,
        ).random_detail

    feedback = ""
    draft_chapter_func = dspy.ChainOfThought(DraftChapter)
    while True:
        result = draft_chapter_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=spine_for_draft,
            act_hint=hint["label"],
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            additional_detail_to_include=random_detail,
            chapter=chapter,
            feedback=feedback,
            story_so_far=story_so_far,
        )

        feedback, is_correct = ui.review_answer(
            "Drafted Chapter:",
            result.prose,
        )
        if is_correct:
            return result.prose


@observe()
def run_generate_story_so_far(
    chapter_plan_so_far: list[str],
    story_so_far: str,
    chapter: str,
) -> str:
    class Summarize(dspy.Signature):
        """Summarize story progress based on chapter plan, previous summary and last chapter."""

        chapter_plan_so_far: list[str] = dspy.InputField(
            desc="Chapters and beats that were already written"
        )
        story_so_far: str = dspy.InputField(
            desc="previous summary of the story"
        )
        chapter: str = dspy.InputField(
            desc="The newly-written chapter N. Summarize its events and APPEND to the prior log."
        )
        summary: str = dspy.OutputField(
            desc="new summary of the story"
        )

    generate_story_so_far = dspy.ChainOfThought(Summarize)
    return generate_story_so_far(
        chapter_plan_so_far=chapter_plan_so_far,
        story_so_far=story_so_far,
        chapter=chapter,
    ).summary
