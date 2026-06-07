"""Variant A — rolling free-text continuity (the original baseline).

Each chapter is drafted with :func:`run_enhance_chapter` and a ``story_so_far``
summary is updated after every chapter via :func:`run_generate_story_so_far`.
The summary is the carry between chapters; nothing structured is preserved.

The two drafting helpers used to live in a top-level ``story.py``; they are
now inlined here (this is the only consumer) per the AGENTS.md generator
layout. Foundation stages (idea/premise/spine/world-bible/chapter-plan + the
act-slicer) live in ``core.foundation``; shared types in ``core.types``.
"""
from __future__ import annotations

import logging
import random

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
)
from generators import register

logger = logging.getLogger(__name__)


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
    reviewer: Reviewer | None = None,
) -> str:
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

        if reviewer is None:
            return result.prose
        feedback, is_correct = reviewer(
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
                reviewer=inp.reviewer,
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
