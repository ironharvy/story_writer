"""Variant C — per-chapter :class:`~dspy.ChainOfThought` drafting; no continuity carry.

Each chapter is drafted independently from its plan entry, the (act-sliced)
spine, and the shared world bible. There is no rolling summary or world
state — this is the "draft the article section-by-section" pattern from
DSPy's ``DraftArticle`` example.

Behaviour difference from the pre-refactor ``pipeline_module.write``: the
chapter outline is now the orchestrator-supplied ``chapters_plan`` (same
plan every variant draws from), not :class:`story_module.WriteStory`'s
internal ``StoryOutline``. This makes variants directly comparable in the
benchmark; if the per-variant outline ever needs to come back, do it as
a distinct generator id (e.g. ``dspy_module_outlined``) rather than
re-conflating the two.
"""
from __future__ import annotations

import logging

import dspy

from _compat import observe
from core.artifact import update_artifact
from core.foundation import act_hint_for_chapter
from core.types import DraftedChapter, DraftingInput, DraftingOutput
from generators import register
from story_module import DraftChapter

logger = logging.getLogger(__name__)


@register(
    id="dspy_module",
    status="promoted",
    description="Per-chapter dspy.ChainOfThought drafting from the shared plan; no continuity carry.",
)
class DspyModuleGenerator:
    id = "dspy_module"
    status = "promoted"
    description = "Per-chapter dspy.ChainOfThought drafting from the shared plan; no continuity carry."

    def __init__(self) -> None:
        self._draft = dspy.ChainOfThought(DraftChapter)

    @observe()
    def draft(self, inp: DraftingInput) -> DraftingOutput:
        update_artifact(inp.output_file, "Final Story", "", level=2)
        chapters: list[DraftedChapter] = []
        total = len(inp.chapters_plan)
        for i, ch in enumerate(inp.chapters_plan, 1):
            hint = act_hint_for_chapter(i, total, inp.spine)
            logger.info("draft_chapter[%d/%d] | %s", i, total, ch.chapter_title)
            result = self._draft(
                story_idea=inp.idea,
                story_title=inp.title,
                story_spine=hint["spine_through_act"],
                act_hint=hint["label"],
                world_bible=inp.world_bible,
                chapter=f"{ch.chapter_title}\n{ch.chapter_beats}",
            )
            update_artifact(
                inp.output_file, f"Chapter {i}: {ch.chapter_title}", result.prose, level=3,
            )
            chapters.append(
                DraftedChapter(chapter_title=ch.chapter_title, prose=result.prose),
            )

        return DraftingOutput(chapters=chapters, continuity_artifact=None)
