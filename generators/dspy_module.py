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

import canon as canon_mod
from _compat import observe
from canon_guard import draft_with_canon_guard
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
        canon = inp.canon if inp.canon is not None else canon_mod.Canon()
        for i, ch in enumerate(inp.chapters_plan, 1):
            hint = act_hint_for_chapter(i, total, inp.spine)
            directives = canon_mod.render_chapter_slice(canon, i)
            logger.info("draft_chapter[%d/%d] | %s", i, total, ch.chapter_title)

            def _draft(_i=i, _ch=ch, _hint=hint, _directives=directives) -> str:
                return self._draft(
                    story_idea=inp.idea,
                    story_title=inp.title,
                    story_spine=_hint["spine_through_act"],
                    act_hint=_hint["label"],
                    world_bible=inp.world_bible,
                    chapter=f"{_ch.chapter_title}\n{_ch.chapter_beats}",
                    canon_directives=_directives,
                ).prose

            prose = draft_with_canon_guard(
                _draft, canon, i, ch.chapter_title, directives,
            )
            update_artifact(
                inp.output_file, f"Chapter {i}: {ch.chapter_title}", prose, level=3,
            )
            chapters.append(
                DraftedChapter(chapter_title=ch.chapter_title, prose=prose),
            )

        return DraftingOutput(chapters=chapters, continuity_artifact=None)
