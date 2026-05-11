"""Multi-stage story drafting expressed as a composable ``dspy.Module``.

Modeled on DSPy's ``DraftArticle`` example (outline → draft each section):
:class:`WriteStory` takes the foundation produced by the interactive pipeline
(idea, title, spine, world bible) and, with no human in the loop, builds a
chapter outline and then drafts each chapter from that outline. Like the
article example there is *no* rolling summary between chapters — each chapter
is drafted from its own outline entry plus the shared world bible and the
act-sliced spine. (If continuity drift shows up in practice, the obvious next
step is to feed the whole outline, or a running recap, into ``DraftChapter``.)

``forward`` is kept pure — no prompts, no I/O — so the program can be
evaluated, replayed and compiled by DSPy optimizers. All interactive review
lives in the calling pipeline.
"""

import logging
from dataclasses import dataclass

import dspy

from _compat import observe
from lm_retry import LMOutputError, call_with_retry
from story import PlanEntry, WorldBible, act_hint_for_chapter

logger = logging.getLogger(__name__)


@dataclass
class DraftedChapter:
    """One chapter of the finished draft: its outline entry plus the prose."""

    chapter_title: str
    chapter_beats: str
    prose: str


class StoryOutline(dspy.Signature):
    """Outline a thorough chapter-by-chapter plan for the story."""

    story_idea: str = dspy.InputField()
    story_title: str = dspy.InputField()
    story_spine: str = dspy.InputField()
    world_bible: WorldBible = dspy.InputField()
    number_of_chapters: int = dspy.InputField()
    chapters: list[PlanEntry] = dspy.OutputField(
        desc="Ordered chapters, each with a title and a few concrete beats",
    )


class DraftChapter(dspy.Signature):
    """Draft one chapter of the story from its outline entry.

    The world_bible (rules, characters, locations, timeline) is reference
    material, not source material — describe scenes in fresh language and never
    reuse evocative phrases from it verbatim. Drive the chapter on a concrete
    scene with at least one moment of friction (a request denied, a plan
    reversed, a cost paid on the page). End on an action or image, not a
    thematic aphorism. Vary sentence rhythm; avoid stacking triadic 'X and Y'
    constructions or 'not X but Y' parallelism.

    Write in a tone appropriate to act_hint and do not foreshadow events from
    later acts — the story_spine you receive only covers beats up to and
    including the current act, so treat anything beyond it as not yet decided.
    """

    story_idea: str = dspy.InputField()
    story_title: str = dspy.InputField()
    story_spine: str = dspy.InputField(
        desc="Spine beats up to and including this chapter's act",
    )
    act_hint: str = dspy.InputField(
        desc="Which act of the three-act structure this chapter belongs to",
    )
    world_bible: WorldBible = dspy.InputField(desc="Reference canon, not to be quoted")
    chapter: str = dspy.InputField(desc="This chapter's title and beats")
    prose: str = dspy.OutputField(desc="Chapter prose")


class WriteStory(dspy.Module):
    """Two stages: build a chapter outline, then draft each chapter from it."""

    def __init__(self) -> None:
        super().__init__()
        self.build_outline = dspy.ChainOfThought(StoryOutline)
        self.draft_chapter = dspy.ChainOfThought(DraftChapter)

    @observe()
    def forward(
        self,
        story_idea: str,
        story_title: str,
        story_spine: str,
        world_bible: WorldBible,
        number_of_chapters: int = 7,
    ) -> dspy.Prediction:
        outline = call_with_retry(
            self.build_outline,
            fields="chapters",
            label="story_outline",
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            world_bible=world_bible,
            number_of_chapters=number_of_chapters,
        )
        total = len(outline.chapters)
        drafted: list[DraftedChapter] = []
        for i, spec in enumerate(outline.chapters, 1):
            hint = act_hint_for_chapter(i, total, story_spine)
            logger.info("draft_chapter[%d/%d] | %s", i, total, spec.chapter_title)
            try:
                section = call_with_retry(
                    self.draft_chapter,
                    fields="prose",
                    label=f"draft_chapter[{i}/{total}]",
                    story_idea=story_idea,
                    story_title=story_title,
                    story_spine=hint["spine_through_act"],
                    act_hint=hint["label"],
                    world_bible=world_bible,
                    chapter=f"{spec.chapter_title}\n{spec.chapter_beats}",
                )
                prose = section.prose
            except LMOutputError as exc:
                logger.error("Chapter %d could not be drafted after retries: %s", i, exc)
                prose = (
                    f"*[Chapter {i} could not be generated — the model returned "
                    f"no usable text. {exc}]*"
                )
            drafted.append(
                DraftedChapter(
                    chapter_title=spec.chapter_title,
                    chapter_beats=spec.chapter_beats,
                    prose=prose,
                )
            )
        return dspy.Prediction(outline=outline.chapters, chapters=drafted)
