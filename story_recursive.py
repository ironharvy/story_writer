"""Recursive story drafting: the story-generation procedure applied to itself.

Drafting a story needs an idea, a spine, a (subset of the) world bible, and a
plan of beats. Drafting a *chapter* needs the same things at a smaller scale —
so this module treats each chapter's beats as a child "idea" and recurses:
project the world bible down to what the unit uses, reuse the parent's
act-sliced spine, decompose the unit into sub-beats, and recurse again until a
depth cap (or an atomic unit) is hit, at which point prose is drafted at the
leaves. Leaf prose is concatenated bottom-up — no smoothing pass, no rolling
summary (matching :mod:`story_module`).

``forward`` is kept pure — no prompts, no I/O — so the program can be evaluated,
replayed and compiled by DSPy optimizers. All interactive review lives in the
calling pipeline.
"""

import logging
from dataclasses import dataclass, field

import dspy

from _compat import observe
from story import PlanEntry, WorldBible, act_hint_for_chapter
from story_module import DraftChapter, StoryOutline

logger = logging.getLogger(__name__)


@dataclass
class StoryNode:
    """One node of the recursive draft.

    Either an internal node (``children`` populated, ``prose`` ``None``) or a
    leaf (``prose`` populated, ``children`` empty). ``beats`` is the plan text
    that produced this node's children, or the brief this leaf was drafted from.
    """

    title: str
    beats: str
    depth: int
    children: list["StoryNode"] = field(default_factory=list)
    prose: str | None = None

    @property
    def is_leaf(self) -> bool:
        return self.prose is not None

    def leaf_count(self) -> int:
        if self.is_leaf:
            return 1
        return sum(child.leaf_count() for child in self.children)

    def assembled_prose(self) -> str:
        if self.is_leaf:
            return self.prose
        return "\n\n".join(child.assembled_prose() for child in self.children)


class ExpandUnit(dspy.Signature):
    """Break one story unit into an ordered list of smaller sub-units.

    The ``unit`` is itself a beat (or set of beats) from the parent's plan; treat
    it as the brief for this stretch of the story and decompose it into concrete
    sub-beats, each with a short title. Stay within the unit — do not introduce
    events that belong to earlier or later parts of the story, and do not
    foreshadow later acts (the ``story_spine`` only covers beats up to and
    including this unit's act).
    """

    story_idea: str = dspy.InputField()
    story_title: str = dspy.InputField()
    story_spine: str = dspy.InputField(
        desc="Spine beats up to and including this unit's act",
    )
    act_hint: str = dspy.InputField(
        desc="Which act of the three-act structure this unit belongs to",
    )
    world_bible: WorldBible = dspy.InputField(
        desc="Reference canon for this unit, not to be quoted",
    )
    unit: str = dspy.InputField(desc="This unit's title and beats")
    sub_units: list[PlanEntry] = dspy.OutputField(
        desc="Ordered sub-units, each with a title and a few concrete beats",
    )


class ProjectWorldBible(dspy.Signature):
    """Select the slice of the world bible relevant to one story unit.

    Keep only the rules, characters, locations, and timeline entries this unit
    actually uses. Copy entries verbatim from the full bible — do not rewrite,
    merge, summarize, or invent. When in doubt, keep the entry. ``story_title``
    is unchanged.
    """

    story_title: str = dspy.InputField()
    world_bible: WorldBible = dspy.InputField(desc="The full canon")
    unit: str = dspy.InputField(desc="This unit's title and beats")
    relevant_bible: WorldBible = dspy.OutputField(
        desc="The subset of the canon relevant to this unit",
    )


class RecursiveWriteStory(dspy.Module):
    """Outline the story, recurse into sub-beats, draft prose at the leaves.

    Stages reused at every level: :class:`~story_module.StoryOutline` (top
    level only), :class:`ExpandUnit` (decompose a unit), :class:`ProjectWorldBible`
    (narrow the canon), and :class:`~story_module.DraftChapter` (leaf prose).
    """

    def __init__(self) -> None:
        super().__init__()
        self.build_outline = dspy.ChainOfThought(StoryOutline)
        self.expand_unit = dspy.ChainOfThought(ExpandUnit)
        self.project_bible = dspy.ChainOfThought(ProjectWorldBible)
        self.draft_chapter = dspy.ChainOfThought(DraftChapter)

    @observe()
    def forward(
        self,
        story_idea: str,
        story_title: str,
        story_spine: str,
        world_bible: WorldBible,
        number_of_chapters: int = 7,
        max_depth: int = 2,
    ) -> dspy.Prediction:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")

        outline = self.build_outline(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            world_bible=world_bible,
            number_of_chapters=number_of_chapters,
        )
        total = len(outline.chapters)
        roots: list[StoryNode] = []
        for i, spec in enumerate(outline.chapters, 1):
            hint = act_hint_for_chapter(i, total, story_spine)
            roots.append(
                self._write_unit(
                    title=spec.chapter_title,
                    beats=spec.chapter_beats,
                    story_idea=story_idea,
                    story_title=story_title,
                    spine_slice=hint["spine_through_act"],
                    act_label=hint["label"],
                    world_bible=world_bible,
                    depth=1,
                    max_depth=max_depth,
                    label=f"ch {i}/{total}",
                )
            )
        return dspy.Prediction(outline=outline.chapters, roots=roots)

    def _write_unit(
        self,
        *,
        title: str,
        beats: str,
        story_idea: str,
        story_title: str,
        spine_slice: str,
        act_label: str,
        world_bible: WorldBible,
        depth: int,
        max_depth: int,
        label: str,
    ) -> StoryNode:
        unit_text = f"{title}\n{beats}"
        node = StoryNode(title=title, beats=beats, depth=depth)

        if depth >= max_depth or not beats.strip():
            logger.info("draft_leaf[%s] depth=%d | %s", label, depth, title)
            node.prose = self.draft_chapter(
                story_idea=story_idea,
                story_title=story_title,
                story_spine=spine_slice,
                act_hint=act_label,
                world_bible=world_bible,
                chapter=unit_text,
            ).prose
            return node

        # Narrow the canon to what this unit actually touches.
        sub_bible = self.project_bible(
            story_title=story_title,
            world_bible=world_bible,
            unit=unit_text,
        ).relevant_bible

        # The unit's beats become the child "idea": decompose into sub-units.
        sub_units = self.expand_unit(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=spine_slice,
            act_hint=act_label,
            world_bible=sub_bible,
            unit=unit_text,
        ).sub_units
        logger.info(
            "expand[%s] depth=%d | %s -> %d sub-units",
            label, depth, title, len(sub_units),
        )
        for j, child_spec in enumerate(sub_units, 1):
            node.children.append(
                self._write_unit(
                    title=child_spec.chapter_title,
                    beats=child_spec.chapter_beats,
                    story_idea=unit_text,        # the unit's brief is the child's idea
                    story_title=story_title,
                    spine_slice=spine_slice,     # cheap variant: reuse the parent's act-slice
                    act_label=act_label,
                    world_bible=sub_bible,
                    depth=depth + 1,
                    max_depth=max_depth,
                    label=f"{label}.{j}",
                )
            )
        return node
