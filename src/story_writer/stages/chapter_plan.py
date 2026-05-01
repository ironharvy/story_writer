from __future__ import annotations

import dspy

from story_writer.config import LENGTH_CHAPTER_COUNTS
from story_writer.models.story import ChapterPlan
from story_writer.stages.base import (
    StageBase,
    StageContext,
    artifact_json,
    parse_json_model,
)


class ChapterPlanSignature(dspy.Signature):
    """Create ordered chapters with embedded beats as JSON."""

    premise: str = dspy.InputField()
    spine: str = dspy.InputField()
    world_bible: str = dspy.InputField()
    length: str = dspy.InputField()
    chapter_count: int = dspy.InputField()
    chapter_plan_json: str = dspy.OutputField(
        desc=(
            "JSON object with length and chapters. Each chapter has number, title, "
            "purpose, spine_alignment, and 4 to 10 beats. "
            "Each beat has order and summary."
        )
    )


class ChapterPlanStage(StageBase):
    name = "chapter_plan"
    output_model = ChapterPlan

    def _execute(self, context: StageContext) -> ChapterPlan:
        result = dspy.Predict(ChapterPlanSignature)(
            premise=artifact_json(context.artifacts["premise"]),
            spine=artifact_json(context.artifacts["spine"]),
            world_bible=artifact_json(context.artifacts["world_bible"]),
            length=context.length.value,
            chapter_count=LENGTH_CHAPTER_COUNTS[context.length],
        )
        return parse_json_model(result.chapter_plan_json, ChapterPlan)
