from __future__ import annotations

import dspy

from story_writer.models.story import EnhancementGuide
from story_writer.stages.base import (
    StageBase,
    StageContext,
    artifact_json,
    parse_json_model,
)


class EnhancementSignature(dspy.Signature):
    """Create per-chapter craft guidance as JSON."""

    premise: str = dspy.InputField()
    world_bible: str = dspy.InputField()
    chapter_plan: str = dspy.InputField()
    enhancement_json: str = dspy.OutputField(
        desc=(
            "JSON object with notes array. Each note has chapter_number, "
            "tension, mystery, "
            "theme_alignment, setup_payoff, emotional_curve, easter_egg."
        )
    )


class EnhancementStage(StageBase):
    name = "enhancement"
    output_model = EnhancementGuide

    def _execute(self, context: StageContext) -> EnhancementGuide:
        result = dspy.Predict(EnhancementSignature)(
            premise=artifact_json(context.artifacts["premise"]),
            world_bible=artifact_json(context.artifacts["world_bible"]),
            chapter_plan=artifact_json(context.artifacts["chapter_plan"]),
        )
        return parse_json_model(result.enhancement_json, EnhancementGuide)
