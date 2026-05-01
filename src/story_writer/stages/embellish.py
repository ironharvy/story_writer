from __future__ import annotations

import random

import dspy

from story_writer.models.story import Embellishment
from story_writer.stages.base import (
    StageBase,
    StageContext,
    artifact_json,
    parse_json_model,
)


class EmbellishSignature(dspy.Signature):
    """Create a small non-plot-changing chapter detail as JSON."""

    chapter: str = dspy.InputField()
    world_bible: str = dspy.InputField()
    embellishment_json: str = dspy.OutputField(
        desc=(
            "JSON object with chapter_number, triggered=true, detail. Detail must be "
            "texture only: scenery, object, sensory detail, or habit, "
            "with no plot change."
        )
    )


class EmbellishStage(StageBase):
    name = "embellish"
    output_model = Embellishment

    def _execute(self, context: StageContext) -> Embellishment:
        chapter = context.artifacts["current_chapter"]
        if random.random() > context.embellish_probability:
            return Embellishment(chapter_number=chapter.number, triggered=False)
        result = dspy.Predict(EmbellishSignature)(
            chapter=artifact_json(chapter),
            world_bible=artifact_json(context.artifacts["world_bible"]),
        )
        return parse_json_model(result.embellishment_json, Embellishment)
