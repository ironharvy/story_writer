from __future__ import annotations

import dspy

from story_writer.models.story import Spine
from story_writer.stages.base import (
    StageBase,
    StageContext,
    artifact_json,
    parse_json_model,
)


class SpineSignature(dspy.Signature):
    """Create a Pixar-style six-beat spine as JSON."""

    premise: str = dspy.InputField()
    feedback: str = dspy.InputField()
    spine_json: str = dspy.OutputField(
        desc=(
            "JSON object with once_upon_a_time, every_day, until_one_day, "
            "because_of_that, until_finally, ever_since_then."
        )
    )


class SpineStage(StageBase):
    name = "spine"
    output_model = Spine

    def _execute(self, context: StageContext) -> Spine:
        result = dspy.Predict(SpineSignature)(
            premise=artifact_json(context.artifacts["premise"]),
            feedback=context.feedback,
        )
        return parse_json_model(result.spine_json, Spine)
