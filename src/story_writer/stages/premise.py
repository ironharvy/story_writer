from __future__ import annotations

import dspy

from story_writer.models.story import Premise
from story_writer.stages.base import (
    StageBase,
    StageContext,
    artifact_json,
    parse_json_model,
)


class PremiseSignature(dspy.Signature):
    """Create a reviewed core story premise as JSON."""

    idea: str = dspy.InputField()
    length: str = dspy.InputField()
    clarifications: str = dspy.InputField()
    feedback: str = dspy.InputField()
    premise_json: str = dspy.OutputField(
        desc=(
            "JSON object with protagonist, want, obstacle, stakes, genre, "
            "length, summary. "
            "The summary must be 2 to 4 sentences."
        )
    )


class PremiseStage(StageBase):
    name = "premise"
    output_model = Premise

    def _execute(self, context: StageContext) -> Premise:
        result = dspy.Predict(PremiseSignature)(
            idea=context.idea,
            length=context.length.value,
            clarifications=artifact_json(context.artifacts.get("clarify", [])),
            feedback=context.feedback,
        )
        return parse_json_model(result.premise_json, Premise)
