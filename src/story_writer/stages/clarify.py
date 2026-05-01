from __future__ import annotations

import dspy

from story_writer.models.story import Clarification
from story_writer.stages.base import StageBase, StageContext, parse_json_list


class ClarifySignature(dspy.Signature):
    """Generate clarifying story-scope questions as JSON."""

    idea: str = dspy.InputField()
    length: str = dspy.InputField()
    questions_json: str = dspy.OutputField(
        desc=(
            "JSON array of 3 to 7 objects with keys question and suggested_answer. "
            "Questions should cover story size, premise, genre, quirks, tone, "
            "and constraints."
        )
    )


class ClarifyStage(StageBase):
    name = "clarify"
    output_model = Clarification

    def _execute(self, context: StageContext) -> list[Clarification]:
        result = dspy.Predict(ClarifySignature)(
            idea=context.idea,
            length=context.length.value,
        )
        return parse_json_list(result.questions_json, Clarification)
