from __future__ import annotations

import dspy

from story_writer.models.story import WorldBible
from story_writer.stages.base import (
    StageBase,
    StageContext,
    artifact_json,
    parse_json_model,
)


class WorldBibleSignature(dspy.Signature):
    """Create the authoritative world bible as JSON."""

    premise: str = dspy.InputField()
    spine: str = dspy.InputField()
    length: str = dspy.InputField()
    feedback: str = dspy.InputField()
    world_bible_json: str = dspy.OutputField(
        desc=(
            "JSON object with logline, characters, locations, rules, timeline, "
            "continuity_constraints. Characters have name, role, traits, facts. "
            "Locations have name, description, facts. "
            "Timeline events have order and event."
        )
    )


class WorldBibleStage(StageBase):
    name = "world_bible"
    output_model = WorldBible

    def _execute(self, context: StageContext) -> WorldBible:
        result = dspy.Predict(WorldBibleSignature)(
            premise=artifact_json(context.artifacts["premise"]),
            spine=artifact_json(context.artifacts["spine"]),
            length=context.length.value,
            feedback=context.feedback,
        )
        return parse_json_model(result.world_bible_json, WorldBible)
