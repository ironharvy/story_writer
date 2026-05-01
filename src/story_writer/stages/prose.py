from __future__ import annotations

import dspy

from story_writer.models.story import Chapter
from story_writer.stages.base import StageBase, StageContext, artifact_json


class ProseSignature(dspy.Signature):
    """Write chapter prose from locked story artifacts."""

    premise: str = dspy.InputField()
    spine: str = dspy.InputField()
    world_bible: str = dspy.InputField()
    chapter: str = dspy.InputField()
    enhancement_notes: str = dspy.InputField()
    embellishment: str = dspy.InputField()
    prior_summary: str = dspy.InputField(
        desc="Short summary of recent prior chapters for continuity."
    )
    prior_openings: str = dspy.InputField(
        desc="Prior chapter openings that must not be repeated or paraphrased."
    )
    qa_feedback: str = dspy.InputField(
        desc=(
            "QA findings from a failed draft. Any forbidden_spans listed here "
            "must be removed exactly from the rewrite."
        )
    )
    prose: str = dspy.OutputField(
        desc=(
            "Polished chapter prose only. Do not include meta commentary, outlines, "
            "assistant language, or markdown headings. Use only named people, "
            "places, and entities from the world bible. If a planned beat mentions "
            "an unnamed person, keep that person unnamed. Open with a chapter-specific "
            "moment that does not reuse prior chapter openings, imagery, or sentence "
            "structure. If QA feedback is provided, rewrite the chapter to resolve it. "
            "Never output any forbidden span listed in QA feedback."
        )
    )


class ProseStage(StageBase):
    name = "prose"
    output_model = Chapter

    def _execute(self, context: StageContext) -> Chapter:
        planned = context.artifacts["current_chapter"]
        notes = context.artifacts["current_enhancement"]
        embellishment = context.artifacts["current_embellishment"]
        result = dspy.Predict(ProseSignature)(
            premise=artifact_json(context.artifacts["premise"]),
            spine=artifact_json(context.artifacts["spine"]),
            world_bible=artifact_json(context.artifacts["world_bible"]),
            chapter=artifact_json(planned),
            enhancement_notes=artifact_json(notes),
            embellishment=artifact_json(embellishment),
            prior_summary=context.artifacts.get("prior_summary", ""),
            prior_openings=context.artifacts.get("prior_openings", ""),
            qa_feedback=context.artifacts.get("qa_feedback", ""),
        )
        return Chapter(
            number=planned.number,
            title=planned.title,
            beats=planned.beats,
            enhancement_notes=notes,
            embellishment=embellishment,
            prose=str(result.prose).strip(),
        )
