from __future__ import annotations

from story_writer.config import routing_for
from story_writer.stages.base import StageBase, StageContext
from story_writer.stages.chapter_plan import ChapterPlanStage
from story_writer.stages.clarify import ClarifyStage
from story_writer.stages.embellish import EmbellishStage
from story_writer.stages.enhancement import EnhancementStage
from story_writer.stages.premise import PremiseStage
from story_writer.stages.prose import ProseStage
from story_writer.stages.spine import SpineStage
from story_writer.stages.world_bible import WorldBibleStage

GENERATION_STAGE_TYPES: tuple[type[StageBase], ...] = (
    ClarifyStage,
    PremiseStage,
    SpineStage,
    WorldBibleStage,
    ChapterPlanStage,
    EnhancementStage,
)


def build_stage(
    stage_type: type[StageBase], model_override: str | None = None
) -> StageBase:
    return stage_type(routing_for(stage_type.name, model_override=model_override))


__all__ = [
    "ChapterPlanStage",
    "ClarifyStage",
    "EmbellishStage",
    "EnhancementStage",
    "GENERATION_STAGE_TYPES",
    "PremiseStage",
    "ProseStage",
    "SpineStage",
    "StageBase",
    "StageContext",
    "WorldBibleStage",
    "build_stage",
]
