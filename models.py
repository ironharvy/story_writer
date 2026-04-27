"""Pipeline dataclasses shared across orchestration modules."""

from dataclasses import dataclass
from typing import Any

from world_bible import WorldBible


@dataclass
class GenerationParams:
    """Model and runtime parameters recorded for reproducibility."""

    model: str = ""
    max_tokens: int = 0
    cache: bool = True
    memory_cache: bool = True
    cache_dir: str = ""

    def as_markdown(self) -> str:
        """Render parameters as a markdown bullet list."""
        return "\n".join(
            [
                f"- model: `{self.model}`",
                f"- max_tokens: `{self.max_tokens}`",
                f"- cache: `{self.cache}`",
                f"- memory_cache: `{self.memory_cache}`",
                f"- cache_dir: `{self.cache_dir}`",
            ]
        )


@dataclass
class ImageArtifacts:
    """Image-related outputs collected during the run."""

    character_visuals: list[Any]
    character_portrait_paths: dict[str, str]
    character_visuals_summary: str
    image_generator: Any | None


@dataclass
class StoryFoundation:
    """Core narrative artifacts produced before story drafting."""

    core_premise: str
    spine_template: str
    world_bible: WorldBible


@dataclass
class StoryRunArtifacts:
    """Collected artifacts needed for final output persistence."""

    core_premise: str
    spine_template: str
    world_bible: WorldBible
    image_artifacts: ImageArtifacts
    story_result: Any
    final_story_text: str
    scene_image_paths: dict[int, str]
