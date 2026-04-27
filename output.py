"""File output helpers for persisting story artifacts to markdown."""

import logging
import os
from typing import Any

from models import GenerationParams, ImageArtifacts, StoryRunArtifacts

logger = logging.getLogger(__name__)


def initialize_artifact(output_file: str) -> None:
    """Create (or reset) the incremental output file with a top-level heading."""
    dir_name = os.path.dirname(output_file)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Story\n\n")


def update_artifact(
    output_file: str,
    section: str,
    value: str,
    level: int = 2,
) -> None:
    """Append a markdown section to the incremental output file."""
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"{'#' * level} {section}\n\n{value}\n\n")


def write_generation_parameters(
    output_file: str,
    params: GenerationParams,
) -> None:
    """Write model/runtime parameters as the first section of the artifact file."""
    update_artifact(output_file, "Generation Parameters", params.as_markdown())


def _write_character_visuals_section(
    file_handle: Any,
    image_artifacts: ImageArtifacts,
) -> None:
    """Write optional character visual section to output file."""
    if not image_artifacts.character_visuals:
        return

    file_handle.write("## Character Visuals\n\n")
    for visual in image_artifacts.character_visuals:
        file_handle.write(f"### {visual.name}\n")
        file_handle.write(f"**Reference:** {visual.reference_mix}\n\n")
        file_handle.write(f"**Features:** {visual.distinguishing_features}\n\n")
        portrait = image_artifacts.character_portrait_paths.get(visual.name)
        if portrait:
            file_handle.write(f"![{visual.name} portrait]({portrait})\n\n")


def _write_story_metadata_section(file_handle: Any, story_result: Any) -> None:
    """Write story metadata sections to output file."""
    file_handle.write("## Chapter Plan\n")
    file_handle.write(f"{story_result.chapter_plan}\n\n")
    file_handle.write("## Enhancers Guide\n")
    file_handle.write(f"{story_result.enhancers_guide}\n\n")
    file_handle.write("## Final Story\n")


def _write_final_story_section(
    file_handle: Any,
    final_story_text: str,
    scene_image_paths: dict[int, str],
) -> None:
    """Write final story body and optional scene images."""
    if not scene_image_paths:
        file_handle.write(f"{final_story_text}\n")
        return

    chapters = [c for c in final_story_text.split("### Chapter ") if c.strip()]
    for index, chapter_text in enumerate(chapters, start=1):
        file_handle.write(f"\n\n### Chapter {chapter_text}")
        scene = scene_image_paths.get(index)
        if scene:
            file_handle.write(f"\n\n![Chapter {index} scene]({scene})\n")


def save_story_output(output_dir: str, artifacts: StoryRunArtifacts) -> str:
    """Write the generated story and artifacts to markdown output file."""
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "story_output.md")
    logger.info("Saving story output to %s...", output_filename)

    with open(output_filename, "w", encoding="utf-8") as file_handle:
        file_handle.write("# Story Output\n\n")
        file_handle.write("## Core Premise\n")
        file_handle.write(f"{artifacts.core_premise}\n\n")
        file_handle.write("## Spine Template\n")
        file_handle.write(f"{artifacts.spine_template}\n\n")
        file_handle.write("## World Bible\n")
        file_handle.write(f"{artifacts.world_bible.full_text}\n\n")
        _write_character_visuals_section(file_handle, artifacts.image_artifacts)
        _write_story_metadata_section(file_handle, artifacts.story_result)
        _write_final_story_section(
            file_handle,
            artifacts.final_story_text,
            artifacts.scene_image_paths,
        )

    return output_filename
