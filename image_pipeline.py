"""Image generation pipeline shared by the interactive run and the standalone
script.

The functions here consume/produce :class:`story_artifacts.StoryArtifacts`
objects so images can be generated either inline during a fresh story run or
after the fact from a saved JSON sidecar.  The concrete image backend is
injected (currently :class:`image_gen.ImageGenerator`), so a future
diffusers-based backend just needs to implement the same two methods:
``generate_character_portrait`` and ``generate_scene_illustration``.
"""

from __future__ import annotations

import logging
from typing import Iterable, Protocol

from story_artifacts import CharacterVisualData, StoryArtifacts

logger = logging.getLogger(__name__)


class ImageBackend(Protocol):
    """Duck-typed interface any image backend must satisfy."""

    def generate_character_portrait(self, prompt: str, character_name: str) -> str: ...

    def generate_scene_illustration(
        self,
        prompt: str,
        reference_image_paths: list[str],
        chapter_index: int,
    ) -> str: ...


def describe_characters(world_bible: str) -> list[CharacterVisualData]:
    """Run the DSPy ``CharacterVisualDescriber`` over a world bible."""
    from story_modules import CharacterVisualDescriber

    describer = CharacterVisualDescriber()
    result = describer(world_bible=world_bible)
    return [
        CharacterVisualData(
            name=cv.name,
            reference_mix=cv.reference_mix,
            distinguishing_features=cv.distinguishing_features,
            full_prompt=cv.full_prompt,
        )
        for cv in result.character_visuals
    ]


def build_character_visuals_summary(
    character_visuals: Iterable[CharacterVisualData],
) -> str:
    """One-line-per-character summary fed to the scene prompt generator."""
    return "\n".join(
        f"- {cv.name}: {cv.reference_mix}. {cv.distinguishing_features}"
        for cv in character_visuals
    )


def split_story_chapters(final_story: str) -> list[str]:
    """Split the final story text by ``### Chapter `` markers.

    Each returned string still begins with the chapter's number/title payload
    (the literal marker prefix is stripped), matching the convention used by
    the main pipeline and the markdown renderer.
    """
    chapters = final_story.split("### Chapter ")
    return [c for c in chapters if c.strip()]


def generate_character_portraits(
    character_visuals: Iterable[CharacterVisualData],
    image_backend: ImageBackend,
    *,
    existing_paths: dict[str, str] | None = None,
    skip_existing: bool = True,
) -> dict[str, str]:
    """Generate portraits for each character, returning name->path."""
    paths: dict[str, str] = dict(existing_paths or {})
    for cv in character_visuals:
        if skip_existing and cv.name in paths:
            logger.debug(
                "Skipping portrait for %s (already present: %s)",
                cv.name,
                paths[cv.name],
            )
            continue
        try:
            path = image_backend.generate_character_portrait(
                prompt=cv.full_prompt, character_name=cv.name
            )
            paths[cv.name] = path
            logger.info("Generated portrait for %s -> %s", cv.name, path)
        except Exception as exc:
            logger.error("Failed to generate portrait for %s: %s", cv.name, exc)
    return paths


def generate_scene_illustrations(
    final_story: str,
    character_visuals_summary: str,
    reference_image_paths: list[str],
    image_backend: ImageBackend,
    *,
    scene_prompt_gen=None,
    existing_paths: dict[int, str] | None = None,
    skip_existing: bool = True,
) -> dict[int, str]:
    """Generate a scene illustration per chapter, returning index->path."""
    if scene_prompt_gen is None:
        from story_modules import SceneImagePromptGenerator

        scene_prompt_gen = SceneImagePromptGenerator()

    paths: dict[int, str] = dict(existing_paths or {})
    chapters = split_story_chapters(final_story)
    for i, chapter_text in enumerate(chapters, start=1):
        if skip_existing and i in paths:
            logger.debug(
                "Skipping scene for chapter %d (already present: %s)",
                i,
                paths[i],
            )
            continue
        try:
            prompt_result = scene_prompt_gen(
                chapter_text=chapter_text,
                character_visuals_summary=character_visuals_summary,
            )
            path = image_backend.generate_scene_illustration(
                prompt=prompt_result.image_prompt,
                reference_image_paths=reference_image_paths,
                chapter_index=i,
            )
            paths[i] = path
            logger.info("Generated scene for chapter %d -> %s", i, path)
        except Exception as exc:
            logger.error("Failed to generate scene for chapter %d: %s", i, exc)
    return paths


def generate_images_for_story(
    artifacts: StoryArtifacts,
    image_backend: ImageBackend,
    *,
    describe_missing: bool = True,
    skip_portraits: bool = False,
    skip_scenes: bool = False,
    skip_existing: bool = True,
) -> StoryArtifacts:
    """Fill in character visuals, portraits, and scene images on ``artifacts``.

    Mutates ``artifacts`` in place and returns it for chaining.  If
    ``describe_missing`` is True and the artifact has no character visuals
    yet, DSPy is called to derive them from the world bible.
    """
    if describe_missing and not artifacts.character_visuals:
        if not artifacts.world_bible:
            logger.warning("Cannot describe characters: world bible is empty.")
        else:
            logger.info("Describing character visuals from world bible...")
            artifacts.character_visuals = describe_characters(artifacts.world_bible)

    if not skip_portraits and artifacts.character_visuals:
        artifacts.character_portrait_paths = generate_character_portraits(
            artifacts.character_visuals,
            image_backend,
            existing_paths=artifacts.character_portrait_paths,
            skip_existing=skip_existing,
        )

    if not skip_scenes and artifacts.final_story:
        summary = build_character_visuals_summary(artifacts.character_visuals)
        reference_paths = list(artifacts.character_portrait_paths.values())
        artifacts.scene_image_paths = generate_scene_illustrations(
            artifacts.final_story,
            summary,
            reference_paths,
            image_backend,
            existing_paths=artifacts.scene_image_paths,
            skip_existing=skip_existing,
        )

    return artifacts
