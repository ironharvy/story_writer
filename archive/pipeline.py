"""Story generation pipeline orchestration.

All interactive pipeline flows live here. UI calls are delegated to the
``ui`` module so that pipeline logic never imports Rich directly.
"""

import argparse
import logging
from argparse import Namespace
from collections.abc import Callable
from typing import Any

import dspy

from _compat import observe
from dspy_optimization import try_load_optimized_module
from exceptions import RECOVERABLE_RUNTIME_EXCEPTIONS
from image_gen import ImageGenerator
from models import ImageArtifacts, StoryFoundation, StoryRunArtifacts
from output import update_artifact
from story_modules import (
    ChapterInpaintingGenerator,
    GenerateCharacterVisualsSignature,
    GenerateCorePremiseSignature,
    GenerateQuestionsSignature,
    GenerateSceneImagePromptSignature,
    GenerateSpineTemplateSignature,
    StoryGenerator,
    SynthesizeEnrichedIdeaSignature,
)
from world_bible import WorldBible
from world_bible_modules import (
    EnhanceCharacterSignature,
    EnhanceLocationSignature,
    GenerateWorldBibleQuestionsSignature,
    WorldBibleGenerator,
)
import ui

logger = logging.getLogger(__name__)


def initialize_text_generators(
    *,
    use_optimized: bool = False,
    optimized_manifest: str | None = None,
) -> dict[str, Any]:
    """Initialize all text generation modules used by the CLI flow."""
    generators = {
        "QuestionGenerator": dspy.Predict(GenerateQuestionsSignature),
        "IdeaSynthesizer": dspy.Predict(SynthesizeEnrichedIdeaSignature),
        "CorePremiseGenerator": dspy.Predict(GenerateCorePremiseSignature),
        "SpineTemplateGenerator": dspy.Predict(GenerateSpineTemplateSignature),
        "WorldBibleQuestionGenerator": dspy.Predict(GenerateWorldBibleQuestionsSignature),
        "WorldBibleGenerator": WorldBibleGenerator(),
        "CharacterEnhancer": dspy.ChainOfThought(EnhanceCharacterSignature),
        "LocationEnhancer": dspy.ChainOfThought(EnhanceLocationSignature),
        "CharacterVisualDescriber": dspy.Predict(GenerateCharacterVisualsSignature),
        "SceneImagePromptGenerator": dspy.Predict(GenerateSceneImagePromptSignature),
        "StoryGenerator": StoryGenerator(),
        "ChapterInpaintingGenerator": ChapterInpaintingGenerator(),
    }

    if use_optimized:
        logger.info(
            "Optimized text modules enabled (manifest=%r)",
            optimized_manifest,
        )
        for module_name, module in generators.items():
            try_load_optimized_module(
                module,
                module_name=module_name,
                manifest_path=optimized_manifest,
                logger=logger,
            )

    return generators


@observe()
def run_idea_enrichment(
    idea: str,
    question_generator: Any,
    idea_synthesizer: Any,
    output_file: str | None = None,
) -> str:
    """Generate clarifying questions, collect user answers, return enriched idea."""
    ui.print_status("Generating questions to interrogate your idea...")
    q_result = question_generator(idea=idea)
    qa_text = ui.get_answers_for_questions(q_result.questions_with_answers)

    ui.print_status("Synthesizing enriched idea from your answers...")
    synth_result = idea_synthesizer(original_idea=idea, qa_text=qa_text)
    enriched_idea = synth_result.enriched_idea

    if output_file:
        update_artifact(output_file, "Enriched Idea", enriched_idea)
    return enriched_idea


@observe()
def run_core_premise_flow(
    idea: str,
    core_premise_generator: Any,
    output_file: str | None = None,
) -> str:
    """Refine core premise via dedicated feedback field."""
    feedback = ""
    while True:
        ui.print_status("Generating Core Premise...")
        cp_result = core_premise_generator(idea=idea, feedback=feedback)
        core_premise = cp_result.core_premise

        ui.print_section("Core Premise", core_premise, color="magenta")

        should_refine, refinement_details = ui.ask_refine("premise")
        if not should_refine:
            if output_file:
                update_artifact(output_file, "Core Premise", core_premise)
            return core_premise

        feedback = refinement_details


@observe()
def run_spine_template_flow(
    idea: str,
    core_premise: str,
    spine_template_generator: Any,
    output_file: str | None = None,
) -> str:
    """Refine spine template via dedicated feedback field."""
    feedback = ""
    while True:
        ui.print_status("Generating Spine Template...")
        st_result = spine_template_generator(
            idea=idea,
            core_premise=core_premise,
            feedback=feedback,
        )
        spine_template = st_result.spine_template

        ui.print_section("Spine Template", spine_template, color="blue")

        should_refine, refinement_details = ui.ask_refine("spine template")
        if not should_refine:
            if output_file:
                update_artifact(output_file, "Spine Template", spine_template)
            return spine_template

        feedback = refinement_details


@observe()
def generate_world_bible(
    core_premise: str,
    spine_template: str,
    world_bible_question_generator: Any,
    world_bible_generator: Any,
    output_file: str | None = None,
) -> WorldBible:
    """Generate world-bible follow-up QA and final world bible with refinement loop."""
    ui.print_status(
        "Generating follow-up questions to help flesh out the World Bible..."
    )
    wb_q_result = world_bible_question_generator(
        core_premise=core_premise,
        spine_template=spine_template,
    )
    wb_qa_text = ui.get_answers_for_questions(wb_q_result.questions_with_answers)

    user_additions = wb_qa_text
    while True:
        ui.print_status("Generating World Bible...")
        wb_result = world_bible_generator(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
        )
        world_bible = wb_result.world_bible_structured

        ui.print_world_bible_sections(world_bible)

        should_refine, refinement_details = ui.ask_refine("world bible")
        if not should_refine:
            if output_file:
                update_artifact(output_file, "World Bible", world_bible.full_text)
            return world_bible

        user_additions = (
            f"{wb_qa_text}\n\nRefinement feedback: {refinement_details}"
        )


def _split_section_items(text: str) -> list[str]:
    """Split a world-bible section into individual items on double newlines."""
    items = [item.strip() for item in text.split("\n\n") if item.strip()]
    return items if items else [text.strip()]


@observe()
def _enhance_single_item(
    enhance_fn: Callable[[str, str], str],
    item: str,
    index: int,
    subject: str,
) -> str:
    """Enhance one character or location with a feedback loop."""
    feedback = ""
    while True:
        ui.print_status(f"Enhancing {subject} {index}...")
        enhanced = enhance_fn(item, feedback)
        ui.print_item_for_review(index, enhanced, subject)

        should_refine, refinement_details = ui.ask_refine(f"this {subject}")
        if not should_refine:
            return enhanced
        feedback = refinement_details
        item = enhanced


def _enhance_section_items(
    section_text: str,
    subject: str,
    enhance_fn: Callable[[str, str], str],
    output_file: str | None = None,
) -> str:
    """Split a section, enhance each item with feedback, and rejoin."""
    if not ui.ask_enhance_items(subject + "s"):
        return section_text
    items = _split_section_items(section_text)
    enhanced: list[str] = []
    for i, item in enumerate(items, 1):
        result = _enhance_single_item(enhance_fn, item, i, subject)
        enhanced.append(result)
        if output_file:
            update_artifact(
                output_file, f"{subject.title()} {i}", result, level=3,
            )
    return "\n\n".join(enhanced)


@observe()
def _enhance_world_bible_items(
    world_bible: WorldBible,
    generators: dict[str, Any],
    core_premise: str,
    spine_template: str,
    output_file: str | None = None,
) -> WorldBible:
    """Offer per-character and per-location enhancement with feedback loops."""

    def _char_enhance(item: str, feedback: str) -> str:
        result = generators["CharacterEnhancer"](
            core_premise=core_premise,
            spine_template=spine_template,
            rules=world_bible.rules,
            character=item,
            feedback=feedback,
        )
        return result.enhanced_character

    def _loc_enhance(item: str, feedback: str) -> str:
        result = generators["LocationEnhancer"](
            core_premise=core_premise,
            spine_template=spine_template,
            rules=world_bible.rules,
            location=item,
            feedback=feedback,
        )
        return result.enhanced_location

    enhanced_characters = _enhance_section_items(
        world_bible.characters, "character", _char_enhance, output_file,
    )
    enhanced_locations = _enhance_section_items(
        world_bible.locations, "location", _loc_enhance, output_file,
    )

    return WorldBible(
        rules=world_bible.rules,
        characters=enhanced_characters,
        locations=enhanced_locations,
        plot_timeline=world_bible.plot_timeline,
    )


def _generate_character_portraits(
    image_generator: ImageGenerator,
    character_visuals: list[Any],
) -> dict[str, str]:
    """Generate and return portrait paths keyed by character name."""
    ui.print_status("Generating character portraits...")
    portrait_paths: dict[str, str] = {}
    for visual in character_visuals:
        try:
            path = image_generator.generate_character_portrait(
                prompt=visual.full_prompt,
                character_name=visual.name,
            )
            portrait_paths[visual.name] = path
            ui.print_portrait_saved(path)
        except RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Failed to generate portrait for %s: %s", visual.name, exc)
            ui.print_portrait_failed(visual.name, exc)
    return portrait_paths


def maybe_generate_character_assets(
    args: Namespace,
    world_bible: WorldBible,
) -> ImageArtifacts:
    """Generate character descriptions and portraits when image mode is enabled."""
    if not args.enable_images:
        return ImageArtifacts([], {}, "", None)

    if not args.replicate_api_token:
        ui.print_missing_replicate_token()
        return ImageArtifacts([], {}, "", None)

    image_generator = ImageGenerator(api_token=args.replicate_api_token)

    ui.print_status("Generating character visual descriptions...")
    cv_describer = dspy.Predict(GenerateCharacterVisualsSignature)
    cv_result = cv_describer(world_bible=world_bible.full_text)
    character_visuals = cv_result.character_visuals
    character_visuals_summary = ui.summarize_character_visuals(character_visuals)
    character_portrait_paths = _generate_character_portraits(
        image_generator=image_generator,
        character_visuals=character_visuals,
    )

    return ImageArtifacts(
        character_visuals=character_visuals,
        character_portrait_paths=character_portrait_paths,
        character_visuals_summary=character_visuals_summary,
        image_generator=image_generator,
    )


def _run_optional_inpainting(
    args: Namespace,
    parser: argparse.ArgumentParser,
    chapter_inpainting_generator: Any,
    story_result: Any,
    world_bible: str,
) -> str:
    """Run chapter inpainting if enabled and return final story text."""
    final_story_text = story_result.story
    if not args.inpaint_chapters:
        return final_story_text

    if args.inpaint_ratio <= 1.0:
        parser.error("--inpaint-ratio must be greater than 1.0")

    ui.print_status("Running chapter inpainting pass...")
    inpaint_result = chapter_inpainting_generator(
        story=story_result.story,
        world_bible=world_bible,
        chapter_plan=story_result.chapter_plan,
        expansion_ratio=args.inpaint_ratio,
    )
    ui.print_inpainting_result(
        story=inpaint_result.story,
        expanded=inpaint_result.expanded_chapters,
        total=inpaint_result.total_chapters,
    )
    return inpaint_result.story


@observe()
def _run_chapter_plan_flow(
    story_generator: Any,
    foundation: StoryFoundation,
    output_file: str | None = None,
) -> list[str]:
    """Generate chapter plan with user feedback loop."""
    feedback = ""
    while True:
        ui.print_status("Generating Chapter Plan...")
        chapters_to_write = story_generator.generate_chapter_plan_entries(
            core_premise=foundation.core_premise,
            spine_template=foundation.spine_template,
            world_bible=foundation.world_bible,
            feedback=feedback,
        )
        ui.print_chapter_plan_entries(chapters_to_write)

        should_refine, refinement_details = ui.ask_refine("chapter plan")
        if not should_refine:
            if output_file:
                plan_text = "\n".join(chapters_to_write)
                update_artifact(output_file, "Chapter Plan", plan_text)
            return chapters_to_write

        feedback = refinement_details


@observe()
def generate_story_text(
    args: Namespace,
    parser: argparse.ArgumentParser,
    generators: dict[str, Any],
    foundation: StoryFoundation,
    output_file: str | None = None,
) -> tuple[Any, str]:
    """Generate chapter plan (with feedback), enhancers, story, and optional inpainting."""
    story_gen = generators["StoryGenerator"]

    chapters_to_write = _run_chapter_plan_flow(
        story_generator=story_gen,
        foundation=foundation,
        output_file=output_file,
    )
    chapter_plan_text = "\n".join(chapters_to_write)

    ui.print_status("Generating Enhancers Guide...")
    enhancers_guide = story_gen.generate_enhancers_guide(
        world_bible=foundation.world_bible,
        chapter_plan_text=chapter_plan_text,
    )
    if output_file:
        update_artifact(output_file, "Enhancers Guide", enhancers_guide)

    ui.print_status("Writing Story Chapters...")
    full_story = story_gen.write_story_chapters(
        world_bible=foundation.world_bible,
        chapter_plan_text=chapter_plan_text,
        chapters_to_write=chapters_to_write,
        enhancers_guide=enhancers_guide,
    )

    story_result = dspy.Prediction(
        chapter_plan=chapter_plan_text,
        enhancers_guide=enhancers_guide,
        story=full_story.strip(),
    )
    ui.print_story_sections(story_result)

    final_story_text = _run_optional_inpainting(
        args=args,
        parser=parser,
        chapter_inpainting_generator=generators["ChapterInpaintingGenerator"],
        story_result=story_result,
        world_bible=foundation.world_bible.full_text,
    )
    if output_file:
        update_artifact(output_file, "Final Story", final_story_text)
    return story_result, final_story_text


def maybe_generate_scene_images(
    args: Namespace,
    final_story_text: str,
    image_artifacts: ImageArtifacts,
) -> dict[int, str]:
    """Generate per-chapter scene images when image mode is enabled."""
    if not args.enable_images or image_artifacts.image_generator is None:
        return {}

    ui.print_status("Generating scene illustrations for each chapter...")
    scene_prompt_gen = dspy.Predict(GenerateSceneImagePromptSignature)

    chapters = [c for c in final_story_text.split("### Chapter ") if c.strip()]
    reference_paths = list(image_artifacts.character_portrait_paths.values())

    scene_image_paths: dict[int, str] = {}
    for i, chapter_text in enumerate(chapters, start=1):
        try:
            prompt_result = scene_prompt_gen(
                chapter_text=chapter_text,
                character_visuals_summary=image_artifacts.character_visuals_summary,
            )
            path = image_artifacts.image_generator.generate_scene_illustration(
                prompt=prompt_result.image_prompt,
                reference_image_paths=reference_paths,
                chapter_index=i,
            )
            scene_image_paths[i] = path
            ui.print_scene_saved(i, path)
        except RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            ui.print_scene_failed(i, exc)
    return scene_image_paths


@observe()
def _build_story_foundation(
    generators: dict[str, Any],
    idea: str,
    output_file: str | None = None,
) -> StoryFoundation:
    """Run premise, spine, and world bible stages."""
    enriched_idea = run_idea_enrichment(
        idea=idea,
        question_generator=generators["QuestionGenerator"],
        idea_synthesizer=generators["IdeaSynthesizer"],
        output_file=output_file,
    )
    core_premise = run_core_premise_flow(
        idea=enriched_idea,
        core_premise_generator=generators["CorePremiseGenerator"],
        output_file=output_file,
    )
    spine_template = run_spine_template_flow(
        idea=enriched_idea,
        core_premise=core_premise,
        spine_template_generator=generators["SpineTemplateGenerator"],
        output_file=output_file,
    )
    world_bible = generate_world_bible(
        core_premise=core_premise,
        spine_template=spine_template,
        world_bible_question_generator=generators["WorldBibleQuestionGenerator"],
        world_bible_generator=generators["WorldBibleGenerator"],
        output_file=output_file,
    )
    world_bible = _enhance_world_bible_items(
        world_bible=world_bible,
        generators=generators,
        core_premise=core_premise,
        spine_template=spine_template,
        output_file=output_file,
    )
    return StoryFoundation(
        core_premise=core_premise,
        spine_template=spine_template,
        world_bible=world_bible,
    )


@observe()
def _run_story_generation_phase(
    args: Namespace,
    parser: argparse.ArgumentParser,
    generators: dict[str, Any],
    foundation: StoryFoundation,
    image_artifacts: ImageArtifacts,
    output_file: str | None = None,
) -> tuple[Any, str, dict[int, str]]:
    """Run story generation and scene image generation."""
    story_result, final_story_text = generate_story_text(
        args=args,
        parser=parser,
        generators=generators,
        foundation=foundation,
        output_file=output_file,
    )
    scene_image_paths = maybe_generate_scene_images(
        args=args,
        final_story_text=final_story_text,
        image_artifacts=image_artifacts,
    )
    return story_result, final_story_text, scene_image_paths


@observe()
def run_story_workflow(
    args: Namespace,
    parser: argparse.ArgumentParser,
    generators: dict[str, Any],
    idea: str,
    output_file: str | None = None,
) -> StoryRunArtifacts:
    """Run the interactive story pipeline and collect output artifacts."""
    foundation = _build_story_foundation(
        generators=generators,
        idea=idea,
        output_file=output_file,
    )
    image_artifacts = maybe_generate_character_assets(
        args=args,
        world_bible=foundation.world_bible,
    )
    ui.ask_continue()
    story_result, final_story_text, scene_image_paths = _run_story_generation_phase(
        args=args,
        parser=parser,
        generators=generators,
        foundation=foundation,
        image_artifacts=image_artifacts,
        output_file=output_file,
    )
    return StoryRunArtifacts(
        core_premise=foundation.core_premise,
        spine_template=foundation.spine_template,
        world_bible=foundation.world_bible,
        image_artifacts=image_artifacts,
        story_result=story_result,
        final_story_text=final_story_text,
        scene_image_paths=scene_image_paths,
    )
