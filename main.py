"""CLI orchestration for the Story Writer pipeline."""

import argparse
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import dspy
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Confirm, Prompt

from dspy_optimization import try_load_optimized_module
from image_gen import ImageGenerator
from logging_config import TokenUsageCallback, setup_logging
from postprocessing import find_similar_sentences, format_report
from story_modules import (
    ChapterInpaintingGenerator,
    CharacterVisualDescriber,
    CorePremiseGenerator,
    QuestionGenerator,
    SceneImagePromptGenerator,
    SpineTemplateGenerator,
    StoryGenerator,
)
from world_bible import WorldBible
from world_bible_modules import WorldBibleGenerator, WorldBibleQuestionGenerator

try:
    from openinference.instrumentation.dspy import DSPyInstrumentor
except ImportError:
    DSPyInstrumentor = None

load_dotenv()

logger = logging.getLogger(__name__)

console = Console()

_RECOVERABLE_RUNTIME_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    KeyError,
    IndexError,
    OSError,
)


@dataclass
class DSPyConfig:
    """Runtime configuration for DSPy language model setup."""

    model_name: str
    api_base: str | None = None
    api_key: str | None = None
    max_tokens: int = 2000
    cache: bool = True
    memory_cache: bool = True
    cache_dir: str | None = None


def dspy_config_from_namespace(args: Namespace) -> DSPyConfig:
    """Build DSPy runtime config from parsed CLI arguments."""
    return DSPyConfig(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        cache=args.cache,
        memory_cache=args.memory_cache,
        cache_dir=args.cache_dir,
    )


def _env_flag_true(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def configure_dspy(config: DSPyConfig) -> None:
    """Configure DSPy LM and optional instrumentation."""
    model_name = config.model_name
    kwargs = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment variables. "
                "Assuming mock or alternative setup."
            )
        else:
            kwargs["api_key"] = env_key
    elif "ollama" in model_name.lower():
        pass
        #kwargs["api_key"] = "" # Ollama typically doesn't need an API key

    dspy.configure_cache(
        enable_disk_cache=config.cache,
        enable_memory_cache=config.memory_cache,
        disk_cache_dir=config.cache_dir,
    )

    logger.info(
        "Configuring DSPy model=%r max_tokens=%s cache=%s memory_cache=%s cache_dir=%r",
        model_name,
        config.max_tokens,
        config.cache,
        config.memory_cache,
        config.cache_dir,
    )
    lm = dspy.LM(
        model_name,
        max_tokens=config.max_tokens,
        cache=config.cache,
        **kwargs,
    )

    if os.environ.get("LANGFUSE_PUBLIC_KEY") and DSPyInstrumentor is not None:
        try:
            DSPyInstrumentor().instrument()
        except _RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Langfuse DSPy callback handler unavailable: %s", exc)

    callbacks = [TokenUsageCallback()] if TokenUsageCallback is not None else []
    dspy.configure(lm=lm, callbacks=callbacks)


def initialize_text_generators(
    *,
    use_optimized: bool = False,
    optimized_manifest: str | None = None,
) -> dict[str, Any]:
    """Initialize all text generation modules used by the CLI flow."""
    generators = {
        "QuestionGenerator": QuestionGenerator(),
        "CorePremiseGenerator": CorePremiseGenerator(),
        "SpineTemplateGenerator": SpineTemplateGenerator(),
        "WorldBibleQuestionGenerator": WorldBibleQuestionGenerator(),
        "WorldBibleGenerator": WorldBibleGenerator(),
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

def get_answers_for_questions(questions_with_answers: Sequence[Any]) -> str:
    """Collect accepted or user-edited answers for generated questions."""
    qa_pairs = []
    for i, qa in enumerate(questions_with_answers):
        console.print(f"\n[bold cyan]Question {i+1}:[/bold cyan] {qa.question}")
        console.print(f"[bold green]Proposed Answer:[/bold green] {qa.proposed_answer}")

        accept = Confirm.ask("Accept this proposed answer?")
        if accept:
            user_answer = qa.proposed_answer
        else:
            user_answer = Prompt.ask("Enter your answer")

        qa_pairs.append(f"Q: {qa.question}\nA: {user_answer}")

    return "\n\n".join(qa_pairs)


@dataclass
class ImageArtifacts:
    """Image-related outputs collected during the run."""

    character_visuals: list[Any]
    character_portrait_paths: dict[str, str]
    character_visuals_summary: str
    image_generator: Any | None


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model/provider-related arguments to parser."""
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "openai/gpt-4o-mini"),
        help=(
            "The language model to use (e.g., openai/gpt-4o-mini, "
            "ollama_chat/llama3). Defaults to MODEL env var."
        ),
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=os.environ.get("LLM_URL"),
        help=(
            "The custom API base URL (e.g., http://localhost:11434 for Ollama). "
            "Defaults to LLM_URL env var."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY"),
        help="The API key for the model. Defaults to API_KEY env var.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="The maximum number of tokens to use for the model. Defaults to 8000.",
    )
    parser.add_argument(
        "--enable-images",
        action="store_true",
        default=False,
        help="Enable image generation (requires Replicate API token).",
    )
    parser.add_argument(
        "--replicate-api-token",
        type=str,
        default=os.environ.get("REPLICATE_API_TOKEN"),
        help="Replicate API token. Defaults to REPLICATE_API_TOKEN env var.",
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime, cache, and optimization arguments to parser."""
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DSPy disk cache.",
    )
    parser.add_argument(
        "--memory-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DSPy in-memory cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("DSPY_CACHE_DIR"),
        help="Override DSPy disk cache directory.",
    )
    parser.add_argument(
        "--use-optimized",
        action=argparse.BooleanOptionalAction,
        default=_env_flag_true("DSPY_USE_OPTIMIZED", default=False),
        help="Enable/disable loading optimized text-pipeline module artifacts.",
    )
    parser.add_argument(
        "--optimized-manifest",
        type=str,
        default=os.environ.get(
            "DSPY_OPTIMIZED_MANIFEST",
            ".tmp/dspy_optimized/text_pipeline_manifest.json",
        ),
        help="Path to optimized text-pipeline manifest JSON.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.environ.get("LOG_FILE", ".tmp/test_debug.log"),
        help=(
            "Path to write detailed logs (default: LOG_FILE env var or "
            ".tmp/test_debug.log)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv full firehose.",
    )


def _add_output_and_quality_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output and post-processing quality arguments to parser."""
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".tmp",
        help="Directory to save generated content. Defaults to '.tmp'.",
    )
    parser.add_argument(
        "--check-similar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run similar-sentence detection on the final story (default: enabled).",
    )
    parser.add_argument(
        "--similar-threshold",
        type=float,
        default=0.65,
        help="Similarity threshold (0-1) for flagging sentence pairs (default: 0.65).",
    )
    parser.add_argument(
        "--inpaint-chapters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run a post-generation chapter expansion pass for richer detail "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--inpaint-ratio",
        type=float,
        default=1.35,
        help=(
            "Target chapter expansion ratio for inpainting "
            "(must be > 1.0, default: 1.35)."
        ),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="AI DSPy Story Writer")
    _add_model_arguments(parser)
    _add_runtime_arguments(parser)
    _add_output_and_quality_arguments(parser)
    return parser


def setup_runtime(args: Namespace) -> None:
    """Configure logging and DSPy runtime from parsed CLI arguments."""
    setup_logging(verbosity=args.verbose, log_file=args.log_file)
    configure_dspy(dspy_config_from_namespace(args))


def run_core_premise_flow(
    idea: str,
    question_generator: Any,
    core_premise_generator: Any,
) -> tuple[str, str, str]:
    """Run iterative questioning and core-premise refinement loop."""
    core_premise = ""
    qa_text = ""

    while True:
        console.print("\n[italic]Generating questions to interrogate your idea...[/italic]")
        q_result = question_generator(idea=idea)
        qa_text = get_answers_for_questions(q_result.questions_with_answers)

        console.print("\n[italic]Generating Core Premise...[/italic]")
        cp_result = core_premise_generator(idea=idea, qa_pairs=qa_text)
        core_premise = cp_result.core_premise

        console.print("\n[bold magenta]--- Core Premise ---[/bold magenta]")
        console.print(core_premise)
        console.print("[bold magenta]--------------------[/bold magenta]")

        should_refine = Confirm.ask(
            (
                "Do you want to refine this premise? (Choosing 'Yes' will let you "
                "provide more details and regenerate, 'No' proceeds)"
            ),
            default=False,
        )
        if not should_refine:
            return idea, core_premise, qa_text

        refinement_details = Prompt.ask("Provide more details or changes")
        idea = (
            f"Original idea: {idea}\n"
            f"Refinements: {refinement_details}\n"
            f"Current Core Premise: {core_premise}"
        )


def run_spine_template_flow(
    idea: str,
    qa_text: str,
    core_premise: str,
    spine_template_generator: Any,
) -> tuple[str, str]:
    """Run iterative spine-template refinement loop."""
    spine_template = ""

    while True:
        console.print("\n[italic]Generating Spine Template...[/italic]")
        st_result = spine_template_generator(
            idea=idea,
            qa_pairs=qa_text,
            core_premise=core_premise,
        )
        spine_template = st_result.spine_template

        console.print("\n[bold blue]--- Spine Template ---[/bold blue]")
        console.print(spine_template)
        console.print("[bold blue]--------------------[/bold blue]")

        should_refine = Confirm.ask(
            (
                "Do you want to refine this spine template? (Choosing 'Yes' will "
                "let you provide more details and regenerate, 'No' proceeds)"
            ),
            default=False,
        )
        if not should_refine:
            return idea, spine_template

        refinement_details = Prompt.ask("Provide more details or changes")
        idea = (
            f"Original idea: {idea}\n"
            f"Refinements for Spine Template: {refinement_details}\n"
            f"Current Spine Template: {spine_template}"
        )


def generate_world_bible(
    core_premise: str,
    spine_template: str,
    world_bible_question_generator: Any,
    world_bible_generator: Any,
) -> WorldBible:
    """Generate world-bible follow-up QA and final world bible."""
    console.print(
        "\n[italic]Generating follow-up questions to help flesh out "
        "the World Bible...[/italic]"
    )
    wb_q_result = world_bible_question_generator(
        core_premise=core_premise,
        spine_template=spine_template,
    )
    wb_qa_text = get_answers_for_questions(wb_q_result.questions_with_answers)

    console.print("\n[italic]Generating World Bible...[/italic]")
    wb_result = world_bible_generator(
        core_premise=core_premise,
        spine_template=spine_template,
        user_additions=wb_qa_text,
    )
    world_bible = wb_result.world_bible_structured

    console.print("\n[bold green]--- World Bible ---[/bold green]")
    console.print(world_bible.full_text)
    console.print("[bold green]-------------------[/bold green]")
    return world_bible


def _summarize_character_visuals(character_visuals: list[Any]) -> str:
    """Render and summarize character visuals for prompt reuse."""
    lines = []
    for visual in character_visuals:
        console.print(f"\n[bold cyan]{visual.name}:[/bold cyan] {visual.reference_mix}")
        console.print(f"  [dim]{visual.distinguishing_features}[/dim]")
        lines.append(
            f"- {visual.name}: {visual.reference_mix}. "
            f"{visual.distinguishing_features}"
        )
    return "\n".join(lines)


def _generate_character_portraits(
    image_generator: ImageGenerator,
    character_visuals: list[Any],
) -> dict[str, str]:
    """Generate and return portrait paths keyed by character name."""
    console.print("\n[italic]Generating character portraits...[/italic]")
    portrait_paths: dict[str, str] = {}
    for visual in character_visuals:
        try:
            path = image_generator.generate_character_portrait(
                prompt=visual.full_prompt,
                character_name=visual.name,
            )
            portrait_paths[visual.name] = path
            console.print(f"  [green]Saved portrait:[/green] {path}")
        except _RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Failed to generate portrait for %s: %s", visual.name, exc)
            console.print(
                f"  [red]Failed to generate portrait for {visual.name}: {exc}[/red]"
            )
    return portrait_paths


def maybe_generate_character_assets(
    args: Namespace,
    world_bible: WorldBible,
) -> ImageArtifacts:
    """Generate character descriptions and portraits when image mode is enabled."""
    if not args.enable_images:
        return ImageArtifacts([], {}, "", None)

    if not args.replicate_api_token:
        console.print(
            "[bold red]Error: --enable-images requires a Replicate API token. "
            "Set REPLICATE_API_TOKEN env var or pass --replicate-api-token.[/bold red]"
        )
        return ImageArtifacts([], {}, "", None)

    image_generator = ImageGenerator(api_token=args.replicate_api_token)

    console.print("\n[italic]Generating character visual descriptions...[/italic]")
    cv_describer = CharacterVisualDescriber()
    cv_result = cv_describer(world_bible=world_bible.full_text)
    character_visuals = cv_result.character_visuals
    character_visuals_summary = _summarize_character_visuals(character_visuals)
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


def _print_story_sections(story_result: Any) -> None:
    """Print generated story planning artifacts and final story."""
    console.print("\n[bold red]--- Chapter Plan ---[/bold red]")
    console.print(story_result.chapter_plan)
    console.print("\n[bold red]--- Enhancers Guide ---[/bold red]")
    console.print(story_result.enhancers_guide)
    console.print("\n[bold red]--- Final Story ---[/bold red]")
    console.print(story_result.story)


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

    console.print("\n[italic]Running chapter inpainting pass...[/italic]")
    inpaint_result = chapter_inpainting_generator(
        story=story_result.story,
        world_bible=world_bible,
        chapter_plan=story_result.chapter_plan,
        expansion_ratio=args.inpaint_ratio,
    )
    console.print(
        "[italic]Chapter inpainting complete "
        f"({inpaint_result.expanded_chapters}/{inpaint_result.total_chapters} "
        "chapters expanded).[/italic]"
    )
    console.print("\n[bold red]--- Inpainted Story ---[/bold red]")
    console.print(inpaint_result.story)
    return inpaint_result.story


@dataclass
class StoryFoundation:
    """Core narrative artifacts produced before story drafting."""

    core_premise: str
    spine_template: str
    world_bible: WorldBible


def generate_story_text(
    args: Namespace,
    parser: argparse.ArgumentParser,
    generators: dict[str, Any],
    foundation: StoryFoundation,
) -> tuple[Any, str]:
    """Generate story and optionally run chapter inpainting."""
    console.print(
        "\n[italic]Generating Story (Chapter Plan, Final Story)...[/italic]"
    )
    story_result = generators["StoryGenerator"](
        core_premise=foundation.core_premise,
        spine_template=foundation.spine_template,
        world_bible=foundation.world_bible,
    )
    _print_story_sections(story_result)
    final_story_text = _run_optional_inpainting(
        args=args,
        parser=parser,
        chapter_inpainting_generator=generators["ChapterInpaintingGenerator"],
        story_result=story_result,
        world_bible=foundation.world_bible.full_text,
    )
    return story_result, final_story_text


def maybe_generate_scene_images(
    args: Namespace,
    final_story_text: str,
    image_artifacts: ImageArtifacts,
) -> dict[int, str]:
    """Generate per-chapter scene images when image mode is enabled."""
    if not args.enable_images or image_artifacts.image_generator is None:
        return {}

    console.print("\n[italic]Generating scene illustrations for each chapter...[/italic]")
    scene_prompt_gen = SceneImagePromptGenerator()

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
            console.print(f"  [green]Chapter {i} scene:[/green] {path}")
        except _RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            console.print(f"  [red]Failed to generate scene for chapter {i}: {exc}[/red]")
    return scene_image_paths


def run_similarity_check(args: Namespace, final_story_text: str) -> None:
    """Run final-story similar sentence detection and log warnings."""
    if not args.check_similar:
        return

    console.print("\n[bold yellow]--- Similar Sentence Check ---[/bold yellow]")
    similar_pairs = find_similar_sentences(
        final_story_text,
        threshold=args.similar_threshold,
    )
    report = format_report(similar_pairs)
    console.print(report)
    if similar_pairs:
        logger.warning(
            "Detected %d similar sentence pair(s) in generated story",
            len(similar_pairs),
        )


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


def save_story_output(output_dir: str, artifacts: "StoryRunArtifacts") -> str:
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


def _build_story_foundation(
    generators: dict[str, Any],
    idea: str,
) -> StoryFoundation:
    """Run premise, spine, and world bible stages."""
    idea, core_premise, qa_text = run_core_premise_flow(
        idea=idea,
        question_generator=generators["QuestionGenerator"],
        core_premise_generator=generators["CorePremiseGenerator"],
    )
    idea, spine_template = run_spine_template_flow(
        idea=idea,
        qa_text=qa_text,
        core_premise=core_premise,
        spine_template_generator=generators["SpineTemplateGenerator"],
    )
    world_bible = generate_world_bible(
        core_premise=core_premise,
        spine_template=spine_template,
        world_bible_question_generator=generators["WorldBibleQuestionGenerator"],
        world_bible_generator=generators["WorldBibleGenerator"],
    )
    return StoryFoundation(
        core_premise=core_premise,
        spine_template=spine_template,
        world_bible=world_bible,
    )


def _run_story_generation_phase(
    args: Namespace,
    parser: argparse.ArgumentParser,
    generators: dict[str, Any],
    foundation: StoryFoundation,
    image_artifacts: ImageArtifacts,
) -> tuple[Any, str, dict[int, str]]:
    """Run story generation, scene image generation, and similarity checks."""
    story_result, final_story_text = generate_story_text(
        args=args,
        parser=parser,
        generators=generators,
        foundation=foundation,
    )
    scene_image_paths = maybe_generate_scene_images(
        args=args,
        final_story_text=final_story_text,
        image_artifacts=image_artifacts,
    )
    run_similarity_check(args=args, final_story_text=final_story_text)
    return story_result, final_story_text, scene_image_paths


def _run_story_workflow(
    args: Namespace,
    parser: argparse.ArgumentParser,
    generators: dict[str, Any],
    idea: str,
) -> StoryRunArtifacts:
    """Run the interactive story pipeline and collect output artifacts."""
    foundation = _build_story_foundation(
        generators=generators,
        idea=idea,
    )
    image_artifacts = maybe_generate_character_assets(
        args=args,
        world_bible=foundation.world_bible,
    )
    Confirm.ask(
        "Press Enter to continue to Story generation...",
        default=True,
        show_default=False,
    )
    story_result, final_story_text, scene_image_paths = _run_story_generation_phase(
        args=args,
        parser=parser,
        generators=generators,
        foundation=foundation,
        image_artifacts=image_artifacts,
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


def _print_completion(output_filename: str) -> None:
    """Print final completion message for CLI run."""
    console.print(
        "\n[bold magenta]Story generation complete! "
        f"Results saved to {output_filename}[/bold magenta]"
    )


def main() -> None:
    """Run the interactive Story Writer CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_runtime(args)
    console.print("[bold magenta]Welcome to the AI DSPy Story Writer![/bold magenta]")

    idea = Prompt.ask("\n[bold yellow]What is your initial story idea/prompt?[/bold yellow]")
    generators = initialize_text_generators(
        use_optimized=args.use_optimized,
        optimized_manifest=args.optimized_manifest,
    )
    artifacts = _run_story_workflow(
        args=args,
        parser=parser,
        generators=generators,
        idea=idea,
    )

    output_filename = save_story_output(
        output_dir=args.output_dir,
        artifacts=artifacts,
    )
    _print_completion(output_filename)

if __name__ == "__main__":
    main()
