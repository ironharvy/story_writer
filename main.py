"""CLI entry point for the Story Writer pipeline."""

import os

from dotenv import load_dotenv

from cli import build_arg_parser
from dspy_runtime import DSPyConfig, configure_dspy, dspy_config_from_namespace
from logging_config import setup_logging
from models import GenerationParams
from output import initialize_artifact, save_story_output, write_generation_parameters
from pipeline import initialize_text_generators, run_story_workflow
from qa import run_quality_checks
import ui

load_dotenv()


def _build_generation_params(config: DSPyConfig) -> GenerationParams:
    """Build a GenerationParams snapshot from the resolved DSPy config."""
    return GenerationParams(
        model=config.model_name,
        max_tokens=config.max_tokens,
        cache=config.cache,
        memory_cache=config.memory_cache,
        cache_dir=config.cache_dir or "",
    )


def main() -> None:
    """Run the interactive Story Writer CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    setup_logging(verbosity=args.verbose, log_file=args.log_file)
    dspy_config = dspy_config_from_namespace(args)
    configure_dspy(dspy_config)

    output_file = os.path.join(args.output_dir, "story_output.md")
    initialize_artifact(output_file)
    write_generation_parameters(output_file, _build_generation_params(dspy_config))

    generators = initialize_text_generators(
        use_optimized=args.use_optimized,
        optimized_manifest=args.optimized_manifest,
    )

    ui.print_welcome()
    idea = ui.ask_idea()

    artifacts = run_story_workflow(
        args=args,
        parser=parser,
        generators=generators,
        idea=idea,
        output_file=output_file,
    )

    run_quality_checks(args=args, final_story_text=artifacts.final_story_text)

    output_filename = save_story_output(
        output_dir=args.output_dir,
        artifacts=artifacts,
    )
    ui.print_completion(output_filename)


if __name__ == "__main__":
    main()
