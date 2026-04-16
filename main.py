"""Interactive CLI entrypoint.

This is intentionally thin: it parses command-line flags, wires up a
:class:`CLIPrompter`, configures DSPy, and hands off to the headless pipeline
runner in :mod:`engine.pipeline`. All generation logic lives in
:mod:`engine.stages`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from rich.console import Console

from engine import PipelineOptions, StoryState, run_all
from engine.config import configure_dspy, initialize_text_generators
from engine.prompter_cli import CLIPrompter
from engine.writers import write_markdown
from logging_config import setup_logging


load_dotenv()

logger = logging.getLogger(__name__)


def _env_flag_true(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI DSPy Story Writer")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "openai/gpt-4o-mini"), help="The language model to use (e.g., openai/gpt-4o-mini, ollama_chat/llama3). Defaults to MODEL env var.")
    parser.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL"), help="The custom API base URL (e.g., http://localhost:11434 for Ollama). Defaults to LLM_URL env var.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY"), help="The API key for the model. Defaults to API_KEY env var.")
    parser.add_argument("--max-tokens", type=int, default=8000, help="The maximum number of tokens to use for the model. Defaults to 8000.")
    parser.add_argument("--enable-images", action="store_true", default=False, help="Enable image generation (requires Replicate API token).")
    parser.add_argument("--replicate-api-token", type=str, default=os.environ.get("REPLICATE_API_TOKEN"), help="Replicate API token. Defaults to REPLICATE_API_TOKEN env var.")
    parser.add_argument("--output-dir", type=str, default=".tmp", help="Directory to save generated content. Defaults to '.tmp'.")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable DSPy disk cache.")
    parser.add_argument("--memory-cache", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable DSPy in-memory cache.")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("DSPY_CACHE_DIR"), help="Override DSPy disk cache directory.")
    parser.add_argument(
        "--use-optimized",
        action=argparse.BooleanOptionalAction,
        default=_env_flag_true("DSPY_USE_OPTIMIZED", default=False),
        help="Enable/disable loading optimized text-pipeline module artifacts.",
    )
    parser.add_argument(
        "--optimized-manifest",
        type=str,
        default=os.environ.get("DSPY_OPTIMIZED_MANIFEST", ".tmp/dspy_optimized/text_pipeline_manifest.json"),
        help="Path to optimized text-pipeline manifest JSON.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.environ.get("LOG_FILE", ".tmp/test_debug.log"),
        help="Path to write detailed logs (default: LOG_FILE env var or .tmp/test_debug.log).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv full firehose.")
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
        help="Run a post-generation chapter expansion pass for richer detail (default: disabled).",
    )
    parser.add_argument(
        "--inpaint-ratio",
        type=float,
        default=1.35,
        help="Target chapter expansion ratio for inpainting (must be > 1.0, default: 1.35).",
    )
    return parser


def _options_from_args(args: argparse.Namespace) -> PipelineOptions:
    return PipelineOptions(
        enable_images=args.enable_images,
        replicate_api_token=args.replicate_api_token,
        inpaint_chapters=args.inpaint_chapters,
        inpaint_ratio=args.inpaint_ratio,
        check_similar=args.check_similar,
        similar_threshold=args.similar_threshold,
        use_optimized=args.use_optimized,
        optimized_manifest=args.optimized_manifest,
    )


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    setup_logging(verbosity=args.verbose, log_file=args.log_file)

    options = _options_from_args(args)
    try:
        options.validate()
    except ValueError as exc:
        # Surface the same CLI-friendly error the old code raised for bad
        # --inpaint-ratio / missing --replicate-api-token combinations.
        parser.error(str(exc))

    configure_dspy(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        cache=args.cache,
        memory_cache=args.memory_cache,
        cache_dir=args.cache_dir,
    )

    console = Console()
    console.print("[bold magenta]Welcome to the AI DSPy Story Writer![/bold magenta]")

    prompter = CLIPrompter(console=console)
    generators = initialize_text_generators(
        use_optimized=args.use_optimized,
        optimized_manifest=args.optimized_manifest,
    )

    state = StoryState()
    state = run_all(state, generators, prompter, options)

    output_path = os.path.join(args.output_dir, "story_output.md")
    write_markdown(state, output_path)
    console.print(
        f"\n[bold magenta]Story generation complete! "
        f"Results saved to {output_path}[/bold magenta]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
