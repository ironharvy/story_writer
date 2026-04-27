"""CLI argument parsing for the Story Writer pipeline."""

import argparse
import os


def _env_flag_true(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


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
        default=os.environ.get("DSPY_CACHE_DIR", ".cache/dspy"),
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
