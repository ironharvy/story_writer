import argparse
import logging
import os

from dspy_runtime import DSPyConfig, configure_dspy
from logging_config import setup_logging
from artifact import update_artifact, initialize_artifact
from pipeline import write


logger = logging.getLogger(__name__)


def format_generation_parameters(args: argparse.Namespace) -> str:
    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    return "\n".join(
        [
            f"- model: `{model_name}`",
            f"- provider: `{args.provider}`",
            f"- max_tokens: `{args.max_tokens}`",
            f"- cache: `{args.cache}`",
            f"- memory_cache: `{args.memory_cache}`",
            f"- cache_dir: `{args.cache_dir}`",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="The maximum number of tokens to use for the model. Defaults to 4096.",
    )
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
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=".tmp/mymain.log")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv full firehose.",
    )
    parser.add_argument("--idea", type=str, required=False, help="The initial story idea/prompt")
    parser.add_argument("--title", type=str, required=False, help="The title of the story")
    parser.add_argument("--output-file", default=".tmp/story.md")
    return parser.parse_args()


def configure_logging(args: argparse.Namespace) -> None:
    setup_logging(
        verbosity=args.verbose,
        log_level=args.log_level,
        log_file=args.log_file,
    )


def configure_runtime(args: argparse.Namespace) -> None:
    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    configure_dspy(
        DSPyConfig(
            model_name=model_name,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            cache=args.cache,
            memory_cache=args.memory_cache,
            cache_dir=args.cache_dir,
        )
    )


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args)
    configure_runtime(args)

    initialize_artifact(args.output_file)
    update_artifact(args.output_file, "Generation Parameters", format_generation_parameters(args))
    update_artifact(args.output_file, "Idea", args.idea)
    
    logger.info("Starting story writer")
    write(args.idea, args.title, args.output_file)
