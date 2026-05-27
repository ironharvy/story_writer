"""Unified CLI entry point — selects a registered generator by `--strategy`.

Replaces the three sibling entry points (``mymain.py`` / ``mymain_ws.py`` /
``mymain_module.py``) and the matching ``pipeline*.py`` orchestrators with a
single flow: parse args → setup runtime → run foundation → run chapters_plan
→ dispatch to the chosen generator's `draft()` method. The generator owns
its drafting loop (and the incremental ``update_artifact`` writes within it);
everything outside of that is the orchestrator.

Example:

    python main.py --idea "..." --title "..." --strategy world_state -vv

Run ``python main.py --list-strategies`` to see what's registered.
"""
from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

import generators  # auto-discovers and registers every variant in generators/
from core.artifact import initialize_artifact, update_artifact
from core.foundation import (
    build_foundation,
    run_generate_chapters_plan,
    sanity_check,
)
from core.types import DraftingInput
from dspy_runtime import DSPyConfig, configure_dspy
from logging_config import setup_logging

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a multi-chapter story via a registered drafting strategy.",
    )
    parser.add_argument(
        "--strategy",
        default="baseline",
        help="Generator id (see --list-strategies for available ones).",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List registered generators and exit.",
    )
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=16384,
        help="Ollama context window size (num_ctx).",
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
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=".tmp/main.log")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--idea", type=str)
    parser.add_argument("--title", type=str)
    parser.add_argument("--number-of-chapters", type=int, default=7)
    parser.add_argument("--output-file", default=".tmp/story.md")
    return parser


def format_generation_parameters(args: argparse.Namespace) -> str:
    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    return "\n".join(
        [
            f"- strategy: `{args.strategy}`",
            f"- model: `{model_name}`",
            f"- provider: `{args.provider}`",
            f"- max_tokens: `{args.max_tokens}`",
            f"- num_ctx: `{args.num_ctx}`",
            f"- cache: `{args.cache}`",
            f"- memory_cache: `{args.memory_cache}`",
            f"- cache_dir: `{args.cache_dir}`",
        ]
    )


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
            num_ctx=args.num_ctx,
            cache=args.cache,
            memory_cache=args.memory_cache,
            cache_dir=args.cache_dir,
        )
    )


def print_strategies() -> None:
    width = max((len(e.id) for e in generators.REGISTRY.values()), default=0)
    for entry in generators.REGISTRY.values():
        print(f"  {entry.id:<{width}}  [{entry.status:<12}]  {entry.description}")


def _record_chapters_plan(output_file: str, chapters_plan) -> None:
    update_artifact(output_file, "Chapters Plan", "", level=3)
    for i, ch in enumerate(chapters_plan, 1):
        update_artifact(
            output_file, f"Chapter {i}: {ch.chapter_title}", ch.chapter_beats, level=4,
        )


def run_pipeline(args: argparse.Namespace) -> None:
    """Orchestrate foundation → chapters_plan → generator dispatch."""
    initialize_artifact(args.output_file)
    update_artifact(
        args.output_file, "Generation Parameters", format_generation_parameters(args),
    )

    updated_idea, story_title, spine, world_bible = build_foundation(
        args.idea, args.title, args.output_file,
    )
    if not sanity_check(updated_idea, story_title, spine, world_bible):
        logger.error("Idea, spine, and world bible are not consistent")
        return

    chapters_plan = run_generate_chapters_plan(
        updated_idea, story_title, spine, world_bible, args.number_of_chapters,
    )
    _record_chapters_plan(args.output_file, chapters_plan)

    entry = generators.get(args.strategy)
    if entry.status == "deprecated":
        logger.warning(
            "strategy %r is deprecated and will be removed; pick another from --list-strategies",
            args.strategy,
        )

    output = entry.cls().draft(
        DraftingInput(
            idea=updated_idea,
            title=story_title,
            spine=spine,
            world_bible=world_bible,
            chapters_plan=chapters_plan,
            output_file=args.output_file,
            number_of_chapters=args.number_of_chapters,
        )
    )
    logger.info(
        "drafted %d chapters via strategy=%s; continuity_artifact=%s",
        len(output.chapters),
        args.strategy,
        type(output.continuity_artifact).__name__,
    )


def main() -> None:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.list_strategies:
        print_strategies()
        return

    if args.strategy not in generators.REGISTRY:
        parser.error(
            f"unknown --strategy {args.strategy!r}; "
            f"available: {sorted(generators.REGISTRY)}",
        )

    configure_logging(args)
    configure_runtime(args)
    logger.info(
        "starting story writer | strategy=%s | chapters=%d",
        args.strategy, args.number_of_chapters,
    )
    run_pipeline(args)


if __name__ == "__main__":
    main()
