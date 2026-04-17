#!/usr/bin/env python3
"""Standalone CLI that adds (or refreshes) images for an existing story run.

Consumes a ``story_output.json`` artifact produced by ``main.py`` and runs the
shared image pipeline against it.  Useful when you generated a story without
``--enable-images`` and want to add portraits and scene illustrations later
without regenerating the text.

Examples:
    # Generate both portraits and scenes, overwriting the JSON in place.
    python scripts/generate_images.py --story-json .tmp/story_output.json

    # Portraits only, writing a fresh JSON and refreshed markdown elsewhere.
    python scripts/generate_images.py \\
        --story-json .tmp/story_output.json \\
        --skip-scenes \\
        --output-json .tmp/story_output.with_images.json \\
        --output-markdown .tmp/story_output.with_images.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the project root importable when running as ``scripts/...``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_gen import ImageGenerator  # noqa: E402
from image_pipeline import generate_images_for_story  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from main import configure_dspy  # noqa: E402
from story_artifacts import load_artifacts, save_artifacts, save_markdown  # noqa: E402


logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate images for an existing story output.",
    )
    parser.add_argument(
        "--story-json",
        type=str,
        required=True,
        help="Path to the story artifacts JSON (produced by main.py).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Where to write the updated artifacts JSON (default: overwrite input).",
    )
    parser.add_argument(
        "--output-markdown",
        type=str,
        default=None,
        help="If set, (re)render and write the markdown view to this path.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Directory to save generated images. Defaults to 'images'.",
    )
    parser.add_argument(
        "--replicate-api-token",
        type=str,
        default=os.environ.get("REPLICATE_API_TOKEN"),
        help="Replicate API token. Defaults to REPLICATE_API_TOKEN env var.",
    )
    parser.add_argument(
        "--skip-portraits",
        action="store_true",
        help="Do not generate character portraits.",
    )
    parser.add_argument(
        "--skip-scenes",
        action="store_true",
        help="Do not generate per-chapter scene illustrations.",
    )
    parser.add_argument(
        "--regenerate-portraits",
        action="store_true",
        help="Regenerate portraits even if they already exist in the JSON.",
    )
    parser.add_argument(
        "--regenerate-scenes",
        action="store_true",
        help="Regenerate scene images even if they already exist in the JSON.",
    )

    # LLM config (needed for CharacterVisualDescriber / SceneImagePromptGenerator).
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "openai/gpt-4o-mini"),
        help="The language model used for prompt derivation.",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=os.environ.get("LLM_URL"),
        help="Custom API base URL.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY"),
        help="API key for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Max tokens for prompt-derivation LLM calls.",
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
        default=os.environ.get("DSPY_CACHE_DIR"),
        help="Override DSPy disk cache directory.",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=os.environ.get("LOG_FILE", ".tmp/generate_images.log"),
        help="Path for detailed logs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv full firehose.",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = _build_parser()
    args = parser.parse_args()

    setup_logging(verbosity=args.verbose, log_file=args.log_file)

    if not args.replicate_api_token:
        logger.error(
            "Missing Replicate API token. Set REPLICATE_API_TOKEN "
            "or pass --replicate-api-token.",
        )
        return 1

    if args.skip_portraits and args.skip_scenes:
        logger.error("Both --skip-portraits and --skip-scenes given; nothing to do.")
        return 1

    configure_dspy(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        cache=args.cache,
        memory_cache=args.memory_cache,
        cache_dir=args.cache_dir,
    )

    artifacts = load_artifacts(args.story_json)
    logger.info(
        "Loaded artifacts: %d character visuals, %d portraits, %d scenes.",
        len(artifacts.character_visuals),
        len(artifacts.character_portrait_paths),
        len(artifacts.scene_image_paths),
    )

    if args.regenerate_portraits:
        artifacts.character_portrait_paths = {}
    if args.regenerate_scenes:
        artifacts.scene_image_paths = {}

    image_backend = ImageGenerator(
        api_token=args.replicate_api_token,
        output_dir=args.images_dir,
    )

    generate_images_for_story(
        artifacts,
        image_backend,
        skip_portraits=args.skip_portraits,
        skip_scenes=args.skip_scenes,
    )

    output_json = args.output_json or args.story_json
    save_artifacts(output_json, artifacts)

    if args.output_markdown:
        save_markdown(args.output_markdown, artifacts)

    logger.info(
        "Done. Portraits: %d, scenes: %d.",
        len(artifacts.character_portrait_paths),
        len(artifacts.scene_image_paths),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
