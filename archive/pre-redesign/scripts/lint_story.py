#!/usr/bin/env python3
"""Lint a finished story markdown and rewrite token-level errors in chapter prose.

Reads the world bible's ``### Characters`` section together with the chapter
prose, asks the configured DSPy LM for verbatim find/replace pairs, and applies
them to the chapter prose region (the body under ``## Final Story``). The plan,
spine, and world bible are never modified.

By default the input file is rewritten in place; a ``.bak`` copy of the
original is left next to it and a ``.replacements.json`` sidecar records what
was applied. Use ``--dry-run`` to emit only the sidecar, or ``--out PATH`` to
write the linted output elsewhere (the sidecar follows the output path).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add repo root so we can import top-level modules when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dspy_runtime import DSPyConfig, configure_dspy  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from story_linter import (  # noqa: E402
    apply,
    lint,
    replacements_to_json,
    validate,
    _extract_inputs,
)


logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("story", type=Path, help="Path to the finished story markdown.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the linted output here instead of rewriting in place.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only write the .replacements.json sidecar; leave the story untouched.",
    )

    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--memory-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("DSPY_CACHE_DIR", ".cache/dspy"),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=".tmp/lint_story.log")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args(argv)


def _sidecar_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".replacements.json")


def _backup_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".bak")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)

    setup_logging(
        verbosity=args.verbose,
        log_level=args.log_level,
        log_file=args.log_file,
    )

    if not args.story.is_file():
        print(f"!! not a file: {args.story}", file=sys.stderr)
        return 1

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

    story_md = args.story.read_text(encoding="utf-8")

    proposed = lint(story_md)
    _, joined_prose = _extract_inputs(story_md)
    validated = validate(proposed, joined_prose)
    rewritten, counts = apply(story_md, validated)

    out_path = args.out if args.out is not None else args.story
    sidecar = _sidecar_path(out_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(replacements_to_json(validated, counts), encoding="utf-8")

    if args.dry_run:
        print(f"dry-run: wrote {len(validated)} replacement(s) to {sidecar}; story not modified")
        _print_summary(counts)
        return 0

    if args.out is None:
        backup = _backup_path(args.story)
        if not backup.exists():
            shutil.copy2(args.story, backup)
    out_path.write_text(rewritten, encoding="utf-8")

    print(f"wrote {out_path} (+ {sidecar})")
    _print_summary(counts)
    return 0


def _print_summary(counts) -> None:
    if not counts:
        print("  no replacements proposed")
        return
    total = sum(n for _, n in counts)
    print(f"  {len(counts)} replacement(s) proposed, {total} total substitution(s)")
    for r, n in counts:
        print(f"  - x{n}  {r.find!r} -> {r.replace!r}  ({r.reason})")


if __name__ == "__main__":
    raise SystemExit(main())
