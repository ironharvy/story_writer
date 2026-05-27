#!/usr/bin/env python3
"""Run the LLM-backed point-of-view consistency check on story markdown.

Classifies each chapter's dominant narrative POV (first / third person, or
"mixed" when a chapter shifts within itself) and flags chapters that disagree
with the dominant POV of the rest of the book. A story that is uniformly first-
or third-person passes.

Mirrors `scripts/run_qa.py`'s output and exit codes (2 if any FAIL), and
`scripts/lint_story.py`'s DSPy configuration flags.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Add repo root so top-level modules import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dspy_runtime import DSPyConfig, configure_dspy  # noqa: E402
from logging_config import setup_logging  # noqa: E402
from pov_check import run as run_pov_check  # noqa: E402


logger = logging.getLogger(__name__)

SEVERITY_RANK = {"fail": 0, "warn": 1, "info": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="story markdown file(s)")

    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("DSPY_CACHE_DIR", ".cache/dspy"),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=".tmp/check_pov.log")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    setup_logging(verbosity=args.verbose, log_level=args.log_level, log_file=args.log_file)

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

    exit_code = 0
    for path in args.paths:
        if not path.is_file():
            print(f"!! not a file: {path}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"\n=== {path} ===")
        findings = run_pov_check(path)
        findings.sort(key=lambda f: (SEVERITY_RANK.get(f.severity, 9), f.check))
        for f in findings:
            tag = {"fail": "FAIL", "warn": "WARN", "info": "info"}.get(f.severity, f.severity)
            print(f"[{tag}] {f.check}: {f.message}")
            if f.severity == "fail":
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
