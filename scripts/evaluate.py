#!/usr/bin/env python3
"""Score a finished story against the executable Definition of Done.

Runs the unified scorecard (``bench/evaluate.py``): deterministic Tier-1 gates
always, plus the LLM-backed Tier-2 gates (POV consistency + prose lint) when
``--with-llm`` is passed. Writes a ``<story>.scorecard.json`` sidecar per file
and prints a human summary. Exit code is 2 if any gate FAILs.

Without ``--with-llm`` this needs no model and reports ``tier1_clean``; the
full ``ship`` verdict requires the Tier-2 gates, so run ``--with-llm`` (with a
configured Ollama/DSPy backend) for a shippable/not-shippable decision.

Mirrors ``scripts/check_pov.py``'s DSPy configuration flags.
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

from bench.evaluate import FullScorecard, evaluate_path  # noqa: E402
from logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

_STATUS_TAG = {
    "pass": "PASS",
    "fail": "FAIL",
    "warn": "WARN",
    "skipped": "skip",
    "manual": "MANUAL",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="story markdown file(s)")
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run the Tier-2 LLM gates (POV consistency + prose lint).",
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Don't write the <story>.scorecard.json sidecar.",
    )

    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("DSPY_CACHE_DIR", ".cache/dspy"),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=".tmp/evaluate.log")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args(argv)


def _configure_llm(args: argparse.Namespace) -> None:
    """Configure the DSPy backend (only needed for the Tier-2 gates)."""
    from dspy_runtime import DSPyConfig, configure_dspy

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


def _render(path: Path, card: FullScorecard) -> str:
    """Render a human-readable summary of one scorecard."""
    lines = [f"\n=== {path} ==="]
    verdict = "SHIP" if card.ship else ("tier-1 clean" if card.tier1_clean else "DO NOT SHIP")
    suffix = "" if card.complete else "  (Tier-2 not run — provisional)"
    lines.append(f"verdict: {verdict}{suffix}")
    for gate in card.gates:
        tag = _STATUS_TAG.get(gate.status, gate.status)
        detail = f": {'; '.join(gate.messages)}" if gate.messages else ""
        lines.append(f"  [{tag}] T{gate.tier} {gate.gate}{detail}")
    for budget, info in card.budgets.items():
        flag = " OVER" if info["over"] else ""
        lines.append(f"  budget {budget}: {info['count']}/{info['budget']}{flag}")
    for item in card.manual_checks:
        lines.append(f"  [MANUAL] {item}")
    return "\n".join(lines)


def _sidecar_path(story_path: Path) -> Path:
    return story_path.with_suffix(story_path.suffix + ".scorecard.json")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    setup_logging(verbosity=args.verbose, log_level=args.log_level, log_file=args.log_file)
    if args.with_llm:
        _configure_llm(args)

    exit_code = 0
    for path in args.paths:
        if not path.is_file():
            logger.error("not a file: %s", path)
            exit_code = max(exit_code, 1)
            continue
        card = evaluate_path(path, run_llm=args.with_llm)
        print(_render(path, card))
        if not args.no_sidecar:
            _sidecar_path(path).write_text(card.to_json(), encoding="utf-8")
        if card.fails:
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
