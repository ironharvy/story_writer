"""Multi-fixture × multi-generator benchmark runner.

Walks `bench/fixtures/*.json`, runs each registered generator (default:
all `promoted`) on each fixture against the configured DSPy backend, and
writes one story markdown per (fixture, strategy) under
`.tmp/bench/<run_id>/<fixture_id>/<strategy>/story.md`. The scorecards
get produced by `bench/score.py` afterwards.

This skeleton does NOT mock the LLM. Run it locally against a real
Ollama (or any DSPy-supported provider) with the same flags as
`main.py`. Each (fixture, strategy) pair is one full pipeline run, so
expect minutes-to-hours of wall-clock depending on the model and the
fixture's chapter count.

Usage::

    python -m bench.run --model qwen3:latest --provider ollama
    python -m bench.run --generators baseline,world_state --fixtures 01_thriller
    python -m bench.run --run-id smoke-1 --output-root .tmp/bench
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import generators  # populates REGISTRY via auto-discovery
from core.artifact import initialize_artifact, update_artifact
from core.foundation import (
    build_foundation,
    run_generate_chapters_plan,
    sanity_check,
)
from core.types import DraftingInput
from dspy_runtime import DSPyConfig, configure_dspy

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass(frozen=True)
class Fixture:
    id: str
    title: str
    idea: str
    number_of_chapters: int
    niche: str

    @classmethod
    def load(cls, path: Path) -> "Fixture":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def load_fixtures(ids: list[str] | None = None) -> list[Fixture]:
    """Load every fixture under FIXTURES_DIR, or just the named ones."""
    all_fixtures = [Fixture.load(p) for p in sorted(FIXTURES_DIR.glob("*.json"))]
    if ids is None:
        return all_fixtures
    by_id = {f.id: f for f in all_fixtures}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"unknown fixture id(s): {missing}; known: {sorted(by_id)}")
    return [by_id[i] for i in ids]


def select_generators(ids: list[str] | None) -> list:
    """Pick which generators to run. Default: every promoted one."""
    if ids is None:
        return generators.promoted()
    entries = []
    for i in ids:
        entries.append(generators.get(i))
    return entries


def run_one(fixture: Fixture, entry, output_root: Path) -> Path:
    """Run one (fixture, generator) pair end-to-end; returns the story path."""
    out_dir = output_root / fixture.id / entry.id
    out_dir.mkdir(parents=True, exist_ok=True)
    story_path = out_dir / "story.md"
    initialize_artifact(str(story_path))
    update_artifact(
        str(story_path),
        "Bench Run",
        f"- fixture: `{fixture.id}` ({fixture.niche})\n"
        f"- strategy: `{entry.id}` [{entry.status}]\n"
        f"- chapters: {fixture.number_of_chapters}",
    )

    updated_idea, story_title, spine, world_bible = build_foundation(
        fixture.idea, fixture.title, str(story_path),
    )
    if not sanity_check(updated_idea, story_title, spine, world_bible):
        logger.warning(
            "fixture=%s strategy=%s: foundation failed sanity_check; skipping draft",
            fixture.id, entry.id,
        )
        return story_path

    chapters_plan = run_generate_chapters_plan(
        updated_idea, story_title, spine, world_bible, fixture.number_of_chapters,
    )

    update_artifact(str(story_path), "Chapters Plan", "", level=3)
    for i, ch in enumerate(chapters_plan, 1):
        update_artifact(
            str(story_path),
            f"Chapter {i}: {ch.chapter_title}",
            ch.chapter_beats,
            level=4,
        )

    entry.cls().draft(
        DraftingInput(
            idea=updated_idea,
            title=story_title,
            spine=spine,
            world_bible=world_bible,
            chapters_plan=chapters_plan,
            output_file=str(story_path),
            number_of_chapters=fixture.number_of_chapters,
        )
    )
    return story_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated fixture ids (default: all under bench/fixtures/).",
    )
    parser.add_argument(
        "--generators",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated generator ids (default: every promoted one).",
    )
    parser.add_argument(
        "--run-id",
        default=_dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        help="Subdirectory under --output-root (default: UTC timestamp).",
    )
    parser.add_argument("--output-root", default=".tmp/bench")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument(
        "--cache", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--memory-cache", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--cache-dir", default=".cache/dspy")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fixtures = load_fixtures(args.fixtures)
    entries = select_generators(args.generators)
    output_root = Path(args.output_root) / args.run_id
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info(
        "bench start | run_id=%s | fixtures=%s | generators=%s | model=%s",
        args.run_id,
        [f.id for f in fixtures],
        [e.id for e in entries],
        model_name,
    )
    for fixture in fixtures:
        for entry in entries:
            logger.info("=== %s × %s ===", fixture.id, entry.id)
            try:
                path = run_one(fixture, entry, output_root)
                logger.info("wrote %s", path)
            except Exception:
                logger.exception("fixture=%s strategy=%s failed", fixture.id, entry.id)
                continue

    logger.info("bench done | run_id=%s | scorecards: python -m bench.score %s",
                args.run_id, output_root)


if __name__ == "__main__":
    sys.exit(main())
