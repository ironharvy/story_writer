"""Score every story under a bench-run directory and build a comparison table.

Walks `<run_root>/<fixture_id>/<strategy_id>/story.md` (the layout
written by `bench/run.py`), runs `bench/criteria.score` on each, and
emits both a per-(fixture, strategy) JSON scorecard alongside the story
and a `<run_root>/results.md` summary table.

By default only the deterministic Tier-1 gates run, so scoring is fast and
needs no model. Pass `--llm-judge` to additionally run the Tier-2/Tier-3
LLM judge (`bench/judge.py`) against a local Ollama model.

Usage::

    python -m bench.score .tmp/bench/20260527T223000Z
    python -m bench.score .tmp/bench/latest --llm-judge --judge-model qwen3:latest
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bench.criteria import Scorecard, merge_judge, score

logger = logging.getLogger(__name__)

ROW = "| {strategy:<14} | {fails:>5} | {warns:>5} | {drift:>5} | {reuse:>5} | {rubric:>6} | {ship:^5} |"
HEADER = ROW.format(strategy="strategy", fails="fails", warns="warns",
                    drift="drift", reuse="reuse", rubric="rubric", ship="ship")
DIVIDER = "|" + "|".join(["-" * 16, "-" * 7, "-" * 7, "-" * 7, "-" * 7, "-" * 8, "-" * 7]) + "|"


def render_fixture_table(fixture_id: str, results: dict[str, Scorecard]) -> str:
    lines = [f"## Fixture: `{fixture_id}`", "", HEADER, DIVIDER]
    for strategy_id in sorted(results):
        s = results[strategy_id]
        lines.append(ROW.format(
            strategy=strategy_id,
            fails=len(s.fails),
            warns=len(s.warns),
            drift=s.budgets["name_drift"]["count"],
            reuse=s.budgets["phrase_reuse"]["count"],
            rubric=f"{s.rubric_average:.2f}" if s.rubric_average is not None else "-",
            ship="✓" if s.ship else "✗",
        ))
    lines.append("")
    return "\n".join(lines)


def walk_run_root(run_root: Path) -> dict[str, dict[str, Path]]:
    """Return {fixture_id: {strategy_id: story.md path}}."""
    by_fixture: dict[str, dict[str, Path]] = {}
    for story in run_root.glob("*/*/story.md"):
        fixture_id = story.parent.parent.name
        strategy_id = story.parent.name
        by_fixture.setdefault(fixture_id, {})[strategy_id] = story
    return by_fixture


def _configure_judge(model: str, provider: str) -> None:
    """Point the global DSPy LM at the judge model (imported lazily)."""
    from dspy_runtime import DSPyConfig, configure_dspy

    model_name = model if "/" in model else f"{provider}/{model}"
    logger.info("configuring LLM judge: %s", model_name)
    configure_dspy(DSPyConfig(model_name=model_name))


def _idea_for_fixtures() -> dict[str, str]:
    """Map fixture id -> original idea, for the T2.4 premise check."""
    try:
        from bench.run import load_fixtures

        return {f.id: f.idea for f in load_fixtures()}
    except Exception as exc:  # fixtures are optional; premise check just skips
        logger.warning("could not load fixtures for premise check: %s", exc)
        return {}


def _score_pair(story_path: Path, idea: str, judge_on: bool) -> Scorecard:
    """Tier-1 scorecard for one story, optionally merged with the LLM judge."""
    card = score(story_path)
    if judge_on:
        from bench.judge import judge_story

        report = judge_story(story_path, idea)
        card = merge_judge(card, report.findings, report.rubric)
    return card


def _preamble(judge_on: bool) -> str:
    judged = (
        " (with --llm-judge: + Tier-2 POV / protagonist / contradiction / premise)"
        if judge_on else ""
    )
    return (
        "Each row is one (fixture, strategy) pair. `fails` are hard gates"
        f"{judged}; `warns` and the per-budget columns are soft limits; "
        "`rubric` is the Tier-3 average (1-5, blank unless judged) per "
        "bench/eval-spec.md.\n"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for scoring a bench run directory (Tier-1, optionally LLM judge)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path,
                        help="Path to a bench run directory (e.g. .tmp/bench/20260527T223000Z)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the summary (default: <run_root>/results.md)")
    parser.add_argument("--llm-judge", action="store_true",
                        help="Also run the Tier-2/Tier-3 LLM judge (needs a model; default off).")
    parser.add_argument("--judge-model", default="qwen3:latest",
                        help="Judge model id (default: qwen3:latest).")
    parser.add_argument("--judge-provider", default="ollama",
                        help="Provider prefix when --judge-model has no '/' (default: ollama).")
    return parser


def main() -> int:
    """Score every story under the run root and write a results.md table."""
    args = build_arg_parser().parse_args()
    if not args.run_root.is_dir():
        print(f"run_root not a directory: {args.run_root}", file=sys.stderr)
        return 2
    by_fixture = walk_run_root(args.run_root)
    if not by_fixture:
        print(f"no story.md files found under {args.run_root}", file=sys.stderr)
        return 1

    if args.llm_judge:
        _configure_judge(args.judge_model, args.judge_provider)
    idea_map = _idea_for_fixtures() if args.llm_judge else {}

    sections = [f"# Bench results — `{args.run_root.name}`\n", _preamble(args.llm_judge)]
    for fixture_id in sorted(by_fixture):
        results: dict[str, Scorecard] = {}
        for strategy_id, story_path in by_fixture[fixture_id].items():
            try:
                sc = _score_pair(story_path, idea_map.get(fixture_id, ""), args.llm_judge)
            except Exception as exc:
                print(f"score {fixture_id}/{strategy_id} failed: {exc}", file=sys.stderr)
                continue
            results[strategy_id] = sc
            (story_path.parent / "scorecard.json").write_text(sc.to_json(), encoding="utf-8")
        sections.append(render_fixture_table(fixture_id, results))

    summary = "\n".join(sections)
    output = args.output or (args.run_root / "results.md")
    output.write_text(summary, encoding="utf-8")
    print(summary)
    print(f"\nwrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
