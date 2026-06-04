"""Score every story under a bench-run directory and build a comparison table.

Walks `<run_root>/<fixture_id>/<strategy_id>/story.md` (the layout
written by `bench/run.py`), runs `bench/criteria.score` on each, and
emits both a per-(fixture, strategy) JSON scorecard alongside the story
and a `<run_root>/results.md` summary table.

Usage::

    python -m bench.score .tmp/bench/20260527T223000Z
    python -m bench.score .tmp/bench/latest --output results.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.criteria import Scorecard, score

ROW = "| {strategy:<14} | {fails:>5} | {warns:>5} | {drift:>5} | {reuse:>5} | {ship:^5} |"
HEADER = ROW.format(strategy="strategy", fails="fails", warns="warns",
                    drift="drift", reuse="reuse", ship="ship")
DIVIDER = "|" + "|".join(["-" * 16, "-" * 7, "-" * 7, "-" * 7, "-" * 7, "-" * 7]) + "|"


def score_one(story_path: Path) -> Scorecard:
    return score(story_path)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path,
                        help="Path to a bench run directory (e.g. .tmp/bench/20260527T223000Z)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write the summary (default: <run_root>/results.md)")
    args = parser.parse_args()

    if not args.run_root.is_dir():
        print(f"run_root not a directory: {args.run_root}", file=sys.stderr)
        return 2

    by_fixture = walk_run_root(args.run_root)
    if not by_fixture:
        print(f"no story.md files found under {args.run_root}", file=sys.stderr)
        return 1

    sections = []
    sections.append(f"# Bench results — `{args.run_root.name}`\n")
    sections.append(
        "Each row is one (fixture, strategy) pair. `fails` is Tier-1 hard "
        "gates (T1.1 chapter floor, T1.2 character presence); `warns` and "
        "the per-budget columns are soft limits per bench/eval-spec.md.\n"
    )

    for fixture_id in sorted(by_fixture):
        results: dict[str, Scorecard] = {}
        for strategy_id, story_path in by_fixture[fixture_id].items():
            try:
                sc = score_one(story_path)
            except Exception as exc:
                print(f"score {fixture_id}/{strategy_id} failed: {exc}", file=sys.stderr)
                continue
            results[strategy_id] = sc
            # Write per-pair scorecard.json next to the story.
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
