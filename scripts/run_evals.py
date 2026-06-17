"""Run the LLM-judge eval gates over a finished story markdown artifact.

Complements ``scripts/run_qa.py`` (deterministic heuristics) and
``scripts/check_pov.py`` (POV) with the reasoning-based gates in ``evals/``:
bible self-consistency, per-chapter bible consistency + reading-comprehension,
cross-chapter continuity, and premise fidelity.

Usage:
    python scripts/run_evals.py .tmp/runs/mages.md \
        --model qwen3.6:27b --provider ollama --num-ctx 16384 \
        [--idea "..."] [--per-chapter] [--report out.md] [--think]

By default it runs the cheap manuscript-level gates (internal consistency,
continuity, premise). Add ``--per-chapter`` to also run the bible-consistency
and comprehension gates on every chapter (slower).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# allow running as a loose script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import dspy  # noqa: E402

from dspy_runtime import DSPyConfig, configure_dspy, thinking_lm  # noqa: E402
from evals import (  # noqa: E402
    bible_consistency,
    bible_internal,
    comprehension,
    continuity,
    premise_fidelity,
)
from evals.loader import load_story  # noqa: E402
from evals.types import GateReport  # noqa: E402

logger = logging.getLogger(__name__)


def _render(report: GateReport) -> str:
    if not report.findings:
        return "  _No issues found._"
    lines = []
    for f in report.findings:
        where = f" ({f.locus})" if f.locus else ""
        fix = f"\n      fix: {f.fix}" if f.fix else ""
        lines.append(f"  [{f.severity.upper()}] {f.check}{where}: {f.message}{fix}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("story", help="Path to a generated story .md")
    ap.add_argument("--model", default="qwen3.6:27b")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--api-key")
    ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--idea", default="", help="Original story idea (defaults to the premise)")
    ap.add_argument("--per-chapter", action="store_true", help="Also run per-chapter gates")
    ap.add_argument("--think", action="store_true", help="Enable model reasoning for the gates")
    ap.add_argument("--report", help="Write the report to this markdown file too")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    configure_dspy(DSPyConfig(
        model_name=model_name, api_key=args.api_key,
        max_tokens=args.max_tokens, num_ctx=args.num_ctx, think=args.think,
    ))

    story = load_story(args.story)
    idea = args.idea or story.premise
    out: list[str] = [f"# Eval report: {story.title}", ""]

    def section(title: str, report: GateReport):
        out.append(f"## {title}")
        out.append(_render(report))
        out.append("")
        print(f"\n=== {title} ===")
        print(_render(report))

    cm = dspy.context(lm=thinking_lm()) if args.think else _nullcontext()
    with cm:
        internal_report, clarifications = bible_internal.check(story.world_bible)
        out.append("## Bible self-consistency")
        out.append(_render(internal_report))
        if clarifications:
            out.append("\n  Canon clarifications:")
            out.extend(f"  - {c}" for c in clarifications)
        out.append("")
        print("\n=== Bible self-consistency ===")
        print(_render(internal_report))

        section("Cross-chapter continuity", continuity.check(story.chapters))
        section("Premise fidelity", premise_fidelity.check(idea, story.premise, story.spine,
                                                            _summary(story.chapters)))

        if args.per_chapter:
            for title, prose in story.chapters.items():
                beats = story.plan.get(title, "")
                section(f"Bible consistency — {title}",
                        bible_consistency.check(story.world_bible, title, prose))
                section(f"Comprehension — {title}",
                        comprehension.check(title, beats, story.world_bible, prose))

    if args.report:
        Path(args.report).write_text("\n".join(out), encoding="utf-8")
        print(f"\nReport written to {args.report}")
    return 0


def _summary(chapters: dict[str, str]) -> str:
    """Cheap stand-in summary: first ~80 words of each chapter, for premise judging
    when no rolling summary is available from the artifact."""
    parts = []
    for title, prose in chapters.items():
        words = prose.split()
        parts.append(f"### {title}\n{' '.join(words[:140])}")
    return "\n\n".join(parts)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *_a):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
