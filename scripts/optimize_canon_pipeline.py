#!/usr/bin/env python3
"""Optimize the chapter drafter against the canon/continuity metric.

Wires the composite per-chapter metric (`bench.metric.make_chapter_metric`) into
`dspy.Evaluate` and a `BootstrapFewShot` optimizer so prompts/demos are tuned to
reduce the real continuity failures. The optimizer COMPILE step needs a
configured LM (an API key / endpoint); the trainset, program, metric, and
evaluator are all built without one, so the wiring is unit-testable in CI and
the live run is invoked explicitly by the user::

    python scripts/optimize_canon_pipeline.py --model openai/gpt-4o-mini \\
        --output .tmp/dspy_optimized/chapter_drafter.json

Without ``--model``/credentials the script builds everything and reports that no
LM is configured rather than attempting a (failing) compile.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

import canon as canon_mod
from bench.metric import make_chapter_metric
from core.foundation import act_hint_for_chapter
from core.types import WorldBible
from dspy_runtime import DSPyConfig, configure_dspy
from story_module import DraftChapter

logger = logging.getLogger(__name__)

_DRAFT_INPUTS = (
    "story_idea", "story_title", "story_spine", "act_hint",
    "world_bible", "chapter", "canon_directives",
)


class ChapterDrafter(dspy.Module):
    """Single-stage program: draft one chapter from its plan + canon directives."""

    def __init__(self) -> None:
        super().__init__()
        self.draft = dspy.ChainOfThought(DraftChapter)

    def forward(self, **kwargs):
        return self.draft(**kwargs)


def _stub_world_bible(canon: canon_mod.Canon) -> WorldBible:
    return WorldBible(
        story_title=canon.title or "Untitled",
        rules_of_the_world=["Power has a physical cost."],
        characters=["The protagonist — grey eyes, brown nape-braid, a right-cheek scar."],
        locations=["The Maw — a sealed pit below the city."],
        timeline=["Year 0: flight. Year 5: the end."],
    )


def build_trainset(canon: canon_mod.Canon, total_chapters: int = 15) -> list[dspy.Example]:
    """Build a small trainset of per-chapter drafting tasks seeded from the canon.

    Each example carries the DraftChapter inputs plus the ``chapter_number`` /
    ``chapter_title`` / ``canon`` the per-chapter metric needs. Chapters with a
    required artifact or a name-arc boundary are favoured so the optimizer is
    pushed on the constraints that actually fail.
    """
    world_bible = _stub_world_bible(canon)
    spine = "Once upon a time...\nUntil one day...\nUntil finally..."
    focus_chapters = sorted({4, 7, 11, 12, 17, *canon.required_artifacts})
    examples: list[dspy.Example] = []
    for number in focus_chapters:
        if number > total_chapters:
            continue
        hint = act_hint_for_chapter(number, total_chapters, spine)
        directives = canon_mod.render_chapter_slice(canon, number)
        ex = dspy.Example(
            story_idea="A weapon raised as a person learns its own name.",
            story_title=canon.title or "Untitled",
            story_spine=hint["spine_through_act"],
            act_hint=hint["label"],
            world_bible=world_bible,
            chapter=f"Chapter {number}\nA pivotal scene unfolds.",
            canon_directives=directives,
            chapter_number=number,
            chapter_title=f"Chapter {number}",
            canon=canon,
        ).with_inputs(*_DRAFT_INPUTS)
        examples.append(ex)
    return examples


def build_optimizer(canon: canon_mod.Canon, threshold: float = 1.0):
    """Construct the BootstrapFewShot optimizer bound to the per-chapter metric."""
    from dspy.teleprompt import BootstrapFewShot

    metric = make_chapter_metric(canon, threshold=threshold)
    return BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=0)


def build_evaluator(trainset, canon: canon_mod.Canon):
    """A dspy.Evaluate over the trainset using the (float) per-chapter metric."""
    return dspy.Evaluate(
        devset=trainset,
        metric=make_chapter_metric(canon),
        num_threads=1,
        display_progress=True,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canon", default=None, help="Canon config path (YAML/JSON).")
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path(".tmp/dspy_optimized/chapter_drafter.json"))
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv()
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    canon = canon_mod.load_canon(args.canon)
    trainset = build_trainset(canon)
    program = ChapterDrafter()
    logger.info("built program + trainset (%d examples)", len(trainset))

    if not args.model:
        logger.warning(
            "no --model/MODEL configured; built the optimizer wiring but skipping "
            "the live compile. Re-run with a model to optimize."
        )
        return 0

    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    configure_dspy(DSPyConfig(
        model_name=model_name, api_key=args.api_key,
        max_tokens=args.max_tokens, num_ctx=args.num_ctx,
    ))

    evaluator = build_evaluator(trainset, canon)
    logger.info("baseline score: %s", evaluator(program))

    optimizer = build_optimizer(canon, threshold=args.threshold)
    compiled = optimizer.compile(program, trainset=trainset)
    logger.info("optimized score: %s", evaluator(compiled))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(args.output))
    logger.info("saved optimized chapter drafter -> %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
