"""Headless story-generation driver.

Reads a JSON file of pre-answered inputs and runs the full pipeline without
touching stdin or rendering rich output. This is a thin stand-in for the
future FastAPI worker — if this script works end-to-end for a given input,
wrapping it in a job queue is mechanical.

Input JSON schema (all fields required unless noted):

    {
      "idea": "A young wizard defends a floating city.",
      "answer_batches": [
        [null, "override answer 2"],   // first answer_questions call (ideation)
        [null]                          // second call (world bible)
      ],
      "premise_decisions": [
        {"accept": true}
      ],
      "options": {                      // optional; defaults match CLI
        "enable_images": false,
        "replicate_api_token": null,
        "inpaint_chapters": false,
        "inpaint_ratio": 1.35,
        "check_similar": true,
        "similar_threshold": 0.65,
        "use_optimized": false,
        "optimized_manifest": null
      }
    }

Usage:
    python scripts/run_headless.py --input script.json --output-dir .tmp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Make the repo root importable when invoked as ``python scripts/run_headless.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from engine import PipelineOptions, StoryState, run_all
from engine.config import configure_dspy, initialize_text_generators
from engine.prompter_scripted import PremiseDecision, ScriptedPrompter
from engine.writers import write_markdown
from logging_config import setup_logging


logger = logging.getLogger(__name__)


def _load_script(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_prompter(script: dict) -> ScriptedPrompter:
    decisions = [
        PremiseDecision(
            accept=bool(d.get("accept", False)),
            refinement_details=str(d.get("refinement_details", "")),
        )
        for d in script.get("premise_decisions", [])
    ]
    return ScriptedPrompter(
        idea=script["idea"],
        answer_batches=script.get("answer_batches", []),
        premise_decisions=decisions,
        notify_sink=lambda level, msg: logger.info("[%s] %s", level, msg),
    )


def _build_options(script: dict) -> PipelineOptions:
    raw = script.get("options", {}) or {}
    return PipelineOptions(
        enable_images=bool(raw.get("enable_images", False)),
        replicate_api_token=raw.get("replicate_api_token")
        or os.environ.get("REPLICATE_API_TOKEN"),
        inpaint_chapters=bool(raw.get("inpaint_chapters", False)),
        inpaint_ratio=float(raw.get("inpaint_ratio", 1.35)),
        check_similar=bool(raw.get("check_similar", True)),
        similar_threshold=float(raw.get("similar_threshold", 0.65)),
        use_optimized=bool(raw.get("use_optimized", False)),
        optimized_manifest=raw.get("optimized_manifest"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to script JSON.")
    parser.add_argument(
        "--output-dir", default=".tmp", help="Directory for markdown + state output."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "openai/gpt-4o-mini"),
        help="LLM model identifier.",
    )
    parser.add_argument("--llm-url", default=os.environ.get("LLM_URL"))
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    load_dotenv()
    setup_logging(verbosity=args.verbose, log_file=None)

    script = _load_script(args.input)
    prompter = _build_prompter(script)
    options = _build_options(script)
    options.validate()

    configure_dspy(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
    )
    generators = initialize_text_generators(
        use_optimized=options.use_optimized,
        optimized_manifest=options.optimized_manifest,
    )

    state = run_all(StoryState(), generators, prompter, options)

    os.makedirs(args.output_dir, exist_ok=True)
    md_path = os.path.join(args.output_dir, "story_output.md")
    write_markdown(state, md_path)

    # Also dump the raw state JSON — this is what a worker would persist.
    state_path = os.path.join(args.output_dir, "story_state.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info("Wrote %s and %s", md_path, state_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
