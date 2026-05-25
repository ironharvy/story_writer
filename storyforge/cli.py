"""StoryForge command line: `generate` a manuscript, or `eval` one.

    python -m storyforge generate --idea "..." [--genre ... --pov ... --chapters 7]
    python -m storyforge generate --idea "..." --eval          # generate then score
    python -m storyforge eval path/to/manuscript.md --idea "..."
    python -m storyforge generate --resume --run-dir runs/<dir>

When run interactively without enough detail, it asks a few clarifying questions
first. Use --yes / --non-interactive to skip and let the model infer.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

from . import config
from .evaluate import harness
from .pipeline import Pipeline

# Length presets named after the word-count categories writers actually use.
# Each preset's words-per-chapter stays inside the eval's 800-2500w band
# (see evaluate/tier1.py BAND_LOW/BAND_HIGH), so longer stories come from MORE
# chapters, not longer ones. Totals are approximate (chapters x words/chapter).
LENGTH_PRESETS = {  # name -> (num_chapters, words_per_chapter)   ~total
    "flash":     (1, 900),    # ~900w     flash fiction
    "short":     (4, 1300),   # ~5,000w   short story
    "novelette": (8, 1500),   # ~12,000w  novelette
    "novella":   (14, 1700),  # ~24,000w  novella
    "novel":     (28, 1800),  # ~50,000w  short novel (hours, local)
    "epic":      (60, 1800),  # ~108,000w stress test (many hours)
}
# Back-compat for the old names so existing commands/docs keep working.
LENGTH_ALIASES = {"standard": "novelette", "long": "novella"}
DEFAULT_LENGTH = "novelette"
# Total >= this many chapters is a long, resumable run worth flagging.
LONG_RUN_CHAPTERS = 20


def _length_dims(name: str | None) -> tuple[int, int]:
    """Resolve a length name (or alias) to (num_chapters, words_per_chapter).

    Unknown names fall back to the default rather than erroring; the --length
    flag itself is constrained by argparse `choices`."""
    key = LENGTH_ALIASES.get(name or "", name or "")
    return LENGTH_PRESETS.get(key, LENGTH_PRESETS[DEFAULT_LENGTH])


def _length_help() -> str:
    rows = []
    for name, (nc, wpc) in LENGTH_PRESETS.items():
        rows.append(f"{name} (~{nc * wpc:,}w)")
    return "story length: " + ", ".join(rows) + "; aliases: " + ", ".join(LENGTH_ALIASES)


def _console():
    try:
        from rich.console import Console
        return Console()
    except Exception:  # pragma: no cover
        return None


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:40] or "story")


def _ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt}" + (f" [{default}]" if default else "") + ": ").strip()
    except EOFError:
        return default
    return ans or default


def _clarify(idea: str, args) -> dict:
    """Interactively fill the few choices that shape the book. Empty answer => infer."""
    print("\nA few quick questions (press Enter to let the model decide):")
    overrides: dict = {}
    g = _ask("Genre", args.genre or "")
    if g:
        overrides["genre"] = g
    t = _ask("Tone", args.tone or "")
    if t:
        overrides["tone"] = t
    pov = _ask("POV (first / limited / omniscient)", args.pov or "limited")
    if pov:
        overrides["pov"] = {"first": "first_person", "limited": "third_person_limited",
                            "omniscient": "third_person_omniscient"}.get(pov, pov)
    print("  length: " + ", ".join(f"{n} (~{nc*wpc:,}w)"
                                    for n, (nc, wpc) in LENGTH_PRESETS.items()))
    length = _ask("Length", args.length or DEFAULT_LENGTH)
    nc, wpc = _length_dims(length)
    overrides["num_chapters"], overrides["words_per_chapter"] = nc, wpc
    e = _ask("Ending (kind of ending you want)", args.ending or "")
    if e:
        overrides["ending"] = e
    print()
    return overrides


def _build_overrides(args) -> dict:
    overrides: dict = {}
    for k in ("genre", "tone", "ending"):
        if getattr(args, k):
            overrides[k] = getattr(args, k)
    if args.pov:
        overrides["pov"] = {"first": "first_person", "limited": "third_person_limited",
                            "omniscient": "third_person_omniscient"}.get(args.pov, args.pov)
    if args.length:
        nc, wpc = _length_dims(args.length)
        overrides["num_chapters"], overrides["words_per_chapter"] = nc, wpc
    if args.chapters:
        overrides["num_chapters"] = args.chapters
    if args.words:
        overrides["words_per_chapter"] = args.words
    return overrides


def _emit(console, msg: str) -> None:
    (console.print if console else print)(msg)


def _warn_if_bare(model: str) -> None:
    """A resolved model with no provider prefix is usually a typo or a missing
    provider; warn rather than let litellm fail cryptically downstream."""
    if "/" not in model:
        print(f"note: '{model}' has no provider prefix; passing to litellm as-is",
              file=sys.stderr)


def cmd_generate(args) -> int:
    console = _console()
    config.configure_logging(args.verbose)
    if config.setup_langfuse():
        _emit(console, "• Langfuse tracing enabled")

    draft_model = config.resolve_model(args.model, config.DEFAULT_DRAFT)
    _warn_if_bare(draft_model)
    _emit(console, f"• Drafting with {draft_model}")

    if args.resume:
        if not args.run_dir:
            print("--resume requires --run-dir", file=sys.stderr)
            return 2
        run_dir = Path(args.run_dir)
        idea = ""
    else:
        idea = args.idea or _ask("Story idea")
        if not idea:
            print("No idea given.", file=sys.stderr)
            return 2
        run_dir = Path(args.run_dir) if args.run_dir else (
            Path("runs") / f"{_slug(idea)}-{_dt.datetime.now():%Y%m%d-%H%M%S}")

    interactive = sys.stdin.isatty() and not (args.yes or args.non_interactive)
    overrides = _clarify(idea, args) if (interactive and not args.resume) else _build_overrides(args)

    nc = overrides.get("num_chapters")
    if nc and nc >= LONG_RUN_CHAPTERS:
        _emit(console, f"• {nc}-chapter run — this is long; it writes incrementally, so "
              f"`--resume --run-dir {run_dir}` continues if interrupted.")

    pipe = Pipeline(draft_model=draft_model, run_dir=run_dir, use_critic=not args.no_critic,
                    use_derepeat=not args.no_derepeat, console=console)
    ms_path = pipe.run(idea, overrides=overrides, resume=args.resume)

    if args.eval:
        state = json.loads((run_dir / "state.json").read_text())
        idea = idea or state.get("idea", "")
        judge_model = config.resolve_model(args.judge, config.DEFAULT_JUDGE)
        _warn_if_bare(judge_model)
        _emit(console, f"\n• Evaluating with judge {judge_model} (tier1_only={args.tier1_only})…")
        sc = harness.evaluate_file(ms_path, idea, draft_model=draft_model,
                                   judge_model=judge_model, tier1_only=args.tier1_only)
        harness.write_scorecard(sc, run_dir / "scorecard.json")
        print("\n" + harness.summarize(sc))
        print(f"\nScorecard: {run_dir / 'scorecard.json'}")
    return 0


def cmd_eval(args) -> int:
    console = _console()
    config.configure_logging(args.verbose)
    if config.setup_langfuse():
        (console.print if console else print)("• Langfuse tracing enabled")
    path = Path(args.manuscript)
    if not path.exists():
        print(f"No such file: {path}", file=sys.stderr)
        return 2
    idea = args.idea
    if not idea:  # fall back to the manuscript's own premise
        from .evaluate.parse import parse_manuscript
        idea = parse_manuscript(path.read_text()).premise
    judge_model = config.resolve_model(args.judge, config.DEFAULT_JUDGE)
    draft_model = config.resolve_model(args.draft_model, "unknown")
    _warn_if_bare(judge_model)
    _emit(console, f"• Judge {judge_model} · recorded draft model {draft_model}")
    sc = harness.evaluate_file(path, idea, draft_model=draft_model, judge_model=judge_model,
                               tier1_only=args.tier1_only)
    out = Path(args.out) if args.out else path.with_suffix(".scorecard.json")
    harness.write_scorecard(sc, out)
    print(harness.summarize(sc))
    print(f"\nScorecard: {out}")
    return 0 if sc["ship"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="storyforge", description="Generate and evaluate manuscripts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate a manuscript from an idea")
    g.add_argument("--idea", help="the story idea")
    g.add_argument("--genre"); g.add_argument("--tone"); g.add_argument("--ending")
    g.add_argument("--pov", choices=["first", "limited", "omniscient"])
    g.add_argument("--length", choices=list(LENGTH_PRESETS) + list(LENGTH_ALIASES),
                   metavar="LENGTH", help=_length_help())
    g.add_argument("--chapters", type=int, help="override chapter count")
    g.add_argument("--words", type=int, help="override words per chapter")
    g.add_argument("--model", help="draft model: preset (fast/quality/groq/deepseek) or "
                   "provider/model (e.g. ollama/qwen3.6:27b, deepseek/deepseek-chat)")
    g.add_argument("--run-dir", help="output dir (also used with --resume)")
    g.add_argument("--resume", action="store_true", help="resume an interrupted run")
    g.add_argument("--no-critic", action="store_true", help="skip the critic->revise pass (faster)")
    g.add_argument("--no-derepeat", action="store_true",
                   help="skip the de-repetition polish pass (faster)")
    g.add_argument("--yes", "--non-interactive", dest="non_interactive", action="store_true",
                   help="don't ask clarifying questions; infer missing choices")
    g.add_argument("--eval", action="store_true", help="evaluate after generating")
    g.add_argument("--judge", help="judge model for --eval: preset or provider/model "
                   "(e.g. ollama/qwen3.6:27b)")
    g.add_argument("--tier1-only", action="store_true", help="eval: deterministic checks only")
    g.add_argument("--verbose", action="store_true")
    g.set_defaults(func=cmd_generate, yes=False)

    e = sub.add_parser("eval", help="evaluate an existing manuscript")
    e.add_argument("manuscript")
    e.add_argument("--idea", help="original idea (for premise-fidelity); defaults to the premise")
    e.add_argument("--judge", help="judge model: preset (fast/quality/groq/deepseek) or "
                   "provider/model (e.g. ollama/qwen3.6:27b, deepseek/deepseek-chat)")
    e.add_argument("--draft-model", help="record which model wrote it (preset or provider/model)")
    e.add_argument("--tier1-only", action="store_true")
    e.add_argument("--out", help="scorecard output path")
    e.add_argument("--verbose", action="store_true")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
