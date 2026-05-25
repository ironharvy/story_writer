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
import os
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


def _round_judge(args, draft_model: str) -> str:
    """The judge for iterative rounds. An explicit --judge / SF_JUDGE_MODEL wins;
    otherwise pick a local judge independent of the drafter. Final safety net: if
    the resolved judge equals the drafter (self-grading), switch and warn."""
    if args.judge or os.getenv("SF_JUDGE_MODEL"):
        jm = config.resolve_model(args.judge, config.DEFAULT_JUDGE)
    else:
        jm = config.pick_independent_judge(draft_model)
    if jm == draft_model:
        alt = config.pick_independent_judge(draft_model)
        if alt != draft_model:
            print(f"note: judge == draft ({draft_model}) (self-grading); switching to {alt}",
                  file=sys.stderr)
            jm = alt
    return jm


def _final_judge(round_judge: str) -> str:
    """The credible final ship decision: hosted DeepSeek when a key is present,
    else fall back to the local round judge with a printed warning."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return config.resolve_model("deepseek", config.PRESETS["deepseek"])
    print(f"note: DEEPSEEK_API_KEY not set; final judge falls back to local {round_judge}",
          file=sys.stderr)
    return round_judge


def _scorecard_key(sc: dict) -> tuple:
    """Comparable quality key for the regression guard — higher is strictly better."""
    rubric = sc.get("rubric") or {}
    return (
        bool(sc.get("ship", False)),
        -len(sc.get("ship_blockers", [])),
        -sum(1 for c in sc.get("checks", []) if c.get("severity") == "fail"),
        float(rubric.get("average", 0.0) or 0.0),
    )


def _climax_numbers(spine) -> set[int]:
    """Chapters that legitimately resolve the conflict — from structural_role, or
    the last chapter as a fallback (the spine prompt makes it climax+resolution)."""
    nums = {p.number for p in spine
            if (getattr(p, "structural_role", "") or "").lower() in ("climax", "resolution")}
    if nums:
        return nums
    return {max(p.number for p in spine)} if spine else set()


def _low_scene_axes(sc: dict) -> bool:
    """True when the scene-craft rubric axes are below floor — the signal that a
    scene-level redraft (not just a line polish) may be warranted."""
    scores = (sc.get("rubric") or {}).get("scores", {}) or {}
    return any(0 < scores.get(ax, 5) < 3 for ax in ("scene_vs_summary", "character_agency"))


def _derive_notes(sc: dict) -> list[str]:
    """Turn a scorecard dict into prioritized, concrete editorial notes for polish().
    Operates on the serialized scorecard (lists of dicts), not Check objects."""
    by_id = {c["id"]: c for c in sc.get("checks", [])}
    notes: list[str] = []

    t23 = by_id.get("T2.3")
    if t23:
        for c in t23.get("details", {}).get("contradictions", []):
            if isinstance(c, dict) and str(c.get("severity", "")).lower() == "hard":
                notes.append(
                    f"Resolve continuity contradiction (chapters {c.get('chapters', '?')}): "
                    f"{c.get('detail', '')}. The events must happen once, in one chapter; "
                    "do not re-stage the same scene twice.")

    t16 = by_id.get("T1.6")
    if t16 and t16.get("severity") == "warn":
        for p in t16.get("details", {}).get("pairs", [])[:3]:
            notes.append(
                f"Chapters '{p.get('a')}' and '{p.get('b')}' overlap heavily — they stage "
                "the same scene. Keep the event in ONE chapter and rewrite the other so it "
                "does not repeat the same beats or imagery.")

    t17 = by_id.get("T1.7")
    if t17 and t17.get("severity") == "fail":
        notes.append("Write a complete, properly punctuated ending for the final chapter; "
                     "it currently reads as truncated.")

    rubric = sc.get("rubric") or {}
    scores = rubric.get("scores", {}) or {}
    justs = rubric.get("justifications", {}) or {}
    for ax, v in scores.items():
        if isinstance(v, (int, float)) and 0 < v < 3:
            notes.append(f"Improve {ax} ({v}/5): {justs.get(ax, '')}".strip())
    avg = rubric.get("average", 0.0) or 0.0
    if scores and avg and avg < (rubric.get("threshold_avg", 4.0) or 4.0):
        notes.append(f"Overall quality is below the bar (rubric average {avg}); deepen "
                     "scenes, sharpen prose, and strengthen the through-line.")

    for b in sc.get("ship_blockers", []):
        if b.startswith("T1.3"):
            notes.append("Fix character-name drift: use each character's canonical "
                         "spelling consistently throughout.")
        elif b.startswith("T1.4"):
            notes.append("Reduce cross-chapter phrase reuse: rephrase repeated imagery and "
                         "signature sentences with fresh wording.")
        elif b.startswith("T1.5"):
            notes.append("Reduce over-repetition of content words; vary diction.")

    for gid, lead in (("T2.1", "Fix point-of-view consistency"),
                      ("T2.2", "Refer to the protagonist by their proper name consistently"),
                      ("T2.4", "Bring the story back to the original premise")):
        c = by_id.get(gid)
        if c and c.get("severity") in ("fail", "warn"):
            notes.append(f"{lead}: {c.get('message', '')}".strip())

    seen: set[str] = set()
    out: list[str] = []
    for n in notes:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= 8:
            break
    return out


def _revise_loop(args, pipe, run_dir: Path, ms_path: Path, idea: str,
                 draft_model: str, console) -> None:
    """Self-evaluate -> revise -> re-evaluate until ship or rounds exhausted, then
    make a final ship decision with a credible (hosted) judge. Never ships a
    manuscript worse than the best seen (regression guard)."""
    from .models import RunState

    round_judge = _round_judge(args, draft_model)
    _warn_if_bare(round_judge)
    rounds = 0 if args.no_revise else max(0, args.revise_rounds)

    def _eval(judge_model: str) -> dict:
        return harness.evaluate_file(ms_path, idea, draft_model=draft_model,
                                     judge_model=judge_model, tier1_only=args.tier1_only)

    _emit(console, f"\n• Evaluating with round judge {round_judge} "
          f"(tier1_only={args.tier1_only})…")
    best_sc = _eval(round_judge)
    best_text = ms_path.read_text()
    _emit(console, harness.summarize(best_sc))

    # tier1_only can never ship (the gate blocks it), so revising is pointless.
    scene_redraft_used = False
    if not args.tier1_only:
        for r in range(rounds):
            if best_sc.get("ship"):
                break
            notes = _derive_notes(best_sc)
            want_scene_redraft = (pipe.scene_level and _low_scene_axes(best_sc)
                                  and not scene_redraft_used)
            if not notes and not want_scene_redraft:
                break
            state = RunState.from_dict(json.loads((run_dir / "state.json").read_text()))
            climax = _climax_numbers(state.spine)
            if want_scene_redraft:
                scene_redraft_used = True
                _emit(console, f"\n• Revise round {r + 1}/{rounds}: scene/agency axis below "
                      "floor — scene-level redraft")
                pipe.redraft_scene_level(state)
            else:
                _emit(console, f"\n• Revise round {r + 1}/{rounds}: addressing {len(notes)} "
                      f"note(s); climax chapter(s) {sorted(climax) or '?'}")
                pipe.polish(state, notes, climax)
            sc = _eval(round_judge)
            if _scorecard_key(sc) > _scorecard_key(best_sc):
                best_sc, best_text = sc, ms_path.read_text()
                _emit(console, harness.summarize(sc))
                _emit(console, "  ✓ round improved the manuscript")
            else:
                _emit(console, "  ▲ round did not improve; restoring best draft and stopping")
                break

    # Restore the best manuscript before the final decision (last round may be worse).
    if ms_path.read_text() != best_text:
        ms_path.write_text(best_text)

    if args.tier1_only:
        final_sc = best_sc
    else:
        final_judge = _final_judge(round_judge)
        _warn_if_bare(final_judge)
        _emit(console, f"\n• Final evaluation with {final_judge}…")
        final_sc = _eval(final_judge)

    harness.write_scorecard(final_sc, run_dir / "scorecard.json")
    print("\n" + harness.summarize(final_sc))
    print(f"\nScorecard: {run_dir / 'scorecard.json'}")


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

    climax_model = config.resolve_model(args.climax_model, "") if args.climax_model else ""
    if climax_model:
        _warn_if_bare(climax_model)
        _emit(console, f"• Climax chapter escalated to {climax_model}")
    if args.scene_level:
        _emit(console, "• Scene-level drafting enabled (more LLM calls)")
    pipe = Pipeline(draft_model=draft_model, run_dir=run_dir, use_critic=not args.no_critic,
                    use_derepeat=not args.no_derepeat, console=console,
                    climax_model=climax_model, scene_level=args.scene_level)
    ms_path = pipe.run(idea, overrides=overrides, resume=args.resume)

    if args.eval:
        state = json.loads((run_dir / "state.json").read_text())
        idea = idea or state.get("idea", "")
        _revise_loop(args, pipe, run_dir, ms_path, idea, draft_model, console)
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
    draft_model = config.resolve_model(args.draft_model, "unknown")
    judge_model = _round_judge(args, draft_model)
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
    g.add_argument("--climax-model", dest="climax_model",
                   help="escalate ONLY the climax chapter to this model (preset or provider/model)")
    g.add_argument("--scene-level", dest="scene_level", action="store_true",
                   help="draft chapters scene-by-scene (more LLM calls, richer scenes)")
    g.add_argument("--yes", "--non-interactive", dest="non_interactive", action="store_true",
                   help="don't ask clarifying questions; infer missing choices")
    g.add_argument("--eval", action="store_true", help="evaluate after generating")
    g.add_argument("--judge", help="judge model for --eval: preset or provider/model "
                   "(e.g. ollama/qwen3.6:27b)")
    g.add_argument("--revise-rounds", type=int, default=2, dest="revise_rounds",
                   help="max self-evaluate->revise rounds during --eval (default 2)")
    g.add_argument("--no-revise", action="store_true",
                   help="eval once without the revise loop (old one-shot behavior)")
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
