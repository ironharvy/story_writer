"""Deterministic Tier-1 checks from bench/eval-spec.md.

Thin wrapper over the existing `qa.py` and `pov_check.py` modules — those
already implement the same algorithms (chapter length, character presence,
name drift, cross-chapter phrase reuse, POV classification). This module
applies the spec's thresholds (hard floor 300 words vs qa.py's runtime
default 80, 5-gram budget of 5, etc.) and emits a scorecard.

Tier-2/Tier-3 LLM-judge checks are out of scope here — they'll plug in
later via a `bench/judge.py` once the prompt + judge model are chosen.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import qa

# Thresholds from bench/eval-spec.md
HARD_CHAPTER_FLOOR = 300         # T1.1 gate
TARGET_CHAPTER_BAND = (800, 2500)  # T1.1 warn outside (not yet enforced)
NAME_DRIFT_BUDGET = 2            # T1.3
PHRASE_REUSE_BUDGET = 5          # T1.4


@dataclass
class Scorecard:
    """One scorecard per draft."""
    ship: bool
    fails: list[dict]
    warns: list[dict]
    budgets: dict[str, dict]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def score(story_md_path: Path, *, judge_tier2: bool = False,
          per_chapter: bool = False, idea: str = "") -> Scorecard:
    """Run all deterministic checks against a story markdown artifact.

    Returns a Scorecard with separated FAILs and WARNs plus per-budget
    counts. `ship=True` only if zero Tier-1 FAILs and every budget respected.

    When ``judge_tier2`` is set, the LLM-judge gates in ``bench.judge`` are also
    run and folded in (the caller must have configured a DSPy LM first). This is
    off by default so the deterministic bench path never touches the model.
    """
    findings = qa.run_all(story_md_path)
    # T1.1 in qa.py uses default min_words=80; re-run with spec floor of 300
    text = Path(story_md_path).read_text(encoding="utf-8")
    findings.extend(qa.check_chapter_length(text, min_words=HARD_CHAPTER_FLOOR))

    fails: list[dict] = []
    warns: list[dict] = []
    drift_count = 0
    reuse_count = 0

    for f in findings:
        record = {"check": f.check, "severity": f.severity, "message": f.message}
        if f.check == "name_drift":
            drift_count += 1
        if f.check == "cross_chapter_phrase_reuse":
            reuse_count += 1
        if f.severity == "fail":
            fails.append(record)
        elif f.severity == "warn":
            warns.append(record)

    if judge_tier2:
        from bench import judge as _judge  # local import keeps dspy out of the default path
        for record in _judge.judge(Path(story_md_path), per_chapter=per_chapter, idea=idea):
            if record["severity"] == "fail":
                fails.append(record)
            elif record["severity"] == "warn":
                warns.append(record)

    budgets = {
        "name_drift": {"count": drift_count, "budget": NAME_DRIFT_BUDGET,
                       "over": drift_count > NAME_DRIFT_BUDGET},
        "phrase_reuse": {"count": reuse_count, "budget": PHRASE_REUSE_BUDGET,
                         "over": reuse_count > PHRASE_REUSE_BUDGET},
    }

    ship = not fails and not any(b["over"] for b in budgets.values())
    return Scorecard(ship=ship, fails=fails, warns=warns, budgets=budgets)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m bench.criteria <story.md>", file=sys.stderr)
        sys.exit(2)
    print(score(Path(sys.argv[1])).to_json())
