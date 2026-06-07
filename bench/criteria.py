"""Deterministic Tier-1 checks from bench/eval-spec.md.

Thin wrapper over the deterministic checks in `qa.py` (chapter length + word
band, character presence, name drift, cross-chapter phrase reuse, structural
completeness, protagonist-placeholder). The length thresholds live in `qa.py`
(the single source of truth): a hard floor of 80 words FAILs as broken, and
the 300–2500 target band WARNs as thin/bloated. This module bundles those into
a scorecard with the spec's soft budgets.

The LLM-backed gates (Tier-2 POV / prose lint, Tier-3 rubric) are out of scope
here. `bench/evaluate.py` is the unified scorecard that adds them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import qa

# Soft budgets from bench/eval-spec.md (T1.3 / T1.4). Length thresholds are
# qa.CHAPTER_MIN_WORDS (hard floor) and qa.CHAPTER_TARGET_BAND (warn band).
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


def score(story_md_path: Path) -> Scorecard:
    """Run all deterministic checks against a story markdown artifact.

    Returns a Scorecard with separated FAILs and WARNs plus per-budget
    counts. `ship=True` only if zero Tier-1 FAILs and every budget respected.
    """
    # run_all covers length (hard floor 80), presence, drift, phrase reuse.
    # The remaining deterministic gates from the spec are added explicitly.
    text = Path(story_md_path).read_text(encoding="utf-8")
    findings = qa.run_all(story_md_path)
    findings.extend(qa.check_structure(text))
    findings.extend(qa.check_placeholder_protagonist(text))
    findings.extend(qa.check_chapter_band(text))

    fails: list[dict] = []
    warns: list[dict] = []
    drift_count = 0
    reuse_count = 0

    for f in findings:
        record = {"check": f.check, "severity": f.severity, "message": f.message}
        # Count only real (warn-level) budget hits, not the "no drift" info line.
        if f.check == "name_drift" and f.severity == "warn":
            drift_count += 1
        if f.check == "cross_chapter_phrase_reuse" and f.severity == "warn":
            reuse_count += 1
        if f.severity == "fail":
            fails.append(record)
        elif f.severity == "warn":
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
