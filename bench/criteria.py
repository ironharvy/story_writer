"""Deterministic Tier-1 checks from bench/eval-spec.md.

Thin wrapper over the existing `qa.py` and `pov_check.py` modules — those
already implement the same algorithms (chapter length, character presence,
name drift, cross-chapter phrase reuse, POV classification). This module
applies the spec's thresholds (hard floor 300 words vs qa.py's runtime
default 80, 5-gram budget of 5, etc.) and emits a scorecard.

Tier-2/Tier-3 LLM-judge checks live in `bench/judge.py`; :func:`merge_judge`
folds their findings + rubric into a `Scorecard` here, and `bench/score.py
--llm-judge` wires the two together. This module stays deterministic and
model-free so the default scoring path is fast.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import qa

# Thresholds from bench/eval-spec.md
HARD_CHAPTER_FLOOR = 300         # T1.1 gate
TARGET_CHAPTER_BAND = (800, 2500)  # T1.1 warn outside (not yet enforced)
NAME_DRIFT_BUDGET = 2            # T1.3
PHRASE_REUSE_BUDGET = 5          # T1.4
RUBRIC_SHIP_AVERAGE = 4.0        # Tier-3 ship threshold (average)
RUBRIC_AXIS_FLOOR = 3            # Tier-3 ship threshold (per-axis floor)


@dataclass
class Scorecard:
    """One scorecard per draft.

    Tier-1 fields (``fails`` / ``warns`` / ``budgets``) are always populated.
    The Tier-2/Tier-3 fields (``rubric`` / ``rubric_average`` / ``judged``) stay
    empty unless an LLM judge is merged in via :func:`merge_judge`.
    """
    ship: bool
    fails: list[dict]
    warns: list[dict]
    budgets: dict[str, dict]
    rubric: dict[str, dict] = field(default_factory=dict)
    rubric_average: float | None = None
    judged: bool = False

    def to_json(self) -> str:
        """Serialise the scorecard to indented JSON."""
        return json.dumps(asdict(self), indent=2)


def score(story_md_path: Path) -> Scorecard:
    """Run all deterministic checks against a story markdown artifact.

    Returns a Scorecard with separated FAILs and WARNs plus per-budget
    counts. `ship=True` only if zero Tier-1 FAILs and every budget respected.
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

    budgets = {
        "name_drift": {"count": drift_count, "budget": NAME_DRIFT_BUDGET,
                       "over": drift_count > NAME_DRIFT_BUDGET},
        "phrase_reuse": {"count": reuse_count, "budget": PHRASE_REUSE_BUDGET,
                         "over": reuse_count > PHRASE_REUSE_BUDGET},
    }

    ship = not fails and not any(b["over"] for b in budgets.values())
    return Scorecard(ship=ship, fails=fails, warns=warns, budgets=budgets)


def _rubric_ok(rubric: dict[str, dict]) -> tuple[float | None, bool]:
    """Return ``(average, ship_ok)`` for a Tier-3 rubric.

    An empty rubric is treated as "not evaluated" -> ``(None, True)`` so it
    never blocks the ship gate on its own. Otherwise ship requires the spec's
    average >= 4.0 with no axis below 3.
    """
    scores = [axis["score"] for axis in rubric.values() if "score" in axis]
    if not scores:
        return None, True
    average = sum(scores) / len(scores)
    ship_ok = average >= RUBRIC_SHIP_AVERAGE and min(scores) >= RUBRIC_AXIS_FLOOR
    return round(average, 2), ship_ok


def merge_judge(card: Scorecard, findings: list[qa.Finding], rubric: dict[str, dict]) -> Scorecard:
    """Fold Tier-2 findings + the Tier-3 rubric into a Tier-1 scorecard.

    Re-derives ``ship`` to additionally require zero Tier-2 FAILs and the
    Tier-3 rubric ship-gate (see :func:`_rubric_ok`), per bench/eval-spec.md.
    """
    fails = list(card.fails)
    warns = list(card.warns)
    for f in findings:
        record = {"check": f.check, "severity": f.severity, "message": f.message}
        if f.severity == "fail":
            fails.append(record)
        elif f.severity == "warn":
            warns.append(record)
    average, rubric_ok = _rubric_ok(rubric)
    ship = not fails and not any(b["over"] for b in card.budgets.values()) and rubric_ok
    return Scorecard(
        ship=ship, fails=fails, warns=warns, budgets=card.budgets,
        rubric=rubric, rubric_average=average, judged=True,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m bench.criteria <story.md>", file=sys.stderr)
        sys.exit(2)
    print(score(Path(sys.argv[1])).to_json())
