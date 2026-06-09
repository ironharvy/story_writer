"""Regression/eval set seeded with the REAL continuity bugs.

Each gold negative is a minimal manuscript carrying one of the defects found in
the manual review, labelled with the deterministic metric that MUST flag it. The
good cases are the matching false-positive traps that must stay clean (a negated
non-feature, a sanctioned spelled-out alias, a sanctioned meta-narrative device).

This is the deterministic backbone the pipeline must keep passing. It is exercised
by ``test_regression.py`` and can be run directly::

    python -m bench.regression
"""
from __future__ import annotations

from dataclasses import dataclass, field

import canon as canon_mod
import canon_metrics
import qa


@dataclass
class RegressionCase:
    """One labelled fixture. A gold negative names the check that must FAIL; a
    good case asserts the suite stays clean (optionally for a specific check)."""

    id: str
    text: str
    must_fail: str | None = None          # gold negative -> this check must FAIL
    must_not_fail: list[str] = field(default_factory=list)  # good case guards


def _story(chapters: dict[str, str], characters: str = "") -> str:
    parts = ["# Story", "", "## World bible", "", "### Characters", "", characters,
             "", "## Final Story", ""]
    for title, body in chapters.items():
        parts += [f"### {title}", "", body, ""]
    return "\n".join(parts)


# --- gold negatives (the real bugs) -----------------------------------------

GOLD_NEGATIVES = [
    RegressionCase(
        id="scaffolding_ch19",
        text=_story({"Chapter 18: The Break":
                     "The breaking was Ch.19's. He had no Ch.19."}),
        must_fail="scaffolding_leak",
    ),
    RegressionCase(
        id="timeline_eight_years",
        text=_story({"Chapter 9: After":
                     "Over the next eight years, the city forgot her name."}),
        must_fail="timeline_age",
    ),
    RegressionCase(
        id="name_reveal_marta_elias",
        text=_story({"Chapter 10: The Warning":
                     'Marta caught his sleeve. "Elias," she whispered, "run now."'}),
        must_fail="name_arc",
    ),
    RegressionCase(
        id="appearance_blue_eyes",
        text=_story({"Chapter 1: Open":
                     "She studied his blue eyes for a long, cold moment."}),
        must_fail="canon_fact",
    ),
]


# --- good cases (must stay clean) -------------------------------------------

GOOD_CASES = [
    RegressionCase(
        id="appearance_no_horns",
        text=_story({"Chapter 1: Open":
                     "No horns, no wings — only his grey eyes, the same as ever."}),
        must_not_fail=["canon_fact"],
    ),
    RegressionCase(
        id="alias_eighty_four",
        text=_story(
            {"Chapter 8: The Roll": "They called for Vessel Eighty-Four, and Vessel "
                                    "Eighty-Four stepped forward."},
            characters="1. Vessel 84, a vessel of the order",
        ),
        must_not_fail=["name_drift"],
    ),
    RegressionCase(
        id="device_reader",
        text=_story({"Chapter 5: Stand":
                     "The reader sees her stand. She did not move from the doorway."}),
        must_not_fail=["scaffolding_leak"],
    ),
]


def deterministic_findings(text: str, canon) -> list:
    """The full deterministic suite used for regression: canon metrics plus the
    alias-aware name-drift check, run on alias-normalized text."""
    norm = canon_mod.apply_accepted_aliases(text, canon)
    findings = list(canon_metrics.run_all(norm, canon))
    findings += qa.check_name_drift(norm)
    return findings


def evaluate_case(case: RegressionCase, canon) -> tuple[bool, str]:
    """Return (passed, detail) for one case against the deterministic suite."""
    findings = deterministic_findings(case.text, canon)
    failed = {f.check for f in findings if f.severity == "fail"}

    if case.must_fail is not None:
        ok = case.must_fail in failed
        return ok, (f"expected FAIL[{case.must_fail}]; "
                    f"got fails={sorted(failed) or '∅'}")

    leaked = [c for c in case.must_not_fail if c in failed]
    return not leaked, (f"expected clean for {case.must_not_fail}; "
                        f"got fails={sorted(failed) or '∅'}")


def run(canon=None) -> tuple[bool, list[tuple[str, bool, str]]]:
    """Evaluate every case; return (all_passed, per-case rows)."""
    canon = canon if canon is not None else canon_mod.load_canon()
    rows: list[tuple[str, bool, str]] = []
    for case in (*GOLD_NEGATIVES, *GOOD_CASES):
        passed, detail = evaluate_case(case, canon)
        rows.append((case.id, passed, detail))
    return all(p for _, p, _ in rows), rows


if __name__ == "__main__":
    import sys

    all_passed, rows = run()
    for case_id, passed, detail in rows:
        print(f"[{'PASS' if passed else 'FAIL'}] {case_id}: {detail}")
    print(f"\n{'ALL PASS' if all_passed else 'REGRESSIONS PRESENT'}")
    sys.exit(0 if all_passed else 1)
