"""Unit tests for the ship-gate logic (no LLM)."""
from __future__ import annotations

from pathlib import Path

from storyforge.evaluate import FAIL, PASS, WARN, Check
from storyforge.evaluate import harness

FIX = Path(__file__).parent / "fixtures"


def test_tier1_only_cannot_ship_but_reports_checks():
    sc = harness.evaluate((FIX / "clean_small.md").read_text(), "a clockmaker's apprentice",
                          draft_model="x", judge_model="y", tier1_only=True)
    assert sc["ship"] is False
    assert any("Tier-2/3 not run" in b for b in sc["ship_blockers"])
    ids = {c["id"] for c in sc["checks"]}
    assert {"T1.1", "T1.2", "T1.3", "T1.4", "T1.5"} <= ids
    assert sc["rubric"] == {}


def test_t1_7_truncated_ending_is_a_ship_blocker():
    sc = harness.evaluate((FIX / "truncated_ending.md").read_text(), "a dying queen's decree",
                          draft_model="x", judge_model="y", tier1_only=True)
    assert sc["ship"] is False
    assert any(b.startswith("T1.7 FAIL") for b in sc["ship_blockers"])


def test_soft_budget_blockers():
    over_drift = {"T1.3": Check("T1.3", "Name drift", WARN, "", {"flags": [1, 2, 3]})}
    assert harness._soft_budget_blockers(over_drift)  # 3 > 2

    ok_drift = {"T1.3": Check("T1.3", "Name drift", WARN, "", {"flags": [1, 2]})}
    assert not harness._soft_budget_blockers(ok_drift)  # 2 == budget

    over_reuse = {"T1.4": Check("T1.4", "reuse", WARN, "", {"count": 6})}
    assert harness._soft_budget_blockers(over_reuse)  # 6 > 5

    over_rep = {"T1.5": Check("T1.5", "rep", WARN, "", {"flagged": [("veins", 30)]})}
    assert harness._soft_budget_blockers(over_rep)

    clean = {
        "T1.3": Check("T1.3", "", PASS, "", {"flags": []}),
        "T1.4": Check("T1.4", "", PASS, "", {"count": 4}),
        "T1.5": Check("T1.5", "", PASS, "", {"flagged": []}),
    }
    assert harness._soft_budget_blockers(clean) == []
