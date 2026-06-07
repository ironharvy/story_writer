"""F3 — LLM-judge interpretation + scorecard merge logic.

These lock in the spec-critical parts that don't need a model: the Tier-2
finding interpretation, the Tier-3 rubric coercion, and the ``merge_judge``
ship-gate (zero Tier-2 FAILs + rubric average >= 4.0 with no axis < 3). The
thin LM-calling wrappers are exercised by the live sweep, not here.
"""
from __future__ import annotations

from types import SimpleNamespace

from bench.criteria import Scorecard, _rubric_ok, merge_judge
from bench.judge import (
    RUBRIC_AXES,
    _coerce_rubric,
    _contradiction_findings,
    _premise_findings,
    _protagonist_findings,
    judge_premise,
)
from qa import Finding


def _clean_card() -> Scorecard:
    """A Tier-1 scorecard that ships (no fails, budgets respected)."""
    return Scorecard(ship=True, fails=[], warns=[], budgets={
        "name_drift": {"count": 0, "budget": 2, "over": False},
        "phrase_reuse": {"count": 1, "budget": 5, "over": False},
    })


def _rubric(value: int = 4) -> dict[str, dict]:
    return {axis: {"score": value, "note": ""} for axis in RUBRIC_AXES}


# --- merge_judge ship-gating ------------------------------------------------


def test_merge_judge_tier2_fail_blocks_ship():
    merged = merge_judge(_clean_card(), [Finding("protagonist_naming", "fail", "unnamed")], _rubric(5))
    assert merged.ship is False
    assert merged.judged is True
    assert any(f["check"] == "protagonist_naming" for f in merged.fails)


def test_merge_judge_clean_with_good_rubric_ships():
    merged = merge_judge(_clean_card(), [Finding("contradictions", "info", "none")], _rubric(4))
    assert merged.ship is True
    assert merged.rubric_average == 4.0


def test_merge_judge_low_axis_blocks_ship():
    rubric = _rubric(5)
    rubric["ending"] = {"score": 2, "note": "fizzles"}
    merged = merge_judge(_clean_card(), [], rubric)
    assert merged.ship is False  # an axis below the floor of 3


def test_merge_judge_low_average_blocks_ship():
    merged = merge_judge(_clean_card(), [], _rubric(3))  # avg 3.0 < 4.0
    assert merged.ship is False
    assert merged.rubric_average == 3.0


def test_merge_judge_warns_do_not_block_ship():
    merged = merge_judge(_clean_card(), [Finding("contradictions", "warn", "wobble")], _rubric(4))
    assert merged.ship is True
    assert any(w["check"] == "contradictions" for w in merged.warns)


def test_merge_judge_respects_tier1_budget_over():
    card = Scorecard(ship=False, fails=[], warns=[], budgets={
        "name_drift": {"count": 5, "budget": 2, "over": True},
        "phrase_reuse": {"count": 1, "budget": 5, "over": False},
    })
    merged = merge_judge(card, [], _rubric(5))
    assert merged.ship is False  # a blown Tier-1 budget still blocks


def test_rubric_ok_empty_is_not_a_blocker():
    assert _rubric_ok({}) == (None, True)


def test_rubric_ok_threshold():
    assert _rubric_ok(_rubric(4))[1] is True
    assert _rubric_ok(_rubric(3))[1] is False


# --- Tier-2 finding interpretation ------------------------------------------


def test_protagonist_findings_generic_referent_fails():
    verdict = SimpleNamespace(protagonist_name="", relies_on_generic_referent=True,
                              inconsistent_names=[], note="cold open")
    assert any(f.severity == "fail" for f in _protagonist_findings(verdict))


def test_protagonist_findings_multiple_names_fails():
    verdict = SimpleNamespace(protagonist_name="Mara", relies_on_generic_referent=False,
                              inconsistent_names=["Mara", "Mira"], note="")
    fails = [f for f in _protagonist_findings(verdict) if f.severity == "fail"]
    assert len(fails) == 1
    assert "multiple names" in fails[0].message


def test_protagonist_findings_clean_is_info():
    verdict = SimpleNamespace(protagonist_name="Mara", relies_on_generic_referent=False,
                              inconsistent_names=[], note="")
    assert [f.severity for f in _protagonist_findings(verdict)] == ["info"]


def test_contradiction_findings_splits_hard_and_minor():
    verdict = SimpleNamespace(hard_contradictions=["dead in ch3, acts in ch5"],
                              minor_issues=["clock drifts"])
    assert sorted(f.severity for f in _contradiction_findings(verdict)) == ["fail", "warn"]


def test_contradiction_findings_none_is_info():
    verdict = SimpleNamespace(hard_contradictions=[], minor_issues=[])
    assert [f.severity for f in _contradiction_findings(verdict)] == ["info"]


def test_premise_findings_drift_fails():
    assert _premise_findings(SimpleNamespace(dramatizes_premise=False, note="generic"))[0].severity == "fail"


def test_premise_findings_ok_is_info():
    assert _premise_findings(SimpleNamespace(dramatizes_premise=True, note="on point"))[0].severity == "info"


def test_judge_premise_empty_idea_skips():
    findings = judge_premise("", "some manuscript")
    assert findings[0].severity == "info"
    assert "skipped" in findings[0].message


# --- Tier-3 rubric coercion -------------------------------------------------


def test_coerce_rubric_clamps_and_drops_bad_axes():
    raw = {axis: 6 for axis in RUBRIC_AXES}  # over the max
    del raw["cohesion"]                       # missing one axis
    raw["arc_structure"] = "bad"              # non-integer score
    rubric = _coerce_rubric(raw, {"ending": "earned"})
    assert "cohesion" not in rubric           # missing axis dropped
    assert "arc_structure" not in rubric      # non-int dropped
    assert rubric["ending"]["score"] == 5     # 6 clamped down to 5
    assert rubric["ending"]["note"] == "earned"
