"""Tests for the LLM-judge logic (bench/judge.py) — no model required.

The LM-calling functions (find_contradictions, judge_premise_fidelity,
score_rubric) are exercised in integration; here we test the deterministic
parts: the check_* threshold/severity logic with injected results, the
coercion of messy LM outputs, and the median aggregation.
"""
from __future__ import annotations

from bench.judge import (
    RUBRIC_AXES,
    AxisScore,
    Contradiction,
    PremiseVerdict,
    _coerce_axis,
    _coerce_contradiction,
    _coerce_premise,
    _median_axes,
    check_continuity,
    check_premise_fidelity,
    check_rubric,
)

_CHAPTERS = {"Chapter 1": "prose one", "Chapter 2": "prose two"}


# --- check_continuity -------------------------------------------------------

def test_continuity_hard_contradiction_fails():
    findings = check_continuity(
        _CHAPTERS, "cast",
        contradictions=[Contradiction("death", "hard", "Soren dies in ch3, speaks in ch5")],
    )
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "Soren" in fails[0].message


def test_continuity_minor_contradiction_warns():
    findings = check_continuity(
        _CHAPTERS, "cast",
        contradictions=[Contradiction("trait", "minor", "eye colour wobble")],
    )
    assert [f.severity for f in findings] == ["warn"]


def test_continuity_clean_reports_info():
    findings = check_continuity(_CHAPTERS, "cast", contradictions=[])
    assert [f.severity for f in findings] == ["info"]


def test_continuity_empty_chapters_no_findings():
    assert check_continuity({}, "cast", contradictions=[]) == []


# --- check_premise_fidelity -------------------------------------------------

def test_premise_dramatized_passes():
    findings = check_premise_fidelity(
        "idea", _CHAPTERS, verdict=PremiseVerdict(dramatizes=True, score=5, note="solid")
    )
    assert [f.severity for f in findings] == ["info"]


def test_premise_off_idea_fails():
    findings = check_premise_fidelity(
        "idea", _CHAPTERS, verdict=PremiseVerdict(dramatizes=False, score=2, note="generic")
    )
    assert [f.severity for f in findings] == ["fail"]


def test_premise_low_score_even_if_dramatizes_fails():
    findings = check_premise_fidelity(
        "idea", _CHAPTERS, verdict=PremiseVerdict(dramatizes=True, score=2)
    )
    assert [f.severity for f in findings] == ["fail"]


# --- check_rubric -----------------------------------------------------------

def _scores(value: int) -> list[AxisScore]:
    return [AxisScore(axis, value) for axis in RUBRIC_AXES]


def test_rubric_all_high_passes():
    findings = check_rubric(_CHAPTERS, scores=_scores(5))
    assert [f.severity for f in findings] == ["info"]


def test_rubric_low_average_fails():
    findings = check_rubric(_CHAPTERS, scores=_scores(3))  # avg 3.0 < 4.0
    assert [f.severity for f in findings] == ["fail"]


def test_rubric_one_axis_below_floor_fails_even_with_high_average():
    scores = [AxisScore(a, 5) for a in RUBRIC_AXES if a != "ending"] + [AxisScore("ending", 2)]
    findings = check_rubric(_CHAPTERS, scores=scores)
    assert findings[0].severity == "fail"
    assert "ending" in findings[0].message


def test_rubric_empty_scores_warns():
    findings = check_rubric(_CHAPTERS, scores=[])
    assert [f.severity for f in findings] == ["warn"]


# --- coercion ---------------------------------------------------------------

def test_coerce_contradiction_from_dict_and_unknown_severity():
    c = _coerce_contradiction({"kind": "timeline", "severity": "weird", "detail": "x"})
    assert c.severity == "hard"  # unknown -> surfaced as hard
    assert _coerce_contradiction({"detail": ""}) is None  # empty detail dropped


def test_coerce_axis_normalises_and_clamps():
    assert _coerce_axis({"axis": "Arc-Structure", "score": 9}).axis == "arc_structure"
    assert _coerce_axis({"axis": "arc structure", "score": 9}).score == 5  # clamped
    assert _coerce_axis({"axis": "nonsense", "score": 3}) is None
    assert _coerce_axis({"axis": "ending", "score": "x"}) is None


def test_coerce_premise_defaults_bad_score_to_zero():
    v = _coerce_premise({"dramatizes": True, "score": "nope"})
    assert v.dramatizes is True
    assert v.score == 0


# --- median aggregation -----------------------------------------------------

def test_median_axes_takes_per_axis_median():
    runs = [_scores(2), _scores(4), _scores(5)]  # median per axis = 4
    merged = _median_axes(runs)
    assert {a.axis for a in merged} == set(RUBRIC_AXES)
    assert all(a.score == 4 for a in merged)
