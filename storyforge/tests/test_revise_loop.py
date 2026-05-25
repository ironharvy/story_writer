"""Unit tests for the self-evaluate -> revise loop helpers (no LLM, no network).

These exercise the pure functions that turn a scorecard into editorial notes,
locate the climax chapters, and decide whether a revise round improved things.
"""
from __future__ import annotations

import json
import types

from storyforge import cli
from storyforge.evaluate import harness
from storyforge.models import ChapterPlan


def _scorecard() -> dict:
    return {
        "checks": [
            {"id": "T2.3", "name": "Continuity", "severity": "fail",
             "details": {"contradictions": [
                 {"severity": "hard", "detail": "Maris dies in ch3 but speaks in ch5",
                  "chapters": "3,5"},
                 {"severity": "minor", "detail": "a small wobble"},
             ]}},
            {"id": "T1.6", "name": "Duplicate scene", "severity": "warn",
             "details": {"pairs": [{"a": "Chapter 4", "b": "Chapter 5", "jaccard": 0.41}]}},
            {"id": "T1.7", "name": "Ending completeness", "severity": "fail", "details": {}},
        ],
        "rubric": {
            "scores": {"arc_structure": 2, "ending": 4, "character_agency": 2,
                       "scene_vs_summary": 4, "prose_quality": 4, "cohesion": 3},
            "justifications": {"arc_structure": "muddled", "character_agency": "passive hero"},
            "average": 3.17, "threshold_avg": 4.0,
        },
        "ship_blockers": ["T1.4 phrase-reuse 9 > 5"],
    }


def test_derive_notes_covers_each_signal():
    notes = cli._derive_notes(_scorecard())
    assert any("continuity contradiction" in n and "do not re-stage" in n for n in notes)
    assert any("stage the same scene" in n for n in notes)            # T1.6 merge note
    assert any("complete" in n and "final chapter" in n for n in notes)  # T1.7
    assert any("arc_structure (2/5)" in n for n in notes)
    assert any("character_agency (2/5)" in n for n in notes)
    assert any("rubric average 3.17" in n for n in notes)             # umbrella
    assert any("phrase reuse" in n.lower() for n in notes)            # ship_blocker
    assert len(notes) <= 8


def test_derive_notes_skips_minor_contradiction_as_hard():
    notes = cli._derive_notes(_scorecard())
    # the minor wobble must not be promoted into a hard-contradiction note
    assert not any("a small wobble" in n for n in notes)


def test_derive_notes_empty_when_clean():
    clean = {"checks": [], "rubric": {"scores": {}, "average": 0.0}, "ship_blockers": []}
    assert cli._derive_notes(clean) == []


def test_climax_numbers_from_structural_role():
    spine = [ChapterPlan(number=1, structural_role="setup"),
             ChapterPlan(number=2, structural_role="rising"),
             ChapterPlan(number=3, structural_role="climax"),
             ChapterPlan(number=4, structural_role="resolution")]
    assert cli._climax_numbers(spine) == {3, 4}


def test_climax_numbers_falls_back_to_last_chapter():
    spine = [ChapterPlan(number=1), ChapterPlan(number=2), ChapterPlan(number=3)]
    assert cli._climax_numbers(spine) == {3}
    assert cli._climax_numbers([]) == set()


def test_scorecard_key_orders_by_quality():
    worse = {"ship": False, "ship_blockers": ["a", "b"],
             "checks": [{"severity": "fail"}], "rubric": {"average": 2.3}}
    better = {"ship": False, "ship_blockers": ["a"], "checks": [], "rubric": {"average": 3.5}}
    shipped = {"ship": True, "ship_blockers": [], "checks": [], "rubric": {"average": 4.2}}
    assert cli._scorecard_key(better) > cli._scorecard_key(worse)
    assert cli._scorecard_key(shipped) > cli._scorecard_key(better)
    # regression guard: a worse re-eval is NOT an improvement over the best
    assert not (cli._scorecard_key(worse) > cli._scorecard_key(better))


def test_low_scene_axes_detects_below_floor():
    assert cli._low_scene_axes({"rubric": {"scores": {"scene_vs_summary": 2}}})
    assert cli._low_scene_axes({"rubric": {"scores": {"character_agency": 1}}})
    assert not cli._low_scene_axes(
        {"rubric": {"scores": {"scene_vs_summary": 4, "character_agency": 3}}})
    assert not cli._low_scene_axes({"rubric": {"scores": {}}})


# --- loop orchestration (LLM + judge mocked) ----------------------------------

def _args(**kw):
    base = dict(judge=None, no_revise=False, revise_rounds=2, tier1_only=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _setup_run(tmp_path):
    ms_path = tmp_path / "manuscript.md"
    ms_path.write_text("original manuscript")
    (tmp_path / "state.json").write_text(json.dumps({
        "idea": "x", "spec": {},
        "spine": [{"number": 1, "title": "C1", "structural_role": "setup"},
                  {"number": 2, "title": "C2", "structural_role": "climax"}],
        "chapters": [],
    }))
    return ms_path


def _mock_judge(monkeypatch, scorecards):
    seq = iter(scorecards)
    monkeypatch.setattr(harness, "evaluate_file", lambda *a, **k: next(seq))
    written = {}
    monkeypatch.setattr(harness, "write_scorecard",
                        lambda sc, p: written.update(sc=sc, path=p))
    monkeypatch.setattr(harness, "summarize", lambda sc: "summary")
    return written


def test_revise_loop_polishes_then_ships(tmp_path, monkeypatch):
    ms_path = _setup_run(tmp_path)
    sc_bad = {"ship": False, "ship_blockers": ["T3 FAIL"], "checks": [],
              "rubric": {"scores": {"arc_structure": 2}, "average": 2.5, "threshold_avg": 4.0}}
    sc_good = {"ship": True, "ship_blockers": [], "checks": [],
               "rubric": {"scores": {"arc_structure": 4}, "average": 4.3}}
    sc_final = {"ship": True, "ship_blockers": [], "checks": [], "rubric": {"average": 4.1}}
    written = _mock_judge(monkeypatch, [sc_bad, sc_good, sc_final])

    class FakePipe:
        scene_level = False

        def __init__(self):
            self.polish_calls = 0

        def polish(self, state, notes, climax):
            self.polish_calls += 1
            assert notes  # the loop must derive concrete notes from the bad scorecard
            assert climax == {2}  # from structural_role
            ms_path.write_text("improved manuscript")

    pipe = FakePipe()
    cli._revise_loop(_args(), pipe, tmp_path, ms_path, "x", "x", None)
    assert pipe.polish_calls == 1
    assert written["sc"] is sc_final          # final scorecard is the recorded one
    assert ms_path.read_text() == "improved manuscript"  # best draft kept


def test_revise_loop_restores_best_on_regression(tmp_path, monkeypatch):
    ms_path = _setup_run(tmp_path)
    sc_bad = {"ship": False, "ship_blockers": ["T3 FAIL"], "checks": [],
              "rubric": {"scores": {"arc_structure": 2}, "average": 2.5, "threshold_avg": 4.0}}
    sc_worse = {"ship": False, "ship_blockers": ["T3 FAIL", "T2.3 FAIL"], "checks": [],
                "rubric": {"scores": {"arc_structure": 1}, "average": 1.8, "threshold_avg": 4.0}}
    sc_final = {"ship": False, "ship_blockers": ["T3 FAIL"], "checks": [],
                "rubric": {"average": 2.5}}
    _mock_judge(monkeypatch, [sc_bad, sc_worse, sc_final])

    class FakePipe:
        scene_level = False

        def __init__(self):
            self.polish_calls = 0

        def polish(self, state, notes, climax):
            self.polish_calls += 1
            ms_path.write_text("worse manuscript")

    pipe = FakePipe()
    cli._revise_loop(_args(revise_rounds=2), pipe, tmp_path, ms_path, "x", "x", None)
    # only one polish attempt: it regressed, so the loop stopped and restored the best
    assert pipe.polish_calls == 1
    assert ms_path.read_text() == "original manuscript"


def test_revise_loop_no_revise_skips_polish(tmp_path, monkeypatch):
    ms_path = _setup_run(tmp_path)
    sc0 = {"ship": False, "ship_blockers": ["T3 FAIL"], "checks": [],
           "rubric": {"scores": {"arc_structure": 2}, "average": 2.5, "threshold_avg": 4.0}}
    sc_final = {"ship": False, "ship_blockers": ["T3 FAIL"], "checks": [], "rubric": {}}
    _mock_judge(monkeypatch, [sc0, sc_final])

    class FakePipe:
        scene_level = False

        def polish(self, *a, **k):
            raise AssertionError("polish must not run with --no-revise")

    cli._revise_loop(_args(no_revise=True), FakePipe(), tmp_path, ms_path, "x", "x", None)
    assert ms_path.read_text() == "original manuscript"
