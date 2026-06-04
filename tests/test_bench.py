"""Tests for the benchmark harness machinery (no LLM required).

bench.run is mostly LLM-driven and lives outside this scope; the parts
worth locking in here are the fixture loader, the generator selector,
and the result aggregation in bench.score.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import generators  # noqa: F401 — populates REGISTRY for select_generators()
from bench.criteria import Scorecard
from bench.run import Fixture, load_fixtures, select_generators
from bench.score import render_fixture_table, walk_run_root


def test_load_fixtures_returns_all_three():
    fixtures = load_fixtures()
    ids = [f.id for f in fixtures]
    assert "01_thriller" in ids
    assert "02_ensemble" in ids
    assert "03_long_arc" in ids


def test_load_fixtures_filters_by_id():
    fixtures = load_fixtures(["01_thriller"])
    assert len(fixtures) == 1
    assert fixtures[0].id == "01_thriller"


def test_load_fixtures_unknown_id_exits():
    with pytest.raises(SystemExit, match="unknown fixture id"):
        load_fixtures(["does_not_exist"])


def test_fixture_carries_required_fields():
    fixtures = load_fixtures()
    for f in fixtures:
        assert isinstance(f, Fixture)
        assert f.title and f.idea and f.niche
        assert f.number_of_chapters >= 3


def test_select_generators_default_is_all_promoted():
    entries = select_generators(None)
    ids = {e.id for e in entries}
    assert {"baseline", "world_state", "dspy_module"}.issubset(ids)
    assert all(e.status == "promoted" for e in entries)


def test_select_generators_filters_by_id():
    entries = select_generators(["baseline"])
    assert len(entries) == 1
    assert entries[0].id == "baseline"


def test_walk_run_root_discovers_layout(tmp_path: Path):
    # Fabricate the bench/run.py output layout.
    for fixture in ("01_thriller", "02_ensemble"):
        for strategy in ("baseline", "world_state"):
            d = tmp_path / fixture / strategy
            d.mkdir(parents=True)
            (d / "story.md").write_text("# stub", encoding="utf-8")
    by_fixture = walk_run_root(tmp_path)
    assert set(by_fixture) == {"01_thriller", "02_ensemble"}
    assert set(by_fixture["01_thriller"]) == {"baseline", "world_state"}


def test_render_fixture_table_includes_strategies_and_ship_marker():
    sc_ship = Scorecard(ship=True, fails=[], warns=[], budgets={
        "name_drift": {"count": 0, "budget": 2, "over": False},
        "phrase_reuse": {"count": 1, "budget": 5, "over": False},
    })
    sc_fail = Scorecard(ship=False, fails=[{"check": "chapter_length",
                                            "severity": "fail",
                                            "message": "Chapter 2 is 120 words"}],
                        warns=[], budgets={
        "name_drift": {"count": 3, "budget": 2, "over": True},
        "phrase_reuse": {"count": 8, "budget": 5, "over": True},
    })
    table = render_fixture_table("01_thriller", {
        "baseline": sc_ship,
        "world_state": sc_fail,
    })
    assert "01_thriller" in table
    assert "baseline" in table
    assert "world_state" in table
    assert "✓" in table  # baseline ships
    assert "✗" in table  # world_state doesn't


def test_scorecard_to_json_roundtrips():
    sc = Scorecard(ship=False,
                   fails=[{"check": "x", "severity": "fail", "message": "y"}],
                   warns=[],
                   budgets={"a": {"count": 1, "budget": 5, "over": False}})
    parsed = json.loads(sc.to_json())
    assert parsed["ship"] is False
    assert parsed["fails"][0]["check"] == "x"
    assert parsed["budgets"]["a"]["count"] == 1
