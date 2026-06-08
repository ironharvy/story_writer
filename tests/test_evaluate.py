"""Tests for the unified scorecard (bench/evaluate.py) — no LLM required.

The Tier-2/3 gates (POV, prose lint, continuity, premise, rubric) are exercised
by *injecting* precomputed results via JudgeInputs, so these tests stay
deterministic while covering the wiring, ship/complete logic, and the
Definition-of-Done map.
"""
from __future__ import annotations

import json

from bench.evaluate import JudgeInputs, evaluate
from bench.judge import RUBRIC_AXES, AxisScore, Contradiction, PremiseVerdict
from pov_check import ChapterPOV
from qa import split_chapters
from story_linter import Replacement


def _prose(word: str, n: int) -> str:
    return " ".join([word] * n) + "."


def _story(chapters: dict[str, str], characters_block: str = "") -> str:
    parts = [
        "# Story", "", "## World bible", "", "### Characters", "",
        characters_block, "", "## Final Story", "",
    ]
    for title, body in chapters.items():
        parts += [f"### {title}", "", body, ""]
    return "\n".join(parts)


def _clean_story() -> str:
    return _story({
        "Chapter 1: Open": _prose("alpha", 400),
        "Chapter 2: Turn": _prose("bravo", 400),
    })


def _good_rubric() -> list[AxisScore]:
    return [AxisScore(axis, 5) for axis in RUBRIC_AXES]


def _clean_judge(story: str, **overrides) -> JudgeInputs:
    """JudgeInputs with every Tier-2/3 result injected clean; override per test."""
    base = dict(
        pov_classifications=[ChapterPOV(t, "third_person") for t in split_chapters(story)],
        linter_replacements=[],
        contradictions=[],
        premise_verdict=PremiseVerdict(dramatizes=True, score=5, note="on premise"),
        rubric_scores=_good_rubric(),
    )
    base.update(overrides)
    return JudgeInputs(**base)


# --- deterministic-only path ------------------------------------------------

def test_deterministic_only_is_tier1_clean_but_not_shippable():
    card = evaluate(_clean_story())
    assert card.tier1_clean is True
    assert card.complete is False  # every judge skipped
    assert card.ship is False
    assert set(card.not_evaluated) == {
        "pov_consistency", "prose_linter", "continuity", "premise_fidelity", "rubric",
    }


def test_dod_marks_unrun_judges_as_skipped_and_ending_as_manual():
    card = evaluate(_clean_story())
    by_item = {d["item"]: d["status"] for d in card.definition_of_done}
    assert by_item["POV consistent across chapters"] == "skipped"
    assert by_item["Story dramatizes the premise"] == "skipped"
    assert by_item["Real ending dramatizes the spine's final beats"] == "manual"
    assert card.manual_checks  # the real-ending check is surfaced when rubric is off


# --- full path (judges injected) --------------------------------------------

def test_full_clean_run_ships():
    story = _clean_story()
    card = evaluate(story, _clean_judge(story))
    assert card.complete is True
    assert card.ship is True
    assert card.fails == []
    by_item = {d["item"]: d["status"] for d in card.definition_of_done}
    assert by_item["POV consistent across chapters"] == "pass"
    assert by_item["Story dramatizes the premise"] == "pass"
    # When the rubric runs, "real ending" is judged by it, not manual.
    assert by_item["Real ending dramatizes the spine's final beats"] == "pass"
    assert card.manual_checks == []


def test_pov_outlier_blocks_ship():
    story = _clean_story()
    judge = _clean_judge(story, pov_classifications=[
        ChapterPOV("Chapter 1: Open", "third_person"),
        ChapterPOV("Chapter 2: Turn", "first_person", "narrator says 'I'"),
    ])
    card = evaluate(story, judge)
    assert card.ship is False
    assert next(g for g in card.gates if g.gate == "pov_consistency").status == "fail"


def test_linter_findings_are_advisory_not_blocking():
    story = _clean_story()
    judge = _clean_judge(
        story, linter_replacements=[Replacement("the child", "Cinder", "placeholder")]
    )
    card = evaluate(story, judge)
    assert card.complete is True
    assert card.ship is True  # a proposed lint is a warn, not a fail
    linter_gate = next(g for g in card.gates if g.gate == "prose_linter")
    assert linter_gate.status == "warn"
    assert linter_gate.messages


# --- Tier-2/3 judge gates ---------------------------------------------------

def test_hard_contradiction_blocks_ship():
    story = _clean_story()
    judge = _clean_judge(story, contradictions=[
        Contradiction("name_collision", "hard", "two distinct characters named Elara"),
    ])
    card = evaluate(story, judge)
    assert card.ship is False
    assert next(g for g in card.gates if g.gate == "continuity").status == "fail"


def test_minor_contradiction_warns_but_ships():
    story = _clean_story()
    judge = _clean_judge(story, contradictions=[
        Contradiction("trait", "minor", "cloak is grey in ch1, brown in ch4"),
    ])
    card = evaluate(story, judge)
    assert card.ship is True
    assert next(g for g in card.gates if g.gate == "continuity").status == "warn"


def test_off_premise_blocks_ship():
    story = _clean_story()
    judge = _clean_judge(
        story, premise_verdict=PremiseVerdict(dramatizes=False, score=2, note="generic tale")
    )
    card = evaluate(story, judge)
    assert card.ship is False
    assert next(g for g in card.gates if g.gate == "premise_fidelity").status == "fail"


def test_weak_rubric_blocks_ship_and_fails_ending_dod():
    story = _clean_story()
    weak = [AxisScore(a, 4) for a in RUBRIC_AXES if a != "ending"] + [AxisScore("ending", 2)]
    card = evaluate(story, _clean_judge(story, rubric_scores=weak))
    assert card.ship is False
    assert next(g for g in card.gates if g.gate == "rubric").status == "fail"
    by_item = {d["item"]: d["status"] for d in card.definition_of_done}
    assert by_item["Real ending dramatizes the spine's final beats"] == "fail"


def test_premise_skipped_when_no_idea_and_no_injection():
    # run_llm on, but no idea and no injected verdict -> premise can't be judged.
    story = _clean_story()
    judge = JudgeInputs(run_llm=False, premise_verdict=None)
    card = evaluate(story, judge)
    premise_gate = next(g for g in card.gates if g.gate == "premise_fidelity")
    assert premise_gate.status == "skipped"


# --- Tier-1 failures still gate ---------------------------------------------

def test_empty_chapter_fails_and_blocks_ship():
    story = _story({"Chapter 1: Open": _prose("alpha", 400), "Chapter 2: Gap": ""})
    card = evaluate(story, _clean_judge(story))
    assert card.ship is False
    assert card.tier1_clean is False
    assert next(g for g in card.gates if g.gate == "chapter_length").status == "fail"


def test_missing_section_fails_structure():
    story = "\n".join([
        "# Story", "", "## World bible", "", "### Characters", "",
        "1. Cinder, the protagonist",
    ])
    card = evaluate(story)
    assert next(g for g in card.gates if g.gate == "structure").status == "fail"
    assert card.tier1_clean is False


def test_protagonist_placeholder_fails():
    story = _story({"Chapter 1: Open": "The protagonist ran. " + _prose("alpha", 400)})
    card = evaluate(story, _clean_judge(story))
    assert next(g for g in card.gates if g.gate == "placeholder_protagonist").status == "fail"
    assert card.ship is False


def test_phrase_reuse_over_budget_blocks_ship():
    distinct = " ".join(
        ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]
    )
    body = (distinct + " ") * 25
    story = _story({"Chapter 1: Open": body, "Chapter 2: Echo": body})
    card = evaluate(story, _clean_judge(story))
    assert card.budgets["phrase_reuse"]["over"] is True
    assert card.ship is False


# --- serialization ----------------------------------------------------------

def test_scorecard_json_roundtrips():
    card = evaluate(_clean_story())
    parsed = json.loads(card.to_json())
    assert parsed["ship"] is False
    assert parsed["tier1_clean"] is True
    assert "definition_of_done" in parsed
    assert parsed["gates"][0]["gate"] == "structure"
