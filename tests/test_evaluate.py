"""Tests for the unified scorecard (bench/evaluate.py) — no LLM required.

The Tier-2 gates (POV consistency, prose lint) are exercised by *injecting*
precomputed results, so these tests stay deterministic and fast while still
covering the wiring, the ship/complete logic, and the Definition-of-Done map.
"""
from __future__ import annotations

import json

from bench.evaluate import evaluate
from pov_check import ChapterPOV
from story_linter import Replacement


def _prose(word: str, n: int) -> str:
    return " ".join([word] * n) + "."


def _story(chapters: dict[str, str], characters_block: str = "") -> str:
    parts = [
        "# Story",
        "",
        "## World bible",
        "",
        "### Characters",
        "",
        characters_block,
        "",
        "## Final Story",
        "",
    ]
    for title, body in chapters.items():
        parts += [f"### {title}", "", body, ""]
    return "\n".join(parts)


def _clean_story() -> str:
    # Two distinct in-band chapters → no length, band, reuse, or placeholder hits.
    return _story({
        "Chapter 1: Open": _prose("alpha", 400),
        "Chapter 2: Turn": _prose("bravo", 400),
    })


def _third_person(story: str) -> list[ChapterPOV]:
    from qa import split_chapters

    return [ChapterPOV(title, "third_person") for title in split_chapters(story)]


# --- deterministic-only path ------------------------------------------------

def test_deterministic_only_is_tier1_clean_but_not_shippable():
    card = evaluate(_clean_story())
    assert card.tier1_clean is True
    assert card.complete is False  # Tier-2 not run
    assert card.ship is False
    assert set(card.not_evaluated) == {"pov_consistency", "prose_linter"}


def test_dod_marks_unrun_pov_as_skipped_and_ending_as_manual():
    card = evaluate(_clean_story())
    by_item = {d["item"]: d["status"] for d in card.definition_of_done}
    assert by_item["POV consistent across chapters"] == "skipped"
    assert by_item["Real ending dramatizes the spine's final beats"] == "manual"
    assert card.manual_checks  # the real-ending check is always surfaced


# --- full path (Tier-2 injected) --------------------------------------------

def test_full_clean_run_ships():
    story = _clean_story()
    card = evaluate(
        story,
        pov_classifications=_third_person(story),
        linter_replacements=[],
    )
    assert card.complete is True
    assert card.ship is True
    assert card.fails == []
    by_item = {d["item"]: d["status"] for d in card.definition_of_done}
    assert by_item["POV consistent across chapters"] == "pass"


def test_pov_outlier_blocks_ship():
    story = _clean_story()
    classifications = [
        ChapterPOV("Chapter 1: Open", "third_person"),
        ChapterPOV("Chapter 2: Turn", "first_person", "narrator says 'I'"),
    ]
    card = evaluate(story, pov_classifications=classifications, linter_replacements=[])
    assert card.ship is False
    pov_gate = next(g for g in card.gates if g.gate == "pov_consistency")
    assert pov_gate.status == "fail"


def test_linter_findings_are_advisory_not_blocking():
    story = _clean_story()
    card = evaluate(
        story,
        pov_classifications=_third_person(story),
        linter_replacements=[Replacement("the child", "Cinder", "placeholder")],
    )
    assert card.complete is True
    assert card.ship is True  # a proposed lint is a warn, not a fail
    linter_gate = next(g for g in card.gates if g.gate == "prose_linter")
    assert linter_gate.status == "warn"
    assert linter_gate.messages


# --- Tier-1 failures --------------------------------------------------------

def test_empty_chapter_fails_and_blocks_ship():
    story = _story({
        "Chapter 1: Open": _prose("alpha", 400),
        "Chapter 2: Gap": "",
    })
    card = evaluate(story, pov_classifications=_third_person(story), linter_replacements=[])
    assert card.ship is False
    assert card.tier1_clean is False
    length_gate = next(g for g in card.gates if g.gate == "chapter_length")
    assert length_gate.status == "fail"
    by_item = {d["item"]: d["status"] for d in card.definition_of_done}
    assert by_item["N non-empty chapters (>= hard floor)"] == "fail"


def test_missing_section_fails_structure():
    story = "\n".join([
        "# Story", "", "## World bible", "", "### Characters", "",
        "1. Cinder, the protagonist",
    ])
    card = evaluate(story)
    structure_gate = next(g for g in card.gates if g.gate == "structure")
    assert structure_gate.status == "fail"
    assert card.tier1_clean is False


def test_protagonist_placeholder_fails():
    story = _story({"Chapter 1: Open": "The protagonist ran. " + _prose("alpha", 400)})
    card = evaluate(story, pov_classifications=_third_person(story), linter_replacements=[])
    gate = next(g for g in card.gates if g.gate == "placeholder_protagonist")
    assert gate.status == "fail"
    assert card.ship is False


def test_phrase_reuse_over_budget_blocks_ship():
    # Two identical in-band chapters share many 5-grams → reuse budget blown.
    distinct = " ".join(
        ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
         "hotel", "india", "juliet", "kilo", "lima", "mike", "november"]
    )
    body = (distinct + " ") * 25  # ~350 words, in band
    story = _story({"Chapter 1: Open": body, "Chapter 2: Echo": body})
    card = evaluate(story, pov_classifications=_third_person(story), linter_replacements=[])
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
