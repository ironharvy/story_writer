"""F1 — reviewer injection / UI decoupling.

Every interactive ``run_*`` stage now takes an optional ``reviewer`` callback.
The orchestrator injects ``ui.review_answer``; passing ``None`` (as benchmarks
and these tests do) returns the first draft without a review loop, so the
stages run with no TTY and no ``ui`` import. These tests use ``DummyLM`` so no
network or model is required.
"""
from __future__ import annotations

import dspy
from dspy.utils import DummyLM

import core.foundation as foundation
import story
import world_state
from core.types import WorldBible

SPINE = "s1\ns2\ns3\ns4\ns5\ns6\ns7"


def _configure(*responses: dict) -> None:
    dspy.configure(lm=DummyLM(list(responses)))


def _reject_then_accept(calls: list[str]):
    """A reviewer that rejects the first draft, then accepts the second."""

    def reviewer(label: str, draft: str) -> tuple[str, bool]:
        calls.append(label)
        return ("revise it", False) if len(calls) == 1 else (draft, True)

    return reviewer


def _world_bible() -> WorldBible:
    return WorldBible(
        story_title="T",
        rules_of_the_world=["r1"],
        characters=["c1"],
        locations=["l1"],
        timeline=["e1"],
    )


# --- reviewer=None returns the first draft (the headless contract) ----------


def test_core_premise_none_returns_first_draft():
    _configure(
        {"reasoning": "a", "core_premise": "FIRST"},
        {"reasoning": "b", "core_premise": "SECOND"},
    )
    assert foundation.run_generate_core_premise("idea", reviewer=None) == "FIRST"


def test_spine_none_returns_first_draft():
    def spine(tag: str) -> dict:
        return {
            "reasoning": "r",
            "once_upon_a_time": f"once-{tag}",
            "every_day": f"every-{tag}",
            "until_one_day": f"until-{tag}",
            "because_of_that": f"because-{tag}",
            "and_because_of_that": f"andbecause-{tag}",
            "until_finally": f"finally-{tag}",
            "ever_since_that_day": f"since-{tag}",
        }

    _configure(spine("A"), spine("B"))
    out = foundation.run_generate_spine("idea", "premise", reviewer=None)
    assert "once-A" in out and "since-A" in out
    assert "-B" not in out  # only the first draft was returned


def test_clarify_idea_none_uses_proposed_answers():
    _configure(
        {"reasoning": "r", "questions": ["Q1?", "Q2?"],
         "proposed_answers": ["A1", "A2"]},
        {"reasoning": "r", "updated_idea": "UPDATED"},
    )
    updated, title = foundation.run_clarify_idea("idea", "Title", reviewer=None)
    assert updated == "UPDATED"
    assert title == "Title"  # provided title => no extra title-generation call


def test_enhance_chapter_none_returns_first_draft(monkeypatch):
    # Force the optional random-detail branch off so exactly one draft call runs.
    monkeypatch.setattr(story.random, "random", lambda: 0.99)
    _configure(
        {"reasoning": "r", "prose": "PROSE_ONE"},
        {"reasoning": "r", "prose": "PROSE_TWO"},
    )
    out = story.run_enhance_chapter(
        "Ch1\nbeats", "idea", "T", SPINE, _world_bible(), "",
        chapter_index=1, total_chapters=3, reviewer=None,
    )
    assert out == "PROSE_ONE"


def test_init_world_state_none_returns_first_draft():
    base = {
        "story_clock": "Opening", "characters": ["Mara: home"],
        "locations": ["cottage: intact"], "plot_threads": ["who?"],
        "key_objects": ["letter: pocket"], "recent_events": [],
    }
    _configure(
        {"reasoning": "r", "world_state": base},
        {"reasoning": "r", "world_state": {**base, "story_clock": "Later"}},
    )
    out = world_state.run_init_world_state("idea", "T", SPINE, _world_bible(), reviewer=None)
    assert out.story_clock == "Opening"


# --- a real reviewer still loops on feedback until accepted ------------------


def test_core_premise_reviewer_loops_until_accepted():
    _configure(
        {"reasoning": "a", "core_premise": "FIRST"},
        {"reasoning": "b", "core_premise": "SECOND"},
    )
    calls: list[str] = []
    out = foundation.run_generate_core_premise("idea", reviewer=_reject_then_accept(calls))
    assert out == "SECOND"
    assert len(calls) == 2  # rejected once, accepted on the second draft


def test_draft_chapter_with_state_reviewer_loops(monkeypatch):
    _configure(
        {"reasoning": "r", "prose": "DRAFT_ONE"},
        {"reasoning": "r", "prose": "DRAFT_TWO"},
    )
    from core.types import WorldState

    state = WorldState(
        story_clock="Opening", characters=["Mara"], locations=["cottage"],
        plot_threads=["who?"], key_objects=["letter"],
    )
    calls: list[str] = []
    out = world_state.run_draft_chapter_with_state(
        "Ch1\nbeats", "idea", "T", SPINE, _world_bible(), state,
        chapter_index=1, total_chapters=3, reviewer=_reject_then_accept(calls),
    )
    assert out == "DRAFT_TWO"
    assert len(calls) == 2
