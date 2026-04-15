"""Unit tests for the :class:`ScriptedPrompter` contract."""

import pytest

from engine.prompter_scripted import PremiseDecision, ScriptedPrompter
from engine.types import QAPair


def _questions(n: int) -> list[QAPair]:
    return [QAPair(f"q{i}", f"p{i}") for i in range(n)]


def test_ask_idea_returns_provided_idea():
    assert ScriptedPrompter(idea="a wizard").ask_idea() == "a wizard"


def test_ask_idea_without_idea_raises():
    with pytest.raises(RuntimeError, match="no idea"):
        ScriptedPrompter().ask_idea()


def test_answer_questions_routes_first_call_to_ideation_and_second_to_world_bible():
    prompter = ScriptedPrompter(
        idea="x",
        ideation_answers=["iA", None, "iC"],
        world_bible_answers=["wA"],
    )

    ideation = prompter.answer_questions(_questions(3))
    assert [qa.user_answer for qa in ideation] == ["iA", None, "iC"]
    # Accepting the proposed answer is represented by user_answer=None;
    # effective_answer should fall back to the proposed value.
    assert ideation[1].effective_answer == "p1"

    wb = prompter.answer_questions(_questions(1))
    assert [qa.user_answer for qa in wb] == ["wA"]


def test_answer_questions_raises_if_script_exhausted():
    prompter = ScriptedPrompter(idea="x", ideation_answers=["only one"])
    with pytest.raises(RuntimeError, match="ideation answers"):
        prompter.answer_questions(_questions(2))


def test_answer_questions_raises_on_third_call():
    prompter = ScriptedPrompter(
        idea="x",
        ideation_answers=["a"],
        world_bible_answers=["b"],
    )
    prompter.answer_questions(_questions(1))
    prompter.answer_questions(_questions(1))
    with pytest.raises(RuntimeError, match="more than twice"):
        prompter.answer_questions(_questions(1))


def test_confirm_premise_walks_decisions_in_order():
    prompter = ScriptedPrompter(
        idea="x",
        premise_decisions=[
            PremiseDecision(accept=False, refinement_details="more noir"),
            PremiseDecision(accept=True),
        ],
    )
    accept, details = prompter.confirm_premise("v1")
    assert not accept and details == "more noir"

    accept, details = prompter.confirm_premise("v2")
    assert accept and details == ""


def test_confirm_premise_raises_when_exhausted():
    prompter = ScriptedPrompter(idea="x", premise_decisions=[])
    with pytest.raises(RuntimeError, match="exhausted"):
        prompter.confirm_premise("v1")


def test_notify_forwards_to_sink_when_provided():
    received: list[tuple[str, str]] = []
    prompter = ScriptedPrompter(
        idea="x",
        notify_sink=lambda level, msg: received.append((level, msg)),
    )
    prompter.notify("info", "hello")
    prompter.notify("error", "boom")
    assert received == [("info", "hello"), ("error", "boom")]


def test_wait_continue_is_noop():
    # Must not raise and must not block.
    ScriptedPrompter(idea="x").wait_continue("press enter")
