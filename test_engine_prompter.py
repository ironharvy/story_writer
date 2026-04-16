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


def test_answer_questions_pops_batches_in_order():
    prompter = ScriptedPrompter(
        idea="x",
        answer_batches=[["iA", None, "iC"], ["wA"]],
    )

    first = prompter.answer_questions(_questions(3))
    assert [qa.user_answer for qa in first] == ["iA", None, "iC"]
    # Accepting the proposed answer is represented by user_answer=None;
    # effective_answer should fall back to the proposed value.
    assert first[1].effective_answer == "p1"

    second = prompter.answer_questions(_questions(1))
    assert [qa.user_answer for qa in second] == ["wA"]


def test_answer_questions_raises_when_batch_too_short():
    prompter = ScriptedPrompter(idea="x", answer_batches=[["only one"]])
    with pytest.raises(RuntimeError, match="batch has 1 answers"):
        prompter.answer_questions(_questions(2))


def test_answer_questions_raises_when_batches_exhausted():
    prompter = ScriptedPrompter(
        idea="x",
        answer_batches=[["a"], ["b"]],
    )
    prompter.answer_questions(_questions(1))
    prompter.answer_questions(_questions(1))
    with pytest.raises(RuntimeError, match="more times than"):
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
