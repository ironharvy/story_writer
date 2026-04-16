"""End-to-end headless pipeline tests.

Exercises :func:`engine.pipeline.run_all` with ``FakeGen`` stand-ins for the
DSPy modules and a :class:`ScriptedPrompter` — no LLM calls, no rich/TTY.
This is the contract the future FastAPI worker will rely on.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from engine import PipelineOptions, StageName, StoryState, run_all
from engine.pipeline import default_stage_order
from engine.prompter_scripted import PremiseDecision, ScriptedPrompter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeGen:
    """Canned-response stand-in for a dspy.Module.

    Records every call for assertions.
    """

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


def _qa(question: str, proposed: str) -> SimpleNamespace:
    return SimpleNamespace(question=question, proposed_answer=proposed)


def _build_generators() -> dict[str, FakeGen]:
    return {
        "QuestionGenerator": FakeGen(
            SimpleNamespace(
                questions_with_answers=[
                    _qa("Who is the protagonist?", "A young wizard"),
                    _qa("Where?", "In a floating city"),
                ]
            )
        ),
        "CorePremiseGenerator": FakeGen(
            SimpleNamespace(core_premise="A young wizard defends a floating city.")
        ),
        "SpineTemplateGenerator": FakeGen(
            SimpleNamespace(spine_template="Once upon a time...")
        ),
        "WorldBibleQuestionGenerator": FakeGen(
            SimpleNamespace(
                questions_with_answers=[_qa("What is the magic system?", "Rune-based")]
            )
        ),
        "WorldBibleGenerator": FakeGen(
            SimpleNamespace(world_bible="Rune-based magic, floating city of Aethel.")
        ),
        "StoryGenerator": FakeGen(
            SimpleNamespace(
                arc_outline="Arc outline",
                chapter_plan="Chapter 1: Arrival\nChapter 2: Oath",
                enhancers_guide="Tension: high",
                story=(
                    "### Chapter 1: Arrival\n\n"
                    "The caravan reached the gates at dusk.\n\n"
                    "### Chapter 2: Oath\n\n"
                    "She swore to protect the archive."
                ),
            )
        ),
        "ChapterInpaintingGenerator": FakeGen(
            SimpleNamespace(
                story=(
                    "### Chapter 1: Arrival\n\n"
                    "The weary caravan, laden with salt and silk, reached "
                    "the gates at dusk.\n\n"
                    "### Chapter 2: Oath\n\n"
                    "Beneath the runed arches she swore to protect the archive."
                ),
                expanded_chapters=2,
                total_chapters=2,
            )
        ),
    }


def _scripted_prompter(**overrides) -> ScriptedPrompter:
    defaults = dict(
        idea="A young wizard defends a floating city.",
        # One ideation round (premise accepted on first try) + one WB round.
        answer_batches=[[None, "A mountain city"], [None]],
        premise_decisions=[PremiseDecision(accept=True)],
    )
    defaults.update(overrides)
    return ScriptedPrompter(**defaults)


# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------


def test_default_stage_order_text_only():
    order = default_stage_order(PipelineOptions(check_similar=False))
    assert order == [
        StageName.IDEATE,
        StageName.PREMISE,
        StageName.SPINE,
        StageName.WORLD_BIBLE_QUESTIONS,
        StageName.WORLD_BIBLE,
        StageName.STORY,
    ]


def test_default_stage_order_with_all_optional():
    order = default_stage_order(
        PipelineOptions(
            enable_images=True,
            replicate_api_token="tok",
            inpaint_chapters=True,
            inpaint_ratio=1.35,
            check_similar=True,
        )
    )
    assert order == [
        StageName.IDEATE,
        StageName.PREMISE,
        StageName.SPINE,
        StageName.WORLD_BIBLE_QUESTIONS,
        StageName.WORLD_BIBLE,
        StageName.CHARACTER_VISUALS,
        StageName.CHARACTER_PORTRAITS,
        StageName.STORY,
        StageName.INPAINT,
        StageName.SCENE_IMAGES,
        StageName.SIMILARITY_CHECK,
    ]


# ---------------------------------------------------------------------------
# End-to-end (text pipeline, no images)
# ---------------------------------------------------------------------------


def test_run_all_text_pipeline_populates_full_state():
    generators = _build_generators()
    prompter = _scripted_prompter()
    options = PipelineOptions(check_similar=False)

    state = run_all(StoryState(), generators, prompter, options)

    assert state.idea == "A young wizard defends a floating city."
    assert state.core_premise == "A young wizard defends a floating city."
    assert state.spine_template == "Once upon a time..."
    assert state.world_bible == "Rune-based magic, floating city of Aethel."
    assert state.chapter_plan.startswith("Chapter 1: Arrival")
    assert "### Chapter 1: Arrival" in state.final_story_text
    # No inpaint → final matches story_text.
    assert state.final_story_text == state.story_text


def test_run_all_invokes_on_stage_complete_for_every_stage():
    generators = _build_generators()
    prompter = _scripted_prompter()
    options = PipelineOptions(check_similar=False)

    seen: list[StageName] = []
    run_all(
        StoryState(),
        generators,
        prompter,
        options,
        on_stage_complete=lambda stage, _state: seen.append(stage),
    )
    assert seen == default_stage_order(options)


def test_run_all_answered_qa_feeds_downstream_generators():
    generators = _build_generators()
    prompter = _scripted_prompter(
        answer_batches=[["overridden protagonist", None], [None]],
    )
    options = PipelineOptions(check_similar=False)

    run_all(StoryState(), generators, prompter, options)

    premise_call = generators["CorePremiseGenerator"].calls[0]
    # First answer was overridden; second falls back to the proposed answer.
    assert "overridden protagonist" in premise_call["qa_pairs"]
    assert "In a floating city" in premise_call["qa_pairs"]


def test_run_all_refine_loop_stacks_idea_across_iterations():
    generators = _build_generators()
    # Two generator rounds happen; the second run of ideation needs answers
    # too. Pad the scripted answers accordingly.
    prompter = ScriptedPrompter(
        idea="seed idea",
        # Two ideation rounds (one refine + one accept) + one WB round.
        answer_batches=[[None, None], [None, None], [None]],
        premise_decisions=[
            PremiseDecision(accept=False, refinement_details="make it darker"),
            PremiseDecision(accept=True),
        ],
    )
    options = PipelineOptions(check_similar=False)

    run_all(StoryState(), generators, prompter, options)

    q_calls = generators["QuestionGenerator"].calls
    assert len(q_calls) == 2, "expected one question generation per refine loop"
    # Second iteration's idea should include the previous premise and
    # the refinement text.
    assert "Original idea: seed idea" in q_calls[1]["idea"]
    assert "make it darker" in q_calls[1]["idea"]


# ---------------------------------------------------------------------------
# Inpainting + similarity check (post-processing stages)
# ---------------------------------------------------------------------------


def test_run_all_with_inpaint_replaces_final_story_text():
    generators = _build_generators()
    prompter = _scripted_prompter()
    options = PipelineOptions(
        inpaint_chapters=True, inpaint_ratio=1.35, check_similar=False
    )

    state = run_all(StoryState(), generators, prompter, options)

    assert "laden with salt and silk" in state.final_story_text
    # Pre-inpaint text stays on ``story_text`` for traceability.
    assert "laden with salt and silk" not in state.story_text
    assert state.story_text != state.final_story_text


def test_run_all_similarity_check_records_report():
    generators = _build_generators()
    prompter = _scripted_prompter()
    options = PipelineOptions(check_similar=True, similar_threshold=0.65)

    state = run_all(StoryState(), generators, prompter, options)

    assert state.similarity_report is not None
    # Our fake story is too short to trigger duplicates, but the report
    # must still be populated (even as an empty result).
    assert isinstance(state.similarity_report, str)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_run_all_validates_options_upfront():
    generators = _build_generators()
    prompter = _scripted_prompter()
    # enable_images without a token must fail fast, before any stage runs.
    options = PipelineOptions(enable_images=True, replicate_api_token=None)

    with pytest.raises(ValueError, match="replicate_api_token"):
        run_all(StoryState(), generators, prompter, options)

    # No generator should have been called.
    assert generators["QuestionGenerator"].calls == []


def test_state_survives_json_roundtrip_after_full_run():
    generators = _build_generators()
    prompter = _scripted_prompter()
    options = PipelineOptions(check_similar=False)

    state = run_all(StoryState(), generators, prompter, options)
    restored = StoryState.from_dict(state.to_dict())
    assert restored == state
