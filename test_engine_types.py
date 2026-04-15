"""Unit tests for ``engine.types`` serialization and validation."""

import pytest

from engine.types import (
    CharacterVisualState,
    PipelineOptions,
    QAPair,
    StageName,
    StoryState,
)


def test_qapair_effective_answer_prefers_user_answer():
    assert QAPair("q", "prop", "override").effective_answer == "override"
    assert QAPair("q", "prop", None).effective_answer == "prop"


def test_qapair_roundtrip():
    original = QAPair("q", "p", "u")
    restored = QAPair.from_dict(original.to_dict())
    assert restored == original


def test_character_visual_state_roundtrip_tolerates_missing_fields():
    cv = CharacterVisualState.from_dict({"name": "Alice"})
    assert cv.name == "Alice"
    assert cv.reference_mix == ""
    assert cv.distinguishing_features == ""
    assert cv.full_prompt == ""


def test_story_state_roundtrip_preserves_nested_state():
    original = StoryState(
        idea="a wizard",
        ideation_qa=[
            QAPair("Q1?", "proposed", "user ans"),
            QAPair("Q2?", "p2", None),
        ],
        core_premise="premise",
        spine_template="spine",
        world_bible_qa=[QAPair("wq?", "wp", None)],
        world_bible="wb",
        character_visuals=[
            CharacterVisualState("Alice", "ref", "feat", "prompt"),
        ],
        character_portrait_paths={"Alice": "/tmp/a.png"},
        arc_outline="arc",
        chapter_plan="cp",
        enhancers_guide="eg",
        story_text="story",
        final_story_text="final",
        scene_image_paths={1: "/tmp/1.png", 2: "/tmp/2.png"},
        similarity_report="similar",
    )
    restored = StoryState.from_dict(original.to_dict())
    assert restored == original


def test_story_state_scene_image_keys_roundtrip_as_ints():
    # JSON serialization forces string keys; ensure we coerce back to int
    # so downstream code can index by chapter number.
    state = StoryState(scene_image_paths={3: "/x.png"})
    as_dict = state.to_dict()
    assert as_dict["scene_image_paths"] == {"3": "/x.png"}
    restored = StoryState.from_dict(as_dict)
    assert restored.scene_image_paths == {3: "/x.png"}


def test_pipeline_options_rejects_bad_inpaint_ratio():
    opts = PipelineOptions(inpaint_chapters=True, inpaint_ratio=1.0)
    with pytest.raises(ValueError, match="inpaint_ratio"):
        opts.validate()


def test_pipeline_options_requires_replicate_token_when_images_enabled():
    opts = PipelineOptions(enable_images=True, replicate_api_token=None)
    with pytest.raises(ValueError, match="replicate_api_token"):
        opts.validate()


def test_pipeline_options_rejects_out_of_range_similarity():
    with pytest.raises(ValueError, match="similar_threshold"):
        PipelineOptions(similar_threshold=1.5).validate()


def test_pipeline_options_accepts_valid_combination():
    PipelineOptions(
        enable_images=True,
        replicate_api_token="tok",
        inpaint_chapters=True,
        inpaint_ratio=1.35,
        similar_threshold=0.65,
    ).validate()


def test_stage_name_values_are_stable_strings():
    # These values are persisted; if this test breaks, a migration is needed.
    assert StageName.IDEATE.value == "ideate"
    assert StageName.WORLD_BIBLE_QUESTIONS.value == "world_bible_questions"
    assert StageName.SCENE_IMAGES.value == "scene_images"
