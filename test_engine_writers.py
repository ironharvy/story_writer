"""Tests for :mod:`engine.writers.markdown`."""

from __future__ import annotations

import os

from engine.types import CharacterVisualState, StoryState
from engine.writers.markdown import render_markdown, write_markdown


def _base_state() -> StoryState:
    return StoryState(
        core_premise="A young wizard defends a floating city.",
        spine_template="Once upon a time...",
        world_bible="Rune-based magic.",
        arc_outline="Arc outline",
        chapter_plan="Chapter 1: Arrival\nChapter 2: Oath",
        enhancers_guide="Tension: high",
        story_text=(
            "### Chapter 1: Arrival\n\nThe caravan reached the gates.\n\n"
            "### Chapter 2: Oath\n\nShe swore."
        ),
        final_story_text=(
            "### Chapter 1: Arrival\n\nThe caravan reached the gates.\n\n"
            "### Chapter 2: Oath\n\nShe swore."
        ),
    )


def test_render_markdown_contains_all_sections():
    md = render_markdown(_base_state())
    for heading in [
        "# Story Output",
        "## Core Premise",
        "## Spine Template",
        "## World Bible",
        "## Arc Outline",
        "## Chapter Plan",
        "## Enhancers Guide",
        "## Final Story",
    ]:
        assert heading in md


def test_render_markdown_omits_character_visuals_when_empty():
    md = render_markdown(_base_state())
    assert "## Character Visuals" not in md


def test_render_markdown_includes_character_visuals_and_portraits():
    state = _base_state()
    state.character_visuals = [
        CharacterVisualState("Alice", "mix", "blue hair", "prompt"),
    ]
    state.character_portrait_paths = {"Alice": "/tmp/alice.png"}
    md = render_markdown(state)
    assert "## Character Visuals" in md
    assert "### Alice" in md
    assert "**Reference:** mix" in md
    assert "**Features:** blue hair" in md
    assert "![Alice portrait](/tmp/alice.png)" in md


def test_render_markdown_interleaves_scene_images_when_present():
    state = _base_state()
    state.scene_image_paths = {1: "/tmp/1.png", 2: "/tmp/2.png"}
    md = render_markdown(state)
    # Legacy format: the story section is reconstructed by splitting on
    # "### Chapter " and re-inserting the scene after each chapter.
    assert "![Chapter 1 scene](/tmp/1.png)" in md
    assert "![Chapter 2 scene](/tmp/2.png)" in md


def test_render_markdown_uses_final_story_text_when_no_scene_images():
    state = _base_state()
    state.final_story_text = "FINAL STORY BODY"
    md = render_markdown(state)
    assert "FINAL STORY BODY" in md


def test_write_markdown_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested" / "dir" / "story.md"
    path = write_markdown(_base_state(), str(nested))
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "# Story Output" in content
