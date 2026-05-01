from __future__ import annotations

from story_writer.config import ProviderConfig, context_warning, routing_for
from story_writer.models.story import StoryLength


def test_known_model_override_carries_context_window() -> None:
    config = routing_for("prose", model_override="qwen3:latest")

    assert config.context_window == 40960


def test_unknown_model_override_warns_about_context() -> None:
    config = routing_for("prose", model_override="small-local:latest")

    warning = context_warning("prose", StoryLength.NOVEL, config)

    assert "Context window unknown" in warning
    assert "small-local:latest" in warning


def test_context_warning_when_budget_approaches_window() -> None:
    config = ProviderConfig(
        provider="ollama",
        model="tiny",
        max_tokens=2048,
        context_window=4096,
    )

    warning = context_warning("chapter_plan", StoryLength.NOVEL, config)

    assert "approaches context window" in warning
