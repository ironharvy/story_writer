"""Model resolution: presets, provider/model ids, and ollama/ normalization.

Pure string logic — no LLM or network.
"""
from __future__ import annotations

from storyforge import config


def test_preset_expands():
    assert config.resolve_model("fast", "X") == config.PRESETS["fast"]
    assert config.resolve_model("quality", "X") == config.PRESETS["quality"]


def test_ollama_prefix_normalized_to_chat_endpoint():
    assert config.resolve_model("ollama/qwen3.6:27b", "X") == "ollama_chat/qwen3.6:27b"


def test_ollama_chat_passes_through_untouched():
    assert config.resolve_model("ollama_chat/qwen3.6:27b", "X") == "ollama_chat/qwen3.6:27b"


def test_hosted_provider_model_passes_through():
    assert config.resolve_model("deepseek/deepseek-chat", "X") == "deepseek/deepseek-chat"
    assert config.resolve_model("groq/llama-3.3-70b-versatile", "X") == \
        "groq/llama-3.3-70b-versatile"


def test_bare_name_passes_through_unchanged():
    # User chose explicit-provider behavior: no auto-prefixing to ollama.
    assert config.resolve_model("qwen3.6:27b", "X") == "qwen3.6:27b"


def test_empty_or_none_uses_normalized_fallback():
    assert config.resolve_model(None, "ollama/foo") == "ollama_chat/foo"
    assert config.resolve_model("", "deepseek/deepseek-chat") == "deepseek/deepseek-chat"


def test_normalized_ollama_is_still_detected_as_ollama():
    assert config.is_ollama(config.resolve_model("ollama/qwen3:latest", "X"))
    assert config.is_ollama(config.resolve_model("quality", "X"))


# --- independent judge selection (no self-grading) ----------------------------

def test_pick_independent_judge_never_equals_draft():
    fast = config.resolve_model("fast", "X")
    quality = config.resolve_model("quality", "X")
    # draft fast -> judge quality; draft quality -> judge fast; neither equals draft
    assert config.pick_independent_judge(fast) == quality
    assert config.pick_independent_judge(quality) == fast
    assert config.pick_independent_judge(fast) != fast
    assert config.pick_independent_judge(quality) != quality


def test_pick_independent_judge_hosted_defaults_to_local_quality():
    quality = config.resolve_model("quality", "X")
    assert config.pick_independent_judge("deepseek/deepseek-chat") == quality
    assert config.pick_independent_judge("groq/llama-3.3-70b-versatile") == quality


def test_pick_independent_judge_handles_ollama_prefix():
    # the bare ollama/ form of the draft must still resolve to a different judge
    quality = config.resolve_model("quality", "X")
    assert config.pick_independent_judge("ollama/qwen3:latest") == quality
