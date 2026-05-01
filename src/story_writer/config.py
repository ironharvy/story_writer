from __future__ import annotations

import os
from dataclasses import dataclass

from story_writer.models.story import StoryLength


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    context_window: int | None = None


DEFAULT_OLLAMA_HOST = "http://localhost:11434"

DEFAULT_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen3:latest": 40960,
    "gemma4:26b": 262144,
    "gemma3:4b": 131072,
    "gemma3:latest": 131072,
    "deepseek-r1:7b": 131072,
}

DEFAULT_ROUTING: dict[str, ProviderConfig] = {
    "clarify": ProviderConfig(
        "ollama", "qwen3:latest", temperature=0.4, context_window=40960
    ),
    "premise": ProviderConfig(
        "ollama", "qwen3:latest", temperature=0.5, context_window=40960
    ),
    "spine": ProviderConfig(
        "ollama", "qwen3:latest", temperature=0.5, context_window=40960
    ),
    "world_bible": ProviderConfig(
        "ollama", "gemma4:26b", temperature=0.45, context_window=262144
    ),
    "chapter_plan": ProviderConfig(
        "ollama", "qwen3:latest", temperature=0.45, context_window=40960
    ),
    "enhancement": ProviderConfig(
        "ollama", "qwen3:latest", temperature=0.6, context_window=40960
    ),
    "embellish": ProviderConfig(
        "ollama", "gemma3:4b", temperature=0.8, context_window=131072
    ),
    "prose": ProviderConfig(
        "ollama",
        "gemma4:26b",
        temperature=0.75,
        max_tokens=8192,
        context_window=262144,
    ),
}

LENGTH_CHAPTER_COUNTS: dict[StoryLength, int] = {
    StoryLength.SHORT: 5,
    StoryLength.NOVELLA: 12,
    StoryLength.NOVEL: 24,
}


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)


def routing_for(stage_name: str, model_override: str | None = None) -> ProviderConfig:
    base = DEFAULT_ROUTING[stage_name]
    if model_override:
        return ProviderConfig(
            provider=base.provider,
            model=model_override,
            temperature=base.temperature,
            max_tokens=base.max_tokens,
            context_window=DEFAULT_MODEL_CONTEXT_WINDOWS.get(model_override),
        )
    return base


def estimated_input_tokens(stage_name: str, length: StoryLength) -> int:
    chapter_count = LENGTH_CHAPTER_COUNTS[length]
    base_estimates = {
        "clarify": 700,
        "premise": 1200,
        "spine": 1800,
        "world_bible": 2800,
        "chapter_plan": 2200 + chapter_count * 180,
        "enhancement": 2600 + chapter_count * 220,
        "embellish": 1800,
        "prose": 5500,
    }
    return base_estimates.get(stage_name, 2000)


def context_warning(
    stage_name: str, length: StoryLength, config: ProviderConfig
) -> str:
    estimated = estimated_input_tokens(stage_name, length) + config.max_tokens
    if config.context_window is None:
        return (
            f"Context window unknown for {config.model}; stage {stage_name} "
            f"estimated budget is {estimated} tokens."
        )
    threshold = int(config.context_window * 0.8)
    if estimated > threshold:
        return (
            f"Stage {stage_name} estimated budget {estimated} tokens approaches "
            f"context window {config.context_window} for {config.model}."
        )
    return ""
