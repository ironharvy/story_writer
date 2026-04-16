"""Headless story generation engine.

This package decouples the story generation pipeline from the interactive CLI
so that it can run inside web workers, tests, or other non-TTY contexts.

Public API:
- :mod:`engine.types`     — data contracts (StoryState, QAPair, ...).
- :mod:`engine.prompter`  — Prompter protocol for human-input abstraction.
- :mod:`engine.config`    — DSPy model configuration and generator wiring.
- :mod:`engine.stages`    — per-stage pipeline functions.
- :mod:`engine.pipeline`  — stage ordering + ``run_all`` orchestration.
- :mod:`engine.writers`   — presentation-layer output (markdown today).
"""

from engine.pipeline import default_stage_order, run_all, run_stage
from engine.prompter import Prompter
from engine.types import (
    CharacterVisualState,
    PipelineOptions,
    QAPair,
    StageName,
    StoryState,
)

__all__ = [
    "CharacterVisualState",
    "PipelineOptions",
    "Prompter",
    "QAPair",
    "StageName",
    "StoryState",
    "default_stage_order",
    "run_all",
    "run_stage",
]
