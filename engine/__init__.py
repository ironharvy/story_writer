"""Headless story generation engine.

This package decouples the story generation pipeline from the interactive CLI
so that it can run inside web workers, tests, or other non-TTY contexts.

Initial scope (step 1-3 of the extraction plan):
- ``engine.types``      — data contracts (StoryState, QAPair, PipelineOptions).
- ``engine.prompter``   — Prompter protocol for human-input abstraction.
- ``engine.config``     — DSPy model configuration and generator wiring.

Later steps will add ``engine.stages`` and ``engine.pipeline``.
"""

from engine.types import (
    PipelineOptions,
    QAPair,
    StageName,
    StoryState,
)
from engine.prompter import Prompter

__all__ = [
    "PipelineOptions",
    "Prompter",
    "QAPair",
    "StageName",
    "StoryState",
]
