"""Presentation-layer writers that serialize :class:`StoryState` to formats
suitable for end users (markdown, JSON, HTML, ...). Writers are intentionally
separate from pipeline stages so the web app can render ``StoryState``
directly without going through a file on disk."""

from engine.writers.markdown import write_markdown

__all__ = ["write_markdown"]
