"""Dataclasses passed between pipeline stages.

These are the *contract* the foundation produces and every chapter-drafting
generator consumes: a structured `WorldBible` plus an ordered `PlanEntry`
list. `WorldState` is the world-state variant's continuity carrier; it
lives here (not in the variant module) so other generators can adopt it
without an import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Character:
    name: str
    description: str
    role: str
    relationships: list[str]
    arc: str


@dataclass
class PlanEntry:
    chapter_title: str
    chapter_beats: str


@dataclass
class WorldBible:
    story_title: str
    rules_of_the_world: list[str]
    characters: list[str]
    locations: list[str]
    timeline: list[str]


@dataclass
class WorldState:
    """A snapshot of everything that changes while the story is being told."""

    story_clock: str
    characters: list[str]
    locations: list[str]
    plot_threads: list[str]
    key_objects: list[str]
    recent_events: list[str] = field(default_factory=list)


def render_world_state(state: WorldState) -> str:
    """Render a ``WorldState`` as a human-readable markdown block."""

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items) if items else "- (none)"

    return "\n".join(
        [
            f"**Story clock:** {state.story_clock}",
            "",
            "**Characters (current state):**",
            _bullets(state.characters),
            "",
            "**Locations (current state):**",
            _bullets(state.locations),
            "",
            "**Open plot threads:**",
            _bullets(state.plot_threads),
            "",
            "**Key objects:**",
            _bullets(state.key_objects),
            "",
            "**Recent events:**",
            _bullets(state.recent_events),
        ]
    )
