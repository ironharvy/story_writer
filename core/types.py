"""Dataclasses passed between pipeline stages and the generator protocol.

These are the *contract* the foundation produces and every chapter-drafting
generator consumes: a structured `WorldBible` plus an ordered `PlanEntry`
list. `WorldState` is the world-state variant's continuity carrier; it
lives here (not in the variant module) so other generators can adopt it
without an import cycle.

`DraftingInput` / `DraftingOutput` / `StoryGenerator` define the protocol
the `generators/` registry implements: each variant takes a `DraftingInput`
(idea, title, spine, world_bible, chapters_plan, output_file) and returns
a `DraftingOutput` (the drafted chapters + an opaque continuity artifact
the variant chose to carry across chapters).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Protocol


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


# --- generator protocol ------------------------------------------------------


@dataclass
class DraftedChapter:
    """One chapter of the finished draft: its title plus the prose."""

    chapter_title: str
    prose: str


@dataclass(frozen=True)
class DraftingInput:
    """The shared 'after-foundation' state every generator drafts from.

    `output_file` is included so generators can keep writing incrementally
    via :func:`core.artifact.update_artifact` during the drafting loop —
    lifting I/O entirely into the orchestrator is a follow-up; for now the
    Protocol's contract is "produces these chapters + this continuity
    artifact", not "owns disk I/O".
    """

    idea: str
    title: str
    spine: str
    world_bible: WorldBible
    chapters_plan: list[PlanEntry]
    output_file: str
    number_of_chapters: int
    # The shared canon/continuity config (canon.Canon). Optional and typed as
    # `object` to keep core dependency-free; generators that use it import the
    # concrete type. `None` -> generators run canon-free.
    canon: object | None = None


@dataclass
class DraftingOutput:
    """The shared return type from every generator's `draft()`.

    `continuity_artifact` is whatever the variant chose to carry across
    chapters — `None` for variants without continuity (baseline summary
    is internal; dspy.Module has none), a final :class:`WorldState` for
    the world-state variant, an aggregate tree for the (deferred)
    recursive variant, etc. Type-erased on purpose.
    """

    chapters: list[DraftedChapter]
    continuity_artifact: object | None = None


class StoryGenerator(Protocol):
    """The plug-in contract for chapter-drafting variants.

    Concrete classes register themselves via
    :func:`generators.register` and expose `id` / `status` / `description`
    as class-level metadata so the registry can render `--list-strategies`
    output without instantiating anything.
    """

    id: ClassVar[str]
    status: ClassVar[str]        # "experimental" | "promoted" | "deprecated"
    description: ClassVar[str]

    def draft(self, inp: DraftingInput) -> DraftingOutput: ...


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
