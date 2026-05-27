"""Registry of chapter-drafting variants implementing :class:`core.types.StoryGenerator`.

Each variant is a self-contained module that calls :func:`register` at
import time. The orchestrator imports submodules to populate ``REGISTRY``;
the CLI's ``--strategy`` flag and ``--list-strategies`` flag both read
from it.

Lifecycle (enforced socially, documented in AGENTS.md once that section
lands):

- ``experimental`` — new variant, must be run via explicit ``--strategy``,
  included in benchmark sweeps. Owner has ~4 weeks to promote or it
  auto-deprecates.
- ``promoted``    — eligible for default selection; counts against the
  cap of 4 simultaneously-promoted variants.
- ``deprecated``  — kept registered for one cycle so users on
  ``--strategy=<id>`` see a deprecation warning, then **deleted** (not
  archived).

Adding a generator:
    @register(id="my_variant", status="experimental",
              description="One-line claim of what makes this variant different")
    class MyVariant:
        id = "my_variant"
        status = "experimental"
        description = "..."
        def draft(self, inp: DraftingInput) -> DraftingOutput: ...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

from core.types import StoryGenerator


@dataclass(frozen=True)
class GeneratorEntry:
    id: str
    status: str            # "experimental" | "promoted" | "deprecated"
    description: str
    cls: Type[StoryGenerator]


REGISTRY: dict[str, GeneratorEntry] = {}


def register(
    *,
    id: str,
    status: str,
    description: str,
) -> Callable[[Type[StoryGenerator]], Type[StoryGenerator]]:
    """Class decorator that adds a generator to the registry on import."""
    if status not in {"experimental", "promoted", "deprecated"}:
        raise ValueError(f"invalid status {status!r}")

    def decorator(cls: Type[StoryGenerator]) -> Type[StoryGenerator]:
        if id in REGISTRY:
            raise ValueError(f"generator id {id!r} already registered")
        REGISTRY[id] = GeneratorEntry(id=id, status=status, description=description, cls=cls)
        return cls

    return decorator


def get(id: str) -> GeneratorEntry:
    """Look up a registered generator by id; raises KeyError with a helpful list."""
    try:
        return REGISTRY[id]
    except KeyError as e:
        known = ", ".join(sorted(REGISTRY)) or "(none — import the variant modules)"
        raise KeyError(f"no generator named {id!r}; known: {known}") from e


def promoted() -> list[GeneratorEntry]:
    """All currently-promoted generators, in registration order."""
    return [e for e in REGISTRY.values() if e.status == "promoted"]
