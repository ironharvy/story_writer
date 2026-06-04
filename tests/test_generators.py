"""Structural smoke tests for the three registered generators.

These don't run the drafting loops (those need an LLM) — they just lock
in the registry membership and the Protocol shape so the contract can't
silently drift.
"""
from __future__ import annotations

# Importing the modules triggers their @register decorators.
import generators  # noqa: F401
import generators.baseline  # noqa: F401
import generators.dspy_module  # noqa: F401
import generators.world_state  # noqa: F401


def test_three_promoted_generators_registered():
    ids = {e.id for e in generators.REGISTRY.values()}
    assert {"baseline", "world_state", "dspy_module"}.issubset(ids)


def test_all_three_are_promoted():
    promoted_ids = {e.id for e in generators.promoted()}
    assert {"baseline", "world_state", "dspy_module"}.issubset(promoted_ids)


def test_each_generator_exposes_protocol_metadata():
    for id_ in ("baseline", "world_state", "dspy_module"):
        entry = generators.get(id_)
        cls = entry.cls
        assert cls.id == id_
        assert cls.status == "promoted"
        assert isinstance(cls.description, str) and cls.description
        assert callable(getattr(cls, "draft"))


def test_descriptions_are_distinct():
    """Each generator claims a different niche."""
    descs = {generators.get(i).cls.description for i in ("baseline", "world_state", "dspy_module")}
    assert len(descs) == 3
