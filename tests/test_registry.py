"""Contract tests for the generator registry.

These don't depend on any concrete generator; they exercise the registry
machinery (register / get / promoted / status validation) so that
generators can be added independently.
"""
from __future__ import annotations

import pytest

import generators
from core.types import DraftingInput, DraftingOutput  # noqa: F401


@pytest.fixture(autouse=True)
def _isolated_registry():
    """Snapshot and restore REGISTRY around each test."""
    snapshot = dict(generators.REGISTRY)
    generators.REGISTRY.clear()
    try:
        yield
    finally:
        generators.REGISTRY.clear()
        generators.REGISTRY.update(snapshot)


def _stub(id_: str, status: str = "experimental"):
    cls = type(
        f"_Stub_{id_}",
        (),
        {
            "id": id_,
            "status": status,
            "description": f"stub {id_}",
            "draft": lambda self, inp: DraftingOutput(chapters=[]),
        },
    )
    return generators.register(id=id_, status=status, description=f"stub {id_}")(cls)


def test_register_adds_to_registry():
    _stub("alpha")
    assert "alpha" in generators.REGISTRY
    entry = generators.REGISTRY["alpha"]
    assert entry.id == "alpha"
    assert entry.status == "experimental"
    assert "stub alpha" in entry.description


def test_get_returns_entry_or_raises_with_known_list():
    _stub("alpha")
    assert generators.get("alpha").id == "alpha"
    with pytest.raises(KeyError, match="no generator named 'missing'"):
        generators.get("missing")


def test_register_rejects_invalid_status():
    with pytest.raises(ValueError, match="invalid status"):
        generators.register(id="bad", status="great", description="x")


def test_register_rejects_duplicate_id():
    _stub("alpha")
    with pytest.raises(ValueError, match="already registered"):
        _stub("alpha")


def test_promoted_filters_by_status():
    _stub("a", status="experimental")
    _stub("b", status="promoted")
    _stub("c", status="deprecated")
    _stub("d", status="promoted")
    promoted_ids = [e.id for e in generators.promoted()]
    assert promoted_ids == ["b", "d"]


def test_registered_class_satisfies_protocol_shape():
    """A registered class exposes the StoryGenerator class-vars + draft()."""
    Cls = _stub("alpha")
    assert hasattr(Cls, "id") and hasattr(Cls, "status") and hasattr(Cls, "description")
    assert callable(getattr(Cls, "draft"))
