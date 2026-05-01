from __future__ import annotations

from pathlib import Path

import pytest

from story_writer.models.run import Manifest
from story_writer.models.story import Premise, StoryLength
from story_writer.run_store import RunStore, slugify, validate_slug


def test_slugify_keeps_local_safe_name() -> None:
    assert slugify("A Girl Looks for a Brother!") == "a-girl-looks-for-a-brother"


def test_validate_slug_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="single path segment"):
        validate_slug("../outside")


def test_run_store_round_trips_manifest_and_model(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    manifest = Manifest(slug="demo", length="short")
    store.create("demo", "An idea", manifest)

    premise = Premise(
        protagonist="Lila",
        want="find Elias",
        obstacle="a memory-eating flood",
        stakes="the valley will forget itself",
        genre="fantasy",
        length=StoryLength.SHORT,
        summary="Lila searches a flooded valley for her forgotten brother.",
    )
    store.write_model("demo", "premise", premise)

    assert store.read_manifest("demo").slug == "demo"
    assert store.read_idea("demo") == "An idea"
    assert store.read_model("demo", "premise", Premise).protagonist == "Lila"


def test_run_store_does_not_reuse_existing_run(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.create("demo", "An idea", Manifest(slug="demo"))

    with pytest.raises(FileExistsError, match="already exists"):
        store.create("demo", "Different idea", Manifest(slug="demo"))
