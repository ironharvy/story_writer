from __future__ import annotations

from pathlib import Path

from story_writer.models.run import Manifest, StageRecord
from story_writer.models.story import Chapter, Premise, StoryLength
from story_writer.render import render_story
from story_writer.run_store import RunStore


def test_render_story_writes_markdown(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    manifest = Manifest(
        slug="demo",
        stages={
            "prose:01": StageRecord(
                status="complete",
                provider="ollama",
                model="qwen3:latest",
                temperature=0.75,
                max_tokens=8192,
                context_window=40960,
                estimated_input_tokens=5500,
            )
        },
    )
    store.create("demo", "An idea", manifest)
    store.write_model(
        "demo",
        "premise",
        Premise(
            protagonist="Lila",
            want="find Elias",
            obstacle="a flood",
            stakes="memory",
            genre="fantasy",
            length=StoryLength.SHORT,
            summary="A search through a flooded valley.",
        ),
    )
    store.write_model(
        "demo",
        "chapter:1",
        Chapter(number=1, title="The Water", prose="Lila listened."),
    )

    path = render_story("demo", store)

    assert path.name == "story.md"
    rendered = path.read_text(encoding="utf-8")
    assert "## Chapter 1 - The Water" in rendered
    assert "## Generation Parameters" in rendered
    assert "prose:01: provider=ollama, model=qwen3:latest" in rendered


def test_render_story_tolerates_legacy_chapter_shape(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.create("demo", "An idea", Manifest(slug="demo"))
    store.write_model(
        "demo",
        "premise",
        Premise(
            protagonist="Lila",
            want="find Elias",
            obstacle="a flood",
            stakes="memory",
            genre="fantasy",
            length=StoryLength.SHORT,
            summary="A search through a flooded valley.",
        ),
    )
    store.write_json(
        "demo",
        "chapter:1",
        {
            "number": 1,
            "title": "Legacy",
            "enhancement_notes": [],
            "embellishment": "a wooden toy",
            "prose": "Lila listened.",
        },
    )

    path = render_story("demo", store)

    assert "## Chapter 1 - Legacy" in path.read_text(encoding="utf-8")


def test_render_story_backfills_known_context_for_legacy_manifest(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path)
    store.create(
        "demo",
        "An idea",
        Manifest(
            slug="demo",
            stages={
                "prose:01": StageRecord(
                    status="complete",
                    provider="ollama",
                    model="qwen3:latest",
                )
            },
        ),
    )
    store.write_model(
        "demo",
        "premise",
        Premise(
            protagonist="Lila",
            want="find Elias",
            obstacle="a flood",
            stakes="memory",
            genre="fantasy",
            length=StoryLength.SHORT,
            summary="A search through a flooded valley.",
        ),
    )
    store.write_model(
        "demo",
        "chapter:1",
        Chapter(number=1, title="The Water", prose="Lila listened."),
    )

    path = render_story("demo", store)
    rendered = path.read_text(encoding="utf-8")

    assert "context_window=40960" in rendered
    assert "max_tokens=8192" in rendered
    assert "estimated_input_tokens=5500" in rendered
