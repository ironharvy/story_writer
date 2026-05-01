from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from story_writer.config import (
    DEFAULT_MODEL_CONTEXT_WINDOWS,
    DEFAULT_ROUTING,
    estimated_input_tokens,
)
from story_writer.models.run import Manifest
from story_writer.models.story import Chapter, Premise, StoryLength
from story_writer.run_store import RunStore


def render_story(
    slug: str, store: RunStore, fmt: str = "md", out: Path | None = None
) -> Path:
    premise = store.read_model(slug, "premise", Premise)
    manifest = store.read_manifest(slug)
    chapters = _read_chapters(slug, store)
    text = (
        _render_markdown(slug, premise, chapters, manifest)
        if fmt == "md"
        else _render_text(premise, chapters, manifest)
    )
    output_path = out or store.run_dir(slug) / (
        "story.md" if fmt == "md" else "story.txt"
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _read_chapters(slug: str, store: RunStore) -> list[Chapter]:
    chapter_dir = store.run_dir(slug) / "chapters"
    chapters: list[Chapter] = []
    for path in sorted(chapter_dir.glob("*.json")):
        text = path.read_text(encoding="utf-8")
        try:
            chapters.append(Chapter.model_validate_json(text))
        except ValidationError:
            chapters.append(_legacy_chapter_from_json(text))
    return chapters


def _legacy_chapter_from_json(text: str) -> Chapter:
    import json

    data = json.loads(text)
    return Chapter(
        number=int(data.get("number", 0)),
        title=str(data.get("title", "Untitled")),
        prose=str(data.get("prose", "")),
    )


def _render_markdown(
    slug: str, premise: Premise, chapters: list[Chapter], manifest: Manifest
) -> str:
    blocks = [f"# {slug}", "", f"> {premise.summary}"]
    for chapter in chapters:
        blocks.extend(
            [
                "",
                f"## Chapter {chapter.number} - {chapter.title}",
                "",
                chapter.prose.strip(),
            ]
        )
    blocks.extend(_render_generation_metadata(manifest, markdown=True))
    return "\n".join(blocks).strip() + "\n"


def _render_text(premise: Premise, chapters: list[Chapter], manifest: Manifest) -> str:
    blocks = [premise.summary]
    for chapter in chapters:
        blocks.extend(
            [f"Chapter {chapter.number}: {chapter.title}", chapter.prose.strip()]
        )
    blocks.extend(_render_generation_metadata(manifest, markdown=False))
    return "\n\n".join(blocks).strip() + "\n"


def _render_generation_metadata(manifest: Manifest, *, markdown: bool) -> list[str]:
    heading = "## Generation Parameters" if markdown else "Generation Parameters"
    lines = [
        "",
        heading,
        "",
        f"- length: {manifest.length}",
        f"- embellish_probability: {manifest.embellish_probability}",
        f"- allow_paid: {manifest.allow_paid}",
    ]
    warnings = [
        record.context_warning
        for record in manifest.stages.values()
        if record.context_warning
    ]
    if warnings:
        lines.extend(["", "Context warnings:"])
        lines.extend(f"- {warning}" for warning in sorted(set(warnings)))
    lines.extend(["", "Stage routing:"])
    for name, record in manifest.stages.items():
        base_name = name.split(":", maxsplit=1)[0]
        default_config = DEFAULT_ROUTING.get(base_name)
        context_window = record.context_window or DEFAULT_MODEL_CONTEXT_WINDOWS.get(
            record.model
        )
        temperature = record.temperature
        if temperature is None and default_config is not None:
            temperature = default_config.temperature
        max_tokens = record.max_tokens
        if max_tokens is None and default_config is not None:
            max_tokens = default_config.max_tokens
        context = "unknown" if context_window is None else str(context_window)
        estimated_tokens = record.estimated_input_tokens
        if estimated_tokens is None:
            estimated_tokens = _legacy_estimated_tokens(base_name, manifest)
        estimated = "unknown" if estimated_tokens is None else str(estimated_tokens)
        lines.append(
            "- "
            f"{name}: provider={record.provider}, model={record.model}, "
            f"temperature={temperature}, max_tokens={max_tokens}, "
            f"context_window={context}, estimated_input_tokens={estimated}"
        )
    return lines


def _legacy_estimated_tokens(stage_name: str, manifest: Manifest) -> int | None:
    try:
        length = StoryLength(manifest.length)
    except ValueError:
        return None
    return estimated_input_tokens(stage_name, length)
