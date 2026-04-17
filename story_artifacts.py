"""Structured story artifacts: JSON sidecar I/O and markdown rendering.

A story run produces text (world bible, chapters, etc.) plus optional image
paths.  This module is the single source of truth for how those pieces are
persisted as JSON and re-rendered as the user-facing markdown, so the main
interactive pipeline and the standalone image-generation script can share it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1


@dataclass
class CharacterVisualData:
    """Plain-data view of a character's visual descriptor, decoupled from DSPy."""

    name: str
    reference_mix: str
    distinguishing_features: str
    full_prompt: str


@dataclass
class StoryArtifacts:
    """All text + image references produced by a single story run."""

    core_premise: str = ""
    spine_template: str = ""
    world_bible: str = ""
    arc_outline: str = ""
    chapter_plan: str = ""
    enhancers_guide: str = ""
    final_story: str = ""
    character_visuals: list[CharacterVisualData] = field(default_factory=list)
    character_portrait_paths: dict[str, str] = field(default_factory=dict)
    scene_image_paths: dict[int, str] = field(default_factory=dict)
    version: int = ARTIFACT_VERSION

    def to_dict(self) -> dict:
        data = asdict(self)
        # JSON object keys must be strings; chapter indices are ints.
        data["scene_image_paths"] = {
            str(k): v for k, v in self.scene_image_paths.items()
        }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StoryArtifacts":
        character_visuals = [
            CharacterVisualData(**cv) for cv in data.get("character_visuals", [])
        ]
        scene_image_paths = {
            int(k): v for k, v in (data.get("scene_image_paths") or {}).items()
        }
        return cls(
            core_premise=data.get("core_premise", ""),
            spine_template=data.get("spine_template", ""),
            world_bible=data.get("world_bible", ""),
            arc_outline=data.get("arc_outline", ""),
            chapter_plan=data.get("chapter_plan", ""),
            enhancers_guide=data.get("enhancers_guide", ""),
            final_story=data.get("final_story", ""),
            character_visuals=character_visuals,
            character_portrait_paths=dict(data.get("character_portrait_paths") or {}),
            scene_image_paths=scene_image_paths,
            version=data.get("version", ARTIFACT_VERSION),
        )


def save_artifacts(path: str | Path, artifacts: StoryArtifacts) -> Path:
    """Serialize artifacts to a JSON file, creating parent dirs as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(artifacts.to_dict(), fh, indent=2, ensure_ascii=False)
    logger.info("Saved story artifacts to %s", out)
    return out


def load_artifacts(path: str | Path) -> StoryArtifacts:
    """Load artifacts from a JSON file produced by ``save_artifacts``."""
    src = Path(path)
    with src.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return StoryArtifacts.from_dict(data)


def render_markdown(artifacts: StoryArtifacts) -> str:
    """Render the human-readable markdown view of a story run."""
    parts: list[str] = []
    parts.append("# Story Output\n\n")
    parts.append("## Core Premise\n")
    parts.append(f"{artifacts.core_premise}\n\n")
    parts.append("## Spine Template\n")
    parts.append(f"{artifacts.spine_template}\n\n")
    parts.append("## World Bible\n")
    parts.append(f"{artifacts.world_bible}\n\n")

    if artifacts.character_visuals:
        parts.append("## Character Visuals\n\n")
        for cv in artifacts.character_visuals:
            parts.append(f"### {cv.name}\n")
            parts.append(f"**Reference:** {cv.reference_mix}\n\n")
            parts.append(f"**Features:** {cv.distinguishing_features}\n\n")
            portrait = artifacts.character_portrait_paths.get(cv.name)
            if portrait:
                parts.append(f"![{cv.name} portrait]({portrait})\n\n")

    parts.append("## Arc Outline\n")
    parts.append(f"{artifacts.arc_outline}\n\n")
    parts.append("## Chapter Plan\n")
    parts.append(f"{artifacts.chapter_plan}\n\n")
    parts.append("## Enhancers Guide\n")
    parts.append(f"{artifacts.enhancers_guide}\n\n")
    parts.append("## Final Story\n")

    if artifacts.scene_image_paths:
        chapters = artifacts.final_story.split("### Chapter ")
        chapters = [c for c in chapters if c.strip()]
        for i, chapter_text in enumerate(chapters, start=1):
            parts.append(f"\n\n### Chapter {chapter_text}")
            scene = artifacts.scene_image_paths.get(i)
            if scene:
                parts.append(f"\n\n![Chapter {i} scene]({scene})\n")
    else:
        parts.append(f"{artifacts.final_story}\n")

    return "".join(parts)


def save_markdown(path: str | Path, artifacts: StoryArtifacts) -> Path:
    """Write the markdown rendering of artifacts to disk."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        fh.write(render_markdown(artifacts))
    logger.info("Saved story markdown to %s", out)
    return out
