from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from story_writer.models.run import Manifest

ModelT = TypeVar("ModelT", bound=BaseModel)


def slugify(value: str, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (slug or "story")[:max_length].strip("-") or "story"


def validate_slug(value: str) -> str:
    slug = value.strip()
    if "/" in slug or "\\" in slug or slug in {".", ".."}:
        msg = "slug must be a single path segment"
        raise ValueError(msg)
    if slug != slugify(slug, max_length=max(len(slug), 1)):
        msg = "slug must contain only lowercase letters, numbers, and hyphens"
        raise ValueError(msg)
    return slug


class RunStore:
    def __init__(self, root: Path = Path("runs")) -> None:
        self.root = root

    def run_dir(self, slug: str) -> Path:
        return self.root / validate_slug(slug)

    def create(
        self,
        slug: str,
        idea: str,
        manifest: Manifest,
        *,
        exist_ok: bool = False,
    ) -> Path:
        path = self.run_dir(slug)
        if path.exists() and not exist_ok:
            msg = f"run {slug!r} already exists; use resume or choose a new --slug"
            raise FileExistsError(msg)
        path.mkdir(parents=True, exist_ok=exist_ok)
        (path / "chapters").mkdir(exist_ok=True)
        (path / "qa").mkdir(exist_ok=True)
        (path / "idea.txt").write_text(idea, encoding="utf-8")
        self.write_manifest(manifest)
        return path

    def read_idea(self, slug: str) -> str:
        return (self.run_dir(slug) / "idea.txt").read_text(encoding="utf-8")

    def manifest_path(self, slug: str) -> Path:
        return self.run_dir(slug) / "manifest.json"

    def read_manifest(self, slug: str) -> Manifest:
        return Manifest.model_validate_json(
            self.manifest_path(slug).read_text(encoding="utf-8")
        )

    def write_manifest(self, manifest: Manifest) -> None:
        path = self.manifest_path(manifest.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def artifact_path(self, slug: str, name: str) -> Path:
        if name.startswith("chapter:"):
            number = int(name.split(":", maxsplit=1)[1])
            return self.run_dir(slug) / "chapters" / f"{number:02d}.json"
        if name.startswith("qa:"):
            number = int(name.split(":", maxsplit=1)[1])
            return self.run_dir(slug) / "qa" / f"{number:02d}.json"
        return self.run_dir(slug) / f"{name}.json"

    def has_artifact(self, slug: str, name: str) -> bool:
        return self.artifact_path(slug, name).exists()

    def write_model(self, slug: str, name: str, model: BaseModel) -> None:
        path = self.artifact_path(slug, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")

    def read_model(self, slug: str, name: str, model_type: type[ModelT]) -> ModelT:
        text = self.artifact_path(slug, name).read_text(encoding="utf-8")
        return model_type.model_validate_json(text)

    def write_json(self, slug: str, name: str, data: Any) -> None:
        path = self.artifact_path(slug, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def read_json(self, slug: str, name: str) -> Any:
        return json.loads(self.artifact_path(slug, name).read_text(encoding="utf-8"))
