from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TEXT_OPTIMIZATION_MODULES = [
    "QuestionGenerator",
    "CorePremiseGenerator",
    "SpineTemplateGenerator",
    "WorldBibleQuestionGenerator",
    "WorldBibleGenerator",
    "StoryGenerator",
]


@dataclass
class OptimizationManifest:
    version: int
    created_at: str
    model: str
    modules: dict[str, str]


def _resolve_module_artifact_path(manifest_path: Path, artifact_ref: str) -> Path:
    artifact_path = Path(artifact_ref)
    if artifact_path.is_absolute():
        return artifact_path
    return (manifest_path.parent / artifact_path).resolve()


def load_manifest(manifest_path: str | Path) -> OptimizationManifest:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OptimizationManifest(
        version=int(payload.get("version", 1)),
        created_at=str(payload.get("created_at", "")),
        model=str(payload.get("model", "")),
        modules=dict(payload.get("modules", {})),
    )


def save_manifest(
    manifest_path: str | Path,
    *,
    model: str,
    modules: dict[str, str],
) -> Path:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "model": model,
        "modules": modules,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def try_load_optimized_module(
    module: Any,
    *,
    module_name: str,
    manifest_path: str | Path | None,
    logger: logging.Logger,
) -> bool:
    if not manifest_path:
        return False

    path = Path(manifest_path)
    if not path.exists():
        logger.warning("Optimized manifest not found: %s", path)
        return False

    try:
        manifest = load_manifest(path)
    except Exception as exc:
        logger.warning("Failed to parse optimized manifest %s: %s", path, exc)
        return False

    artifact_ref = manifest.modules.get(module_name)
    if not artifact_ref:
        logger.info("No optimized artifact registered for %s in %s", module_name, path)
        return False

    artifact_path = _resolve_module_artifact_path(path, artifact_ref)
    if not artifact_path.exists():
        logger.warning(
            "Optimized artifact for %s missing: %s", module_name, artifact_path
        )
        return False

    load_fn = getattr(module, "load", None)
    if not callable(load_fn):
        logger.warning(
            "Module %s does not support load(); skipping optimized artifact",
            module_name,
        )
        return False

    try:
        load_fn(str(artifact_path))
    except Exception as exc:
        logger.warning(
            "Failed to load optimized artifact for %s from %s: %s",
            module_name,
            artifact_path,
            exc,
        )
        return False

    logger.info("Loaded optimized artifact for %s from %s", module_name, artifact_path)
    return True
