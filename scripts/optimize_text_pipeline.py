#!/usr/bin/env python3
"""Generate and persist DSPy text-pipeline optimization artifacts."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from dspy_optimization import TEXT_OPTIMIZATION_MODULES, save_manifest
from main import configure_dspy, dspy_config_from_namespace, initialize_text_generators


logger = logging.getLogger(__name__)


def _parse_modules(raw: str | None) -> list[str]:
    if not raw:
        return list(TEXT_OPTIMIZATION_MODULES)

    requested = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(TEXT_OPTIMIZATION_MODULES))
    if unknown:
        raise ValueError(
            "Unknown module(s): "
            + ", ".join(unknown)
            + ". Allowed: "
            + ", ".join(TEXT_OPTIMIZATION_MODULES)
        )
    return requested


def _optimize_or_snapshot_module(module_name: str, module: dspy.Module, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{module_name}.json"
    module.save(str(artifact_path))
    logger.info("Saved optimized artifact for %s -> %s", module_name, artifact_path)
    return artifact_path


def main() -> int:
    """Parse CLI args and save selected optimized module artifacts."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Compile/save DSPy text pipeline module artifacts.",
    )
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "openai/gpt-4o-mini"))
    parser.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY"))
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("DSPY_CACHE_DIR"))
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Comma-separated subset of modules to optimize. Defaults to all text modules.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".tmp/dspy_optimized/text_pipeline",
        help="Directory for per-module artifacts.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=".tmp/dspy_optimized/text_pipeline_manifest.json",
        help="Output manifest JSON path.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    configure_dspy(dspy_config_from_namespace(args))

    selected_modules = _parse_modules(args.modules)
    generators = initialize_text_generators(use_optimized=False)

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)

    manifest_modules: dict[str, str] = {}
    for module_name in selected_modules:
        module = generators[module_name]
        artifact_path = _optimize_or_snapshot_module(module_name, module, output_dir)
        manifest_modules[module_name] = str(artifact_path)

    save_manifest(
        manifest_path,
        model=args.model,
        modules=manifest_modules,
    )
    logger.info("Wrote manifest to %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
