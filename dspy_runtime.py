"""Shared DSPy runtime configuration helpers."""

import logging
import os
from argparse import Namespace
from dataclasses import dataclass

import dspy

from exceptions import RECOVERABLE_RUNTIME_EXCEPTIONS
from logging_config import TokenUsageCallback

try:
    from openinference.instrumentation.dspy import DSPyInstrumentor
except ImportError:
    DSPyInstrumentor = None

logger = logging.getLogger(__name__)


@dataclass
class DSPyConfig:
    """Runtime configuration for DSPy language model setup."""

    model_name: str
    api_base: str | None = None
    api_key: str | None = None
    max_tokens: int = 8000
    num_ctx: int | None = None
    cache: bool = True
    memory_cache: bool = True
    cache_dir: str | None = None


def dspy_config_from_namespace(args: Namespace) -> DSPyConfig:
    """Build DSPy runtime config from parsed CLI arguments."""
    return DSPyConfig(
        model_name=args.model,
        api_base=getattr(args, "llm_url", None),
        api_key=getattr(args, "api_key", None),
        max_tokens=getattr(args, "max_tokens", 8000),
        cache=getattr(args, "cache", True),
        memory_cache=getattr(args, "memory_cache", True),
        cache_dir=getattr(args, "cache_dir", None),
    )


def _log_ollama_runtime(model_name: str, num_ctx: int | None) -> None:
    """If the model is served by ollama, load it with our options and log actual context_length / VRAM."""
    if not model_name.startswith("ollama/"):
        return
    try:
        import urllib.request
        import json as _json

        api_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        ollama_model = model_name.split("/", 1)[1]
        if ":" not in ollama_model:
            ollama_model = f"{ollama_model}:latest"

        # Empty-prompt generate just loads the model with the requested options
        # so /api/ps reflects what the real DSPy calls will hit.
        options = {"num_ctx": num_ctx} if num_ctx is not None else {}
        payload = {
            "model": ollama_model,
            "prompt": "",
            "stream": False,
            "keep_alive": "30s",
            "options": options,
        }
        req = urllib.request.Request(
            f"{api_base}/api/generate",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=60).read()

        with urllib.request.urlopen(f"{api_base}/api/ps", timeout=5) as r:
            ps = _json.loads(r.read())
        for m in ps.get("models", []):
            if m["name"] == ollama_model:
                logger.info(
                    "Ollama runtime model=%s context_length=%d size_vram=%.2fGB",
                    m["name"], m["context_length"], m["size_vram"] / 1e9,
                )
                return
        logger.warning("Ollama probe: model %s not found in /api/ps", ollama_model)
    except RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
        logger.warning("Ollama runtime probe failed: %s", exc)


def configure_dspy(config: DSPyConfig) -> None:
    """Configure DSPy LM and optional instrumentation."""
    kwargs: dict[str, str] = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    elif "openai" in config.model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            kwargs["api_key"] = env_key
        else:
            logger.warning(
                "OPENAI_API_KEY not found in environment variables. "
                "Assuming mock or alternative setup."
            )

    dspy.configure_cache(
        enable_disk_cache=config.cache,
        enable_memory_cache=config.memory_cache,
        disk_cache_dir=config.cache_dir,
    )

    if config.num_ctx is not None:
        kwargs["num_ctx"] = config.num_ctx

    logger.info(
        "Configuring DSPy model=%r max_tokens=%s num_ctx=%s cache=%s memory_cache=%s cache_dir=%r",
        config.model_name,
        config.max_tokens,
        config.num_ctx,
        config.cache,
        config.memory_cache,
        config.cache_dir,
    )
    lm = dspy.LM(
        config.model_name,
        max_tokens=config.max_tokens,
        cache=config.cache,
        **kwargs,
    )
    logger.info("DSPy LM kwargs=%s", lm.kwargs)
    _log_ollama_runtime(config.model_name, config.num_ctx)

    if os.environ.get("LANGFUSE_PUBLIC_KEY") and DSPyInstrumentor is not None:
        try:
            DSPyInstrumentor().instrument()
        except RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Langfuse DSPy callback handler unavailable: %s", exc)

    callbacks = [TokenUsageCallback()] if TokenUsageCallback is not None else []
    dspy.configure(lm=lm, callbacks=callbacks)
