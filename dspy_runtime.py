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


_API_KEY_ENV_VARS = ("LLM_API_KEY", "API_KEY", "OPENAI_API_KEY")


def resolve_api_key(explicit: str | None = None) -> str | None:
    """Resolve an API key from explicit value or provider-agnostic env vars.

    Checks LLM_API_KEY, API_KEY, OPENAI_API_KEY in that order.
    """
    if explicit:
        return explicit
    for name in _API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def configure_dspy(config: DSPyConfig) -> None:
    """Configure DSPy LM and optional instrumentation."""
    kwargs: dict[str, str] = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base

    api_key = resolve_api_key(config.api_key)
    if api_key:
        kwargs["api_key"] = api_key
    elif "ollama" not in config.model_name.lower() and config.model_name != "mock":
        logger.warning(
            "No API key found (checked LLM_API_KEY, API_KEY, OPENAI_API_KEY). "
            "Assuming mock or local setup."
        )

    dspy.configure_cache(
        enable_disk_cache=config.cache,
        enable_memory_cache=config.memory_cache,
        disk_cache_dir=config.cache_dir,
    )

    logger.info(
        "Configuring DSPy model=%r max_tokens=%s cache=%s memory_cache=%s cache_dir=%r",
        config.model_name,
        config.max_tokens,
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

    if os.environ.get("LANGFUSE_PUBLIC_KEY") and DSPyInstrumentor is not None:
        try:
            DSPyInstrumentor().instrument()
        except RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Langfuse DSPy callback handler unavailable: %s", exc)

    callbacks = [TokenUsageCallback()] if TokenUsageCallback is not None else []
    dspy.configure(lm=lm, callbacks=callbacks)
