"""DSPy and logging configuration for the simulation package.

Self-contained — no imports from the parent story_writer project.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import dspy

try:
    from openinference.instrumentation.dspy import DSPyInstrumentor
except ImportError:
    DSPyInstrumentor = None

logger = logging.getLogger(__name__)

_RECOVERABLE_RUNTIME_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    KeyError,
    IndexError,
    OSError,
)


# ---------------------------------------------------------------------------
# DSPy Configuration
# ---------------------------------------------------------------------------


@dataclass
class DSPyConfig:
    """Runtime configuration for DSPy language model setup."""

    model_name: str
    api_base: str | None = None
    api_key: str | None = None
    max_tokens: int = 2000
    cache: bool = True
    memory_cache: bool = True
    cache_dir: str | None = None


def configure_dspy(config: DSPyConfig) -> None:
    """Configure DSPy LM and optional instrumentation."""
    model_name = config.model_name
    kwargs: dict[str, Any] = {}
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.api_key is not None:
        kwargs["api_key"] = config.api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment variables. "
                "Assuming mock or alternative setup.",
            )
        else:
            kwargs["api_key"] = env_key

    dspy.configure_cache(
        enable_disk_cache=config.cache,
        enable_memory_cache=config.memory_cache,
        disk_cache_dir=config.cache_dir,
    )

    logger.info(
        "Configuring DSPy model=%r max_tokens=%s cache=%s",
        model_name,
        config.max_tokens,
        config.cache,
    )
    lm = dspy.LM(
        model_name,
        max_tokens=config.max_tokens,
        cache=config.cache,
        **kwargs,
    )

    if os.environ.get("LANGFUSE_PUBLIC_KEY") and DSPyInstrumentor is not None:
        try:
            DSPyInstrumentor().instrument()
        except _RECOVERABLE_RUNTIME_EXCEPTIONS as exc:
            logger.warning("Langfuse DSPy instrumentation unavailable: %s", exc)

    callbacks = [TokenUsageCallback()] if TokenUsageCallback is not None else []
    dspy.configure(lm=lm, callbacks=callbacks)


# ---------------------------------------------------------------------------
# Logging Formatters
# ---------------------------------------------------------------------------


class HumanFormatter(logging.Formatter):
    """Concise, readable format for terminal output."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
        level = record.levelname
        if self.use_color:
            color = self.LEVEL_COLORS.get(level, "")
            level = f"{color}{level:<8}{self.RESET}"
        else:
            level = f"{level:<8}"

        msg = record.getMessage()
        extras = self._format_extras(record)
        if extras:
            msg = f"{msg} {extras}"

        base = f"{ts} [{level}] {record.name}: {msg}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            base = f"{base}\n{record.exc_text}"
        return base

    @staticmethod
    def _format_extras(record: logging.LogRecord) -> str:
        known = ("model", "tokens_in", "tokens_out", "cost", "latency_ms")
        parts = []
        for key in known:
            val = getattr(record, key, None)
            if val is not None:
                if key == "cost":
                    parts.append(f"cost=${val:.4f}")
                elif key == "latency_ms":
                    parts.append(f"latency={val / 1000:.1f}s")
                else:
                    parts.append(f"{key}={val}")
        return " ".join(parts)


class JSONFormatter(logging.Formatter):
    """Structured JSON format for production / piped output."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        entry: dict[str, Any] = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("model", "tokens_in", "tokens_out", "cost", "latency_ms"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------


def setup_logging(
    verbosity: int = 0,
    log_level: str | None = None,
    log_format: str | None = None,
    log_file: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG (LLM debug), 3=DEBUG (firehose).
        log_level: Override level by name. Env var LOG_LEVEL also works.
        log_format: "json" or "text". Env var LOG_FORMAT also works.
        log_file: Optional file path for log output.
    """
    level_str = log_level or os.environ.get("LOG_LEVEL")
    if level_str:
        level = getattr(logging, level_str.upper(), logging.WARNING)
    else:
        level_map = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: logging.DEBUG,
        }
        level = level_map.get(verbosity, logging.DEBUG)

    fmt = log_format or os.environ.get("LOG_FORMAT")
    if fmt is None:
        fmt = "text" if sys.stderr.isatty() else "json"

    formatter: logging.Formatter
    if fmt == "json":
        formatter = JSONFormatter()
    else:
        formatter = HumanFormatter(use_color=sys.stderr.isatty())

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    file_path = log_file if log_file is not None else os.environ.get("LOG_FILE")
    if file_path:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(JSONFormatter())
        root.addHandler(file_handler)

    if level_str:
        return

    _http_loggers = ("httpx", "httpcore", "urllib3", "requests", "langfuse")
    _llm_loggers = ("litellm", "dspy", "langfuse", "openai", "anthropic")

    if verbosity <= 1:
        for name in _http_loggers + _llm_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    elif verbosity == 2:
        for name in _llm_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in _http_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    else:
        for name in _llm_loggers + _http_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Token Usage Callback
# ---------------------------------------------------------------------------


def _log_llm_call(
    target_logger: logging.Logger,
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost: float | None = None,
    latency_ms: float | None = None,
) -> None:
    """Log an LLM call with structured fields."""
    target_logger.info(
        "llm call",
        extra={
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "latency_ms": latency_ms,
        },
    )


try:
    from dspy.utils.callback import BaseCallback
except ImportError:
    BaseCallback = None


if BaseCallback is not None:

    class TokenUsageCallback(BaseCallback):
        """Log token usage after every DSPy LM call."""

        def __init__(self) -> None:
            self._calls: dict[str, dict[str, Any]] = {}
            self._logger = logging.getLogger("token_usage")

        def on_lm_start(
            self,
            call_id: str,
            instance: Any,
            inputs: dict[str, Any],
        ) -> None:
            self._calls[call_id] = {
                "instance": instance,
                "t0": time.perf_counter(),
            }

        def on_lm_end(
            self,
            call_id: str,
            outputs: dict[str, Any] | None,
            exception: Exception | None = None,
        ) -> None:
            call_info = self._calls.pop(call_id, None)
            if exception or call_info is None:
                return

            instance = call_info["instance"]
            elapsed_ms = (time.perf_counter() - call_info["t0"]) * 1000

            if not getattr(instance, "history", None):
                return

            entry = instance.history[-1]
            usage = entry.get("usage") or {}
            model = entry.get("model", getattr(instance, "model", "unknown"))

            _log_llm_call(
                self._logger,
                model=model,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                cost=entry.get("cost"),
                latency_ms=elapsed_ms,
            )

else:
    TokenUsageCallback = None  # type: ignore[assignment,misc]
