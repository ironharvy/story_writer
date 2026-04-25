"""
Shared logging configuration for ironharvy's project ecosystem.

Usage:
    from logging_config import setup_logging

    # In your CLI entry point:
    setup_logging(verbosity=1)  # 0=quiet, 1=-v, 2=-vv, 3=-vvv

    # Or from argparse:
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()
    setup_logging(verbosity=args.verbose)

    # Then in any module:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("this shows at -v")
    logger.debug("this shows at -vv")

Environment variables:
    LOG_LEVEL    — Override log level (DEBUG, INFO, WARNING, ERROR)
    LOG_FORMAT   — "json" for structured output, "text" for human-readable
    LOG_FILE     — Optional file path to write logs to (in addition to stderr)

See ADR-003 in think_tank for the full standard.
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


class HumanFormatter(logging.Formatter):
    """Concise, readable format for terminal output."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[1;31m",  # bold red
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        ts = dt.strftime("%Y-%m-%d %H:%M:%S")
        ms = int(record.msecs)
        ts = f"{ts}.{ms:03d}"

        level = record.levelname
        if self.use_color:
            color = self.LEVEL_COLORS.get(level, "")
            level = f"{color}{level}{self.RESET}"

        msg = record.getMessage()

        # Append extra fields (tokens, cost, latency, model, etc.)
        extras = self._format_extras(record)
        if extras:
            msg = f"{msg} {extras}"

        base = f"{ts} {level} [{record.name}] {msg}"
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            base = f"{base}\n{record.exc_text}"
        return base

    @staticmethod
    def _format_extras(record: logging.LogRecord) -> str:
        known_extra_keys = ("model", "tokens_in", "tokens_out", "cost", "latency_ms")
        parts = []
        for key in known_extra_keys:
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
        entry = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Include LLM-specific extras if present
        for key in ("model", "tokens_in", "tokens_out", "cost", "latency_ms"):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val

        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(entry, default=str)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def setup_logging(
    verbosity: int = 0,
    log_level: str | None = None,
    log_format: str | None = None,
    log_file: str | None = None,
) -> None:
    """
    Configure logging for the application.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG (LLM debug), 3=DEBUG (full firehose).
                   Mapped from CLI -v/-vv/-vvv flags.
        log_level: Override level by name. Env var LOG_LEVEL also works.
        log_format: "json" or "text". Env var LOG_FORMAT also works.
                    Default: "text" if stderr is a TTY, "json" otherwise.
        log_file: Optional file path to write logs to (in addition to stderr).
                  Env var LOG_FILE also works. File logging is disabled when
                  neither this argument nor LOG_FILE resolves to a non-empty
                  string — there is no implicit fallback path.
    """
    # Resolve level
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

    # Resolve format
    fmt = log_format or os.environ.get("LOG_FORMAT")
    if fmt is None:
        fmt = "text" if sys.stderr.isatty() else "json"

    # Build formatter
    if fmt == "json":
        formatter = JSONFormatter()
    else:
        use_color = sys.stderr.isatty()
        formatter = HumanFormatter(use_color=use_color)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-init
    root.handlers.clear()

    # Stderr handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    root.addHandler(stderr_handler)

    # Optional file handler — only enabled when log_file is passed explicitly
    # or LOG_FILE is set in the environment. Pass log_file="" or set LOG_FILE=""
    # to disable. No implicit fallback path is used so that library consumers
    # and containerised environments are not surprised by on-disk output.
    file_path = log_file if log_file is not None else os.environ.get("LOG_FILE")
    if file_path:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(HumanFormatter(use_color=False))
        root.addHandler(file_handler)

    # --- Verbosity-based logger tuning ---
    #
    # HTTP transport loggers — these are the noisy ones that drown out
    # useful LLM debug info. Only unleashed at -vvv.
    _http_loggers = ("httpx", "httpcore", "urllib3", "requests", "langfuse")
    #
    # LLM-related loggers — the stuff you actually want at -vv.
    # litellm is what DSPy uses under the hood for all LLM calls.
    _llm_loggers = ("litellm", "dspy", "langfuse", "openai", "anthropic")

    # When LOG_LEVEL is set explicitly, the caller expects that level to apply
    # uniformly — including to third-party loggers. Skip the verbosity-based
    # per-logger tuning so it doesn't silently override the env var.
    if level_str:
        return

    if verbosity <= 0:
        # Quiet: suppress everything except warnings
        for name in _http_loggers + _llm_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    elif verbosity == 1:
        # -v: INFO for app code, suppress third-party debug noise
        for name in _http_loggers + _llm_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    elif verbosity == 2:
        # -vv: LLM debug — let litellm/dspy/langfuse/openai/anthropic
        # through at DEBUG, but keep HTTP transport quiet
        for name in _llm_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in _http_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)

    else:
        # -vvv: Full firehose — everything at DEBUG, including HTTP
        for name in _llm_loggers + _http_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)
        os.environ["LANGFUSE_DEBUG"] = "true"


# ---------------------------------------------------------------------------
# Convenience: LLM call logging helper
# ---------------------------------------------------------------------------


def log_llm_call(
    logger: logging.Logger,
    *,
    model: str,
    tokens_in: int,
    tokens_out: int,
    cost: float | None = None,
    latency_ms: float | None = None,
    message: str = "llm call",
) -> None:
    """
    Log an LLM call with structured fields.

    Example:
        log_llm_call(logger, model="claude-sonnet-4-20250514", tokens_in=500,
                      tokens_out=200, cost=0.003, latency_ms=1100)
    """
    logger.info(
        message,
        extra={
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
            "latency_ms": latency_ms,
        },
    )


# ---------------------------------------------------------------------------
# DSPy callback: automatic token-usage logging for every LM call
# ---------------------------------------------------------------------------

try:
    from dspy.utils.callback import BaseCallback
except ImportError:
    BaseCallback = None


if BaseCallback is not None:

    class TokenUsageCallback(BaseCallback):
        """Log token usage after every DSPy LM call.

        Register via ``dspy.configure(callbacks=[TokenUsageCallback()])``.
        Works with any real LM backend (litellm/Ollama/OpenAI). MockLM
        bypasses the ``@with_callbacks`` decorator so this callback will
        not fire during mock-only test runs — that is expected.
        """

        def __init__(self) -> None:
            self._calls: dict[str, dict[str, Any]] = {}
            self._logger = logging.getLogger("token_usage")

        def on_lm_start(
            self,
            call_id: str,
            instance: Any,
            inputs: dict[str, Any],
        ) -> None:
            del inputs
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
            del outputs
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

            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            cost = entry.get("cost")

            log_llm_call(
                self._logger,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                latency_ms=elapsed_ms,
            )

else:
    TokenUsageCallback = None  # type: ignore[assignment,misc]
