"""Configuration: env loading, model-role resolution, Langfuse wiring.

Local Ollama is the default backend per the goal brief. Everything is overridable
by environment variable or CLI flag, and all three providers (Ollama, Groq,
DeepSeek) are reachable through litellm with the same call shape.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (one level above this package) if present.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()  # also respect a cwd-local .env / real environment

# --- Named model presets (litellm model strings) --------------------------------
# "fast" builds the workflow cheaply; "quality" validates real prose; hosted
# presets are escalation / independent-judge candidates only.
PRESETS = {
    "fast": "ollama_chat/qwen3:latest",
    "quality": "ollama_chat/qwen3.6:27b",
    "groq": os.getenv("SF_GROQ_MODEL", "groq/llama-3.3-70b-versatile"),
    "deepseek": os.getenv("SF_DEEPSEEK_MODEL", "deepseek/deepseek-chat"),
}

# Use `or` chains so an empty-string env var (present but blank in .env) falls
# through to the default rather than yielding "".
OLLAMA_BASE = (os.getenv("SF_OLLAMA_BASE") or os.getenv("LLM_URL")
               or "http://localhost:11434")
# Ollama defaults num_ctx to 4096, which silently truncates our rolling context +
# long chapter output. Pin it generously.
NUM_CTX = int(os.getenv("SF_NUM_CTX", "24576"))
MAX_TOKENS = int(os.getenv("SF_MAX_TOKENS", "8192"))

# Default roles: draft fast, judge with the *stronger, different* model so the
# judge is independent of the drafter (avoids self-preference bias) per the spec.
DEFAULT_DRAFT = os.getenv("SF_DRAFT_MODEL", PRESETS["fast"])
DEFAULT_JUDGE = os.getenv("SF_JUDGE_MODEL", PRESETS["quality"])


def _normalize_model(model: str) -> str:
    """Accept the natural `ollama/` prefix and map it to litellm's chat endpoint.

    This pipeline sends system+user chat messages, so the right litellm provider
    is `ollama_chat/` (the `/api/chat` endpoint); plain `ollama/` would use
    `/api/generate`. Idempotent: an already-correct `ollama_chat/…` is untouched.
    """
    if model.startswith("ollama/"):
        return "ollama_chat/" + model[len("ollama/"):]
    return model


def resolve_model(name: str | None, fallback: str) -> str:
    """Resolve a preset name or a raw `provider/model` litellm string.

    Presets (fast/quality/groq/deepseek) expand to their litellm ids; anything
    else is treated as a litellm id and passed through (after `ollama/`
    normalization). A bare token with no provider is left as-is — litellm will
    error if it isn't a real id.
    """
    raw = PRESETS.get(name, name) if name else fallback
    return _normalize_model(raw)


def is_ollama(model: str) -> bool:
    return model.startswith("ollama")


@dataclass
class RunConfig:
    draft_model: str
    judge_model: str
    num_ctx: int = NUM_CTX
    max_tokens: int = MAX_TOKENS
    temperature: float = 0.8  # prose wants some warmth; planning steps lower it
    think: bool = False  # qwen3 is a thinking model; off by default for speed/parse


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # litellm is extremely chatty and pre-loads optional providers we don't use.
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING if verbose else logging.ERROR)


_LANGFUSE_READY = False


def setup_langfuse() -> bool:
    """Enable Langfuse tracing through litellm if keys are present. Best-effort."""
    global _LANGFUSE_READY
    if _LANGFUSE_READY:
        return True
    pub = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pub and sec):
        return False
    # litellm/langfuse SDK read LANGFUSE_HOST; our .env uses LANGFUSE_BASE_URL.
    base = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")
    if base:
        os.environ["LANGFUSE_HOST"] = base
    try:
        import litellm

        # langfuse v4 is OTEL-based; litellm's "langfuse_otel" callback matches it.
        # The legacy "langfuse" callback targets langfuse v2 and raises on every call
        # under v4 (langfuse.version was renamed to _version).
        if "langfuse_otel" not in (litellm.callbacks or []):
            litellm.callbacks = list(litellm.callbacks or []) + ["langfuse_otel"]
        _LANGFUSE_READY = True
        return True
    except Exception:
        return False
