"""Guarded LLM client over litellm.

Why guarded: the eval spec lists "verbose model truncated -> parse returns None ->
hard crash" as a build-blocking defect. Every model call here retries on
error/empty/parse-failure and never returns None to callers.

Handles qwen3's <think> reasoning blocks (stripped) and robust JSON extraction.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import litellm

from . import config

litellm.drop_params = True  # silently drop params a given provider doesn't accept
litellm.suppress_debug_info = True

log = logging.getLogger("storyforge.llm")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.DOTALL)  # unterminated leading think


class LLMError(RuntimeError):
    """Raised when a model call cannot produce usable output after retries."""


def strip_think(text: str) -> str:
    """Remove qwen3 <think>...</think> reasoning so it never reaches the manuscript."""
    if not text:
        return text
    text = _THINK_RE.sub("", text)
    # Handle a stray closing tag with no opener (rare truncation/streaming artifact).
    if "</think>" in text and "<think>" not in text:
        text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def _messages(system: str | None, user: str) -> list[dict[str, str]]:
    msgs: list[dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs


def complete(
    user: str,
    *,
    system: str | None = None,
    model: str,
    cfg: config.RunConfig,
    temperature: float | None = None,
    max_tokens: int | None = None,
    retries: int = 3,
    trace_name: str = "completion",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Return clean text from the model. Retries on error/empty. Never returns None."""
    temperature = cfg.temperature if temperature is None else temperature
    max_tokens = max_tokens or cfg.max_tokens
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _messages(system, user),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "metadata": {"trace_name": trace_name, **(metadata or {})},
    }
    if config.is_ollama(model):
        kwargs["api_base"] = config.OLLAMA_BASE
        kwargs["num_ctx"] = cfg.num_ctx
        # qwen3 is a thinking model; with reasoning on, short token budgets are
        # consumed by the (separate) thinking stream and content comes back empty.
        # Disable it explicitly for prose/JSON; litellm forwards this to Ollama.
        kwargs["think"] = cfg.think

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            resp = litellm.completion(**kwargs)
            raw = resp.choices[0].message.content or ""
            text = strip_think(raw)
            if text.strip():
                log.debug("%s ok in %.1fs (%d chars) [%s]", trace_name,
                          time.time() - t0, len(text), model)
                return text
            last_err = LLMError("empty content after strip")
            log.warning("%s empty (attempt %d/%d)", trace_name, attempt, retries)
        except Exception as e:  # noqa: BLE001 — surface any provider error as retryable
            last_err = e
            log.warning("%s error (attempt %d/%d): %s", trace_name, attempt, retries,
                        str(e)[:200])
            # Honor a provider's "try again in Xs" rate-limit hint when present.
            hint = re.search(r"try again in ([\d.]+)s", str(e))
            if hint:
                time.sleep(min(float(hint.group(1)) + 0.5, 30))
                continue
        time.sleep(min(2 ** attempt, 8))
    raise LLMError(f"{trace_name} failed after {retries} attempts: {last_err}")


# --- JSON extraction ------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def extract_json(text: str) -> Any:
    """Best-effort parse of JSON embedded in model output. Raises ValueError on miss."""
    text = strip_think(text).strip()
    # 1) direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) fenced ```json ... ```
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    # 3) first balanced object or array
    for o, c in (("{", "}"), ("[", "]")):
        blob = _balanced(text, o, c)
        if blob:
            try:
                return json.loads(blob)
            except Exception:
                continue
    raise ValueError("no parseable JSON found")


def complete_json(
    user: str,
    *,
    system: str | None = None,
    model: str,
    cfg: config.RunConfig,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    retries: int = 3,
    trace_name: str = "json",
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Return parsed JSON from the model. Retries with a corrective nudge on parse miss."""
    sys = (system or "") + "\nRespond with ONLY valid JSON. No prose, no code fences."
    prompt = user
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        text = complete(
            prompt, system=sys, model=model, cfg=cfg, temperature=temperature,
            max_tokens=max_tokens, retries=2, trace_name=trace_name, metadata=metadata,
        )
        try:
            return extract_json(text)
        except ValueError as e:
            last_err = e
            log.warning("%s JSON parse failed (attempt %d/%d)", trace_name, attempt, retries)
            prompt = (
                user
                + "\n\nYour previous reply was not valid JSON. Return ONLY a single "
                "valid JSON value, nothing else."
            )
    raise LLMError(f"{trace_name} produced no valid JSON after {retries} attempts: {last_err}")
