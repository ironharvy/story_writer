from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import dspy

from story_writer.config import ProviderConfig, ollama_host


class PaidProviderBlockedError(RuntimeError):
    pass


def build_lm(config: ProviderConfig, allow_paid: bool = False) -> dspy.LM:
    if config.provider != "ollama" and not allow_paid:
        msg = f"Provider {config.provider!r} requires --allow-paid."
        raise PaidProviderBlockedError(msg)
    if config.provider == "ollama":
        return dspy.LM(
            f"ollama_chat/{config.model}",
            api_base=ollama_host(),
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    return dspy.LM(
        config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


@contextmanager
def stage_lm(config: ProviderConfig, allow_paid: bool = False) -> Iterator[None]:
    with dspy.context(lm=build_lm(config, allow_paid=allow_paid)):
        yield
