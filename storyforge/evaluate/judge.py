"""Judge model wrapper + manuscript views for the LLM-judge tiers.

The judge should be *independent* of the drafter (different model) to avoid
self-preference bias. It returns structured JSON; llm.complete_json handles
extraction + retry so a chatty model never crashes the harness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .. import config, llm
from .parse import Chapter, Manuscript

_WORD = re.compile(r"\S+")

JUDGE_SYSTEM = (
    "You are a meticulous, impartial literary editor evaluating a manuscript. "
    "Judge only what is on the page. Be precise and conservative: do not invent "
    "problems, and do not excuse real ones. Always answer in the exact JSON shape requested."
)


@dataclass
class Judge:
    model: str
    cfg: config.RunConfig

    def ask(self, user: str, *, trace_name: str, temperature: float = 0.1,
            max_tokens: int = 2048):
        return llm.complete_json(
            user, system=JUDGE_SYSTEM, model=self.model, cfg=self.cfg,
            temperature=temperature, max_tokens=max_tokens, trace_name=trace_name,
        )


def truncate_words(text: str, limit: int) -> str:
    words = _WORD.findall(text)
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " […]"


def chapter_block(c: Chapter, limit: int) -> str:
    return f"### {c.label}\n{truncate_words(c.prose, limit)}"


def build_view(m: Manuscript, per_chapter_words: int = 1100) -> str:
    """A context-bounded view of the whole manuscript for the judge."""
    parts = [f"# {m.title}", f"## Premise\n{m.premise}"]
    if m.characters:
        cast = "\n".join(f"- {c.canonical}: {c.raw}" for c in m.characters)
        parts.append(f"## Characters\n{cast}")
    parts.append("## Manuscript")
    for c in m.chapters:
        parts.append(chapter_block(c, per_chapter_words))
    return "\n\n".join(parts)
