"""Regression tests for the LLM-judge (Tier-2) checks.

These need a reachable judge model, so they auto-skip when Ollama is down. They
use the fast local model as judge — these defects are blatant enough that even the
8B model should catch them; the production gate uses the stronger independent judge.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from storyforge import config
from storyforge.evaluate import FAIL, parse, tier2
from storyforge.evaluate.judge import Judge

FIX = Path(__file__).parent / "fixtures"
JUDGE_MODEL = "ollama_chat/qwen3:latest"


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(config.OLLAMA_BASE + "/api/tags", timeout=2)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")


def _judge() -> Judge:
    cfg = config.RunConfig(draft_model=JUDGE_MODEL, judge_model=JUDGE_MODEL,
                           temperature=0.1, think=False)
    return Judge(model=JUDGE_MODEL, cfg=cfg)


def load(name: str) -> parse.Manuscript:
    return parse.parse_manuscript((FIX / name).read_text())


def test_t2_1_pov_lurch_fails():
    c = tier2.t2_1_pov(load("pov_lurch.md"), _judge())
    assert c.severity == FAIL, c.message


def test_t2_2_placeholder_protagonist_fails():
    c = tier2.t2_2_protagonist(load("placeholder_protagonist.md"), _judge())
    assert c.severity == FAIL, c.message


def test_t2_3_duplicate_name_fails():
    c = tier2.t2_3_continuity(load("duplicate_name.md"), _judge())
    assert c.severity == FAIL, c.message


def test_t2_4_premise_drift_fails():
    # A clockmaker story judged against a totally unrelated idea must read as drift.
    idea = "A cat burglar pulls off a heist inside a floating casino on Mars."
    c = tier2.t2_4_premise_fidelity(load("clean_small.md"), idea, _judge())
    assert c.severity == FAIL, c.message
