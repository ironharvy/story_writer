"""Evaluation harness implementing STORY_EVAL_SPEC.md.

Tier 1 = deterministic (no LLM). Tier 2 = LLM-judge gates. Tier 3 = LLM-judge
rubric. `harness.evaluate()` runs them and emits a machine-readable scorecard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    """One evaluation check result: id -> severity + message (+ optional details)."""

    id: str
    name: str
    severity: str  # PASS | WARN | FAIL
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"id": self.id, "name": self.name, "severity": self.severity,
             "message": self.message}
        if self.details:
            d["details"] = self.details
        return d
