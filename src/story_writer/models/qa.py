from __future__ import annotations

from pydantic import BaseModel, Field


class RuleResult(BaseModel):
    rule_id: str
    passed: bool
    severity: str
    message: str
    spans: list[str] = Field(default_factory=list)
