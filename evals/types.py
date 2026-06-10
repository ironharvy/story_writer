"""Shared types for the LLM-judge eval gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Severity = Literal["info", "warn", "fail"]

_ORDER = {"info": 0, "warn": 1, "fail": 2}


@dataclass
class EvalFinding:
    """One judged issue in a chapter or manuscript.

    ``check`` names the gate; ``severity`` follows the qa.py convention
    (info/warn/fail). ``message`` describes the problem for a human reading the
    report; ``fix`` is the actionable revision guidance fed back to the drafter
    when the reviewed generator decides to re-draft. ``locus`` optionally pins
    the finding to a chapter title or entity name.
    """

    check: str
    severity: Severity
    message: str
    fix: str = ""
    locus: str = ""

    def is_blocking(self) -> bool:
        return self.severity == "fail"


@dataclass
class GateReport:
    """All findings from one gate over one target, plus a convenience verdict."""

    gate: str
    findings: list[EvalFinding] = field(default_factory=list)

    @property
    def blocking(self) -> list[EvalFinding]:
        return [f for f in self.findings if f.severity == "fail"]

    @property
    def warnings(self) -> list[EvalFinding]:
        return [f for f in self.findings if f.severity == "warn"]

    def passed(self) -> bool:
        return not self.blocking

    def revision_brief(self) -> str:
        """Concatenate the actionable fixes for blocking + warning findings into
        a single feedback string the drafter can act on. Empty when clean."""
        lines: list[str] = []
        for f in self.findings:
            if f.severity == "info":
                continue
            tag = "MUST FIX" if f.severity == "fail" else "SHOULD FIX"
            detail = f.fix or f.message
            where = f" [{f.locus}]" if f.locus else ""
            lines.append(f"- ({tag}){where} {detail}")
        return "\n".join(lines)


def worst_severity(findings: list[EvalFinding]) -> Severity:
    """Return the most severe level across ``findings`` (``info`` if empty)."""
    worst: Severity = "info"
    for f in findings:
        if _ORDER[f.severity] > _ORDER[worst]:
            worst = f.severity
    return worst
