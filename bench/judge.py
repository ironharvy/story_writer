"""LM-judge canon/continuity metrics (Tier-2 of bench/eval-spec.md).

These mirror :mod:`pov_check` and :mod:`story_linter`: a ``dspy.ChainOfThought``
reads the manuscript and the canon, returns a structured verdict, and the verdict
is coerced into :class:`qa.Finding` records so it folds into the same
info/warn/fail report.

The heuristics in :mod:`canon_metrics` can't read for sense; these judges can:

* :func:`judge_appearance` — is the POV character physically described at all,
  and does any description contradict ``locked_appearance``?
* :func:`judge_invariant`  — a semantic ``canon_invariant`` (Azazel stays in the
  Maw after Ch.11; the Command-of-the-Void leash frays; Kira is not a strawman).
* :func:`judge_dangling_payoff` — setups that are never paid off.

Every check takes an optional pre-computed ``verdict``/``verdicts`` argument so it
can be unit-tested without an LM (exactly as ``pov_check.check_pov_consistency``
accepts ``classifications``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from qa import Finding, split_chapters

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    """One LM judgement: did the manuscript satisfy the rule?"""

    holds: bool          # True = the canon rule is satisfied
    evidence: str = ""   # short justification / the offending detail


def _manuscript_text(text: str) -> str:
    chapters = split_chapters(text)
    return "\n\n".join(f"### {title}\n\n{body}" for title, body in chapters.items())


def _coerce(item: object) -> Verdict:
    """Coerce a DSPy output into a :class:`Verdict`."""
    if isinstance(item, Verdict):
        return item
    if isinstance(item, dict):
        holds = item.get("holds")
        evidence = item.get("evidence") or item.get("note") or ""
    else:
        holds = getattr(item, "holds", None)
        evidence = getattr(item, "evidence", "") or ""
    return Verdict(holds=bool(holds), evidence=str(evidence).strip())


def judge_rule(manuscript: str, rule: str) -> Verdict:
    """Ask the configured DSPy LM whether ``rule`` holds over ``manuscript``.

    Raises whatever the underlying DSPy call raises — failures stay visible.
    """
    import dspy

    class JudgeCanonRule(dspy.Signature):
        """Judge whether a manuscript satisfies a single canon/continuity rule.

        You are given the chapters of one story (each introduced by a
        ``### Chapter N: ...`` heading) and one rule that the story must satisfy.
        Read for sense — consider only the narration and events, ignoring what is
        said inside quoted dialogue when the rule is about facts of the world.
        Decide whether the rule HOLDS. Set ``holds`` true only if the manuscript
        clearly satisfies the rule; set it false if the manuscript violates it.
        ``evidence`` is a short justification naming the chapter and the offending
        (or confirming) detail.
        """

        rule: str = dspy.InputField(desc="The canon/continuity rule to check")
        manuscript: str = dspy.InputField(desc="The story's chapters with headings")
        holds: bool = dspy.OutputField(desc="True if the rule is satisfied")
        evidence: str = dspy.OutputField(desc="Short justification (chapter + detail)")

    result = dspy.ChainOfThought(JudgeCanonRule)(rule=rule, manuscript=manuscript)
    return _coerce(result)


def _finding(check: str, verdict: Verdict, fail_severity: str = "fail") -> Finding:
    if verdict.holds:
        return Finding(check=check, severity="info",
                       message=f"{check}: ok" + (f" ({verdict.evidence})" if verdict.evidence else ""))
    suffix = f" ({verdict.evidence})" if verdict.evidence else ""
    return Finding(check=check, severity=fail_severity, message=f"{check}: violated{suffix}")


# --- appearance --------------------------------------------------------------

def _appearance_spec(canon) -> str:
    proto = (getattr(canon, "locked_appearance", {}) or {}).get("protagonist", {})
    parts = [f"{k}: {proto[k]}" for k in ("eyes", "hair", "hair_style", "build") if proto.get(k)]
    marks = proto.get("permanent_marks") or []
    if marks:
        parts.append("permanent mark(s): " + ", ".join(marks))
    return "; ".join(parts)


def judge_appearance(
    text: str, canon, described: Verdict | None = None, consistent: Verdict | None = None
) -> list[Finding]:
    """Two findings: the POV character must be physically described at all, and
    any description must not contradict ``locked_appearance``."""
    manuscript = _manuscript_text(text)
    spec = _appearance_spec(canon)

    if described is None:
        described = judge_rule(
            manuscript,
            "The point-of-view / protagonist character is given concrete physical "
            "description somewhere in the prose (not left a faceless cipher).",
        )
    findings = [_finding("appearance_described", described)]

    if spec:
        if consistent is None:
            consistent = judge_rule(
                manuscript,
                f"Every physical description of the protagonist is consistent with "
                f"this locked appearance and never contradicts it: {spec}. "
                f"Damage may accumulate but there is no transformation.",
            )
        findings.append(_finding("appearance_vs_spec", consistent))
    return findings


# --- semantic invariants -----------------------------------------------------

def judge_invariants(text: str, canon, verdicts: dict[str, Verdict] | None = None) -> list[Finding]:
    """One finding per ``canon_invariant``; ``verdicts`` maps invariant id ->
    Verdict for testing without an LM."""
    manuscript = _manuscript_text(text)
    invariants = getattr(canon, "canon_invariants", []) or []
    findings: list[Finding] = []
    for inv in invariants:
        inv_id = inv.get("id", "invariant")
        rule = inv.get("rule", "")
        verdict = (verdicts or {}).get(inv_id)
        if verdict is None:
            verdict = judge_rule(manuscript, rule)
        findings.append(_finding(inv_id, verdict))
    return findings


# --- dangling setup/payoff ---------------------------------------------------

def judge_dangling_payoff(text: str, verdict: Verdict | None = None) -> list[Finding]:
    """Flag setups introduced and never paid off (a dangling thread)."""
    manuscript = _manuscript_text(text)
    if verdict is None:
        verdict = judge_rule(
            manuscript,
            "Every setup or promise the story makes (an introduced object, threat, "
            "question, or named plan) is paid off or resolved by the end; no major "
            "thread is left dangling.",
        )
    return [_finding("dangling_payoff", verdict, fail_severity="warn")]


# --- runner ------------------------------------------------------------------

def run_all(text: str, canon, verdicts: dict[str, Verdict] | None = None) -> list[Finding]:
    """Run every LM judge. ``verdicts`` (keyed by check/invariant id) short-circuits
    the LM for any provided verdict — used by tests and to compose cached runs."""
    verdicts = verdicts or {}
    out: list[Finding] = []
    out.extend(judge_appearance(
        text, canon,
        described=verdicts.get("appearance_described"),
        consistent=verdicts.get("appearance_vs_spec"),
    ))
    out.extend(judge_invariants(text, canon, verdicts=verdicts))
    out.extend(judge_dangling_payoff(text, verdict=verdicts.get("dangling_payoff")))
    return out


def run(path: Path, canon) -> list[Finding]:
    """Run the LM judges on a story markdown file (calls the LM)."""
    return run_all(Path(path).read_text(encoding="utf-8"), canon)
