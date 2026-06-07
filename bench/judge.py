"""LLM-judge for the Tier-2 and Tier-3 checks in ``bench/eval-spec.md``.

Tier-1 (``bench/criteria.py``) is deterministic and runs on every draft. This
module adds the checks that need a model:

- **T2.1 POV / narrator consistency** — reuses :mod:`pov_check`'s classifier
  rather than re-implementing it.
- **T2.2 protagonist named + consistent**, **T2.3 contradictions**,
  **T2.4 premise fidelity** — DSPy signatures whose instructions are taken
  verbatim from the eval-spec sections.
- **Tier-3 qualitative rubric** — the 1–5 axes from ``bench/rubric.md``.

The judge uses the globally-configured DSPy LM, so configure it with the judge
model (an independent local Ollama model is recommended, to avoid
self-preference bias) before calling :func:`judge_story`. Wiring lives in
``bench/score.py`` behind ``--llm-judge`` so the deterministic harness stays
fast by default.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import dspy

import pov_check
import qa
from qa import Finding

logger = logging.getLogger(__name__)

# Tier-3 axes, in the order of bench/rubric.md.
RUBRIC_AXES = (
    "arc_structure",
    "ending",
    "character_agency",
    "scene_vs_summary",
    "prose_quality",
    "cohesion",
)


@dataclass
class JudgeReport:
    """Tier-2 findings + the Tier-3 rubric for one manuscript."""

    findings: list[Finding] = field(default_factory=list)
    rubric: dict[str, dict] = field(default_factory=dict)


# --- signatures (instructions verbatim from bench/eval-spec.md) --------------


class ProtagonistNaming(dspy.Signature):
    """T2.2 — protagonist is named and named consistently.

    The protagonist must be referred to by a proper name in the prose. It is a
    failure if chapters lean on "the protagonist" / "the child" / "the boy" /
    "the woman" as the primary referent (the classic cold-open defect). It is a
    failure if the protagonist is called by two different names across chapters.
    Judge the narration only — ignore names that appear only inside quoted
    dialogue, letters, or reported speech.
    """

    chapters_text: str = dspy.InputField(
        desc="The story's chapters, each as '### <label>' then its prose."
    )
    protagonist_name: str = dspy.OutputField(
        desc="The protagonist's primary proper name, or '' if none is used."
    )
    relies_on_generic_referent: bool = dspy.OutputField(
        desc="True if a generic referent (the boy/the woman/the protagonist) "
        "is the primary referent for the protagonist."
    )
    inconsistent_names: list[str] = dspy.OutputField(
        desc="Distinct names used for the one protagonist across chapters, if "
        "more than one are used; otherwise an empty list."
    )
    note: str = dspy.OutputField(desc="One-phrase justification.")


class Contradictions(dspy.Signature):
    """T2.3 — continuity / no contradictions.

    List factual contradictions across the manuscript: a character dead in ch3
    acting in ch5, eye/hair colour or relationships that flip, a timeline that
    doesn't add up, two distinct characters sharing one name, locations that
    teleport. Hard contradictions are failures; minor wobble is a warning.
    Judge plot facts and narration, not stylistic choices.
    """

    manuscript: str = dspy.InputField(desc="The full manuscript prose, chapter by chapter.")
    hard_contradictions: list[str] = dspy.OutputField(
        desc="Hard factual contradictions, each a short phrase. Empty if none."
    )
    minor_issues: list[str] = dspy.OutputField(
        desc="Minor continuity wobble (warnings), each a short phrase. Empty if none."
    )


class PremiseFidelity(dspy.Signature):
    """T2.4 — premise fidelity.

    Given the user's original idea and the manuscript: does the story actually
    dramatize THIS premise, or did it drift into a generic tale? A manuscript
    that ignores the user's core idea is a failure regardless of prose quality.
    """

    idea: str = dspy.InputField(desc="The user's original story idea.")
    manuscript: str = dspy.InputField(desc="The full manuscript prose.")
    dramatizes_premise: bool = dspy.OutputField(
        desc="True if the story dramatizes the user's core idea."
    )
    note: str = dspy.OutputField(desc="One-phrase justification of the verdict.")


class QualitativeRubric(dspy.Signature):
    """Tier-3 — qualitative rubric, scoring each axis 1 (low) to 5 (high).

    - arc_structure: 1 = episodic, no shape; 5 = clear setup -> rising action
      -> climax -> resolution.
    - ending: 1 = stops / fizzles / deus ex machina; 5 = earned, lands the
      premise's promise.
    - character_agency: 1 = events happen to passive figures; 5 = choices drive
      the plot, characters want things.
    - scene_vs_summary: 1 = tells/summarizes throughout; 5 = dramatizes key
      beats in scene.
    - prose_quality: 1 = flat, repetitive, garbled; 5 = varied, controlled,
      few tics.
    - cohesion: 1 = threads dropped, setups unpaid; 5 = setups pay off, threads
      resolve.

    Judge the narration only; quoted dialogue shouldn't confound the
    prose-quality or scene-vs-summary axes.
    """

    manuscript: str = dspy.InputField(desc="The full manuscript prose.")
    scores: dict[str, int] = dspy.OutputField(
        desc="One 1-5 integer per axis, keyed by: arc_structure, ending, "
        "character_agency, scene_vs_summary, prose_quality, cohesion."
    )
    notes: dict[str, str] = dspy.OutputField(
        desc="One short note per axis, using the same keys as scores."
    )


# --- finding interpretation (pure; unit-tested without an LM) ----------------


def _protagonist_findings(verdict: object) -> list[Finding]:
    """Map a ProtagonistNaming verdict onto T2.2 findings."""
    name = (getattr(verdict, "protagonist_name", "") or "").strip()
    generic = bool(getattr(verdict, "relies_on_generic_referent", False))
    names = {str(n).strip() for n in (getattr(verdict, "inconsistent_names", []) or []) if str(n).strip()}
    note = (getattr(verdict, "note", "") or "").strip()
    suffix = f" ({note})" if note else ""
    findings: list[Finding] = []
    if generic or not name:
        findings.append(Finding("protagonist_naming", "fail",
                                f"protagonist is unnamed or relies on a generic referent{suffix}"))
    if len(names) >= 2:
        findings.append(Finding("protagonist_naming", "fail",
                                f"protagonist called by multiple names across chapters: {sorted(names)}"))
    if not findings:
        findings.append(Finding("protagonist_naming", "info",
                                f"protagonist named consistently: {name!r}"))
    return findings


def _contradiction_findings(verdict: object) -> list[Finding]:
    """Map a Contradictions verdict onto T2.3 findings."""
    hard = [str(x).strip() for x in (getattr(verdict, "hard_contradictions", []) or []) if str(x).strip()]
    minor = [str(x).strip() for x in (getattr(verdict, "minor_issues", []) or []) if str(x).strip()]
    findings = [Finding("contradictions", "fail", f"hard contradiction: {h}") for h in hard]
    findings += [Finding("contradictions", "warn", f"minor continuity issue: {m}") for m in minor]
    if not findings:
        findings.append(Finding("contradictions", "info", "no contradictions found"))
    return findings


def _premise_findings(verdict: object) -> list[Finding]:
    """Map a PremiseFidelity verdict onto T2.4 findings."""
    ok = bool(getattr(verdict, "dramatizes_premise", True))
    note = (getattr(verdict, "note", "") or "").strip()
    suffix = f" ({note})" if note else ""
    if not ok:
        return [Finding("premise_fidelity", "fail", f"manuscript drifts off the user's idea{suffix}")]
    return [Finding("premise_fidelity", "info", f"dramatizes the user's premise{suffix}")]


def _coerce_rubric(scores: object, notes: object) -> dict[str, dict]:
    """Build {axis: {score: 1-5, note}} from a QualitativeRubric verdict."""
    scores = scores if isinstance(scores, dict) else {}
    notes = notes if isinstance(notes, dict) else {}
    out: dict[str, dict] = {}
    for axis in RUBRIC_AXES:
        if axis not in scores:
            logger.warning("judge: rubric missing axis %r", axis)
            continue
        try:
            score = int(scores[axis])
        except (TypeError, ValueError):
            logger.warning("judge: rubric axis %r has non-int score %r", axis, scores[axis])
            continue
        out[axis] = {"score": max(1, min(5, score)), "note": str(notes.get(axis, "")).strip()}
    return out


# --- LM-calling wrappers + orchestration ------------------------------------


def _manuscript(chapters: dict[str, str]) -> str:
    """Render the chapters as narration-only text for the judge."""
    return "\n\n".join(f"### {title}\n\n{body}" for title, body in chapters.items())


def judge_protagonist(chapters_text: str) -> list[Finding]:
    return _protagonist_findings(dspy.ChainOfThought(ProtagonistNaming)(chapters_text=chapters_text))


def judge_contradictions(manuscript: str) -> list[Finding]:
    return _contradiction_findings(dspy.ChainOfThought(Contradictions)(manuscript=manuscript))


def judge_premise(idea: str, manuscript: str) -> list[Finding]:
    if not idea.strip():
        return [Finding("premise_fidelity", "info", "skipped (no original idea available)")]
    return _premise_findings(dspy.ChainOfThought(PremiseFidelity)(idea=idea, manuscript=manuscript))


def judge_rubric(manuscript: str) -> dict[str, dict]:
    verdict = dspy.ChainOfThought(QualitativeRubric)(manuscript=manuscript)
    return _coerce_rubric(getattr(verdict, "scores", {}), getattr(verdict, "notes", {}))


def judge_story(story_md_path: Path, idea: str = "") -> JudgeReport:
    """Run every Tier-2 + Tier-3 LLM check against a story markdown artifact.

    Uses the globally-configured DSPy LM as the judge (configure it with the
    judge model first). ``idea`` is the user's original prompt, fed to the T2.4
    premise-fidelity check; pass ``""`` to skip that one.
    """
    text = Path(story_md_path).read_text(encoding="utf-8")
    chapters = qa.split_chapters(text)
    if not chapters:
        return JudgeReport(findings=[Finding("judge", "warn", "no chapters found to judge")])
    manuscript = _manuscript(chapters)
    findings = list(pov_check.check_pov_consistency(chapters))  # T2.1
    findings += judge_protagonist(manuscript)                   # T2.2
    findings += judge_contradictions(manuscript)                # T2.3
    findings += judge_premise(idea, manuscript)                 # T2.4
    return JudgeReport(findings=findings, rubric=judge_rubric(manuscript))
