"""LLM-backed Tier-2 / Tier-3 judges for the story scorecard.

Implements the parts of ``bench/eval-spec.md`` that "heuristics can't read":

- **T2.3 continuity** — factual contradictions across the manuscript
  (timeline, trait flips, name collisions, teleporting locations). Hard
  contradictions FAIL; minor wobble WARNs.
- **T2.4 premise fidelity** — does the story dramatize the user's original
  idea, or drift into a generic tale? Off-premise FAILs.
- **Tier-3 rubric** — the six craft axes (arc, ending, agency, scene-vs-
  summary, prose, cohesion) scored 1–5. Below the ship threshold FAILs.

Each judge follows the :mod:`pov_check` template: the DSPy signature is nested
in the calling function so ``import dspy`` stays lazy and this module is
importable (and the gates testable) without a model. Public ``check_*``
functions accept precomputed results so callers/tests can inject a verdict
instead of calling the LM.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from qa import Finding

logger = logging.getLogger(__name__)

RUBRIC_AXES = (
    "arc_structure",
    "ending",
    "character_agency",
    "scene_vs_summary",
    "prose_quality",
    "cohesion",
)
RUBRIC_MIN_AVG = 4.0          # ship threshold (eval-spec Tier-3)
RUBRIC_MIN_AXIS = 3           # no single axis may fall below this
PREMISE_MIN_SCORE = 3         # below this the story has drifted off-idea


@dataclass
class Contradiction:
    """One factual contradiction found in the manuscript."""

    kind: str       # timeline | trait | name_collision | location | death | ...
    severity: str   # "hard" | "minor"
    detail: str


@dataclass
class PremiseVerdict:
    """Whether the manuscript dramatizes the user's original idea."""

    dramatizes: bool
    score: int      # 1–5 fidelity
    note: str = ""


@dataclass
class AxisScore:
    """A single Tier-3 rubric axis score."""

    axis: str       # one of RUBRIC_AXES
    score: int      # 1–5
    note: str = ""


def _format_chapters(chapters: dict[str, str]) -> str:
    return "\n\n".join(f"### {title}\n\n{body}" for title, body in chapters.items())


def _note(text: str) -> str:
    return f" — {text}" if text else ""


# --- coercion (robust to dict / object / dataclass LM outputs) ---------------

def _coerce_contradiction(item: object) -> Contradiction | None:
    if isinstance(item, Contradiction):
        kind, severity, detail = item.kind, item.severity, item.detail
    elif isinstance(item, dict):
        kind = item.get("kind") or ""
        severity = item.get("severity") or ""
        detail = item.get("detail") or item.get("description") or ""
    else:
        kind = getattr(item, "kind", "") or ""
        severity = getattr(item, "severity", "") or ""
        detail = getattr(item, "detail", "") or ""
    detail = str(detail).strip()
    if not detail:
        return None
    severity = str(severity).strip().lower()
    if severity not in ("hard", "minor"):
        severity = "hard"  # surface unknown severities rather than hide them
    return Contradiction(kind=str(kind).strip() or "unspecified", severity=severity, detail=detail)


def _coerce_premise(obj: object) -> PremiseVerdict:
    if isinstance(obj, PremiseVerdict):
        return obj
    if isinstance(obj, dict):
        dramatizes, score, note = obj.get("dramatizes"), obj.get("score"), obj.get("note") or ""
    else:
        dramatizes = getattr(obj, "dramatizes", None)
        score = getattr(obj, "score", None)
        note = getattr(obj, "note", "") or ""
    try:
        score_int = max(0, min(5, int(score)))
    except (TypeError, ValueError):
        score_int = 0
    return PremiseVerdict(dramatizes=bool(dramatizes), score=score_int, note=str(note).strip())


def _coerce_axis(item: object) -> AxisScore | None:
    if isinstance(item, AxisScore):
        axis, score, note = item.axis, item.score, item.note
    elif isinstance(item, dict):
        axis, score, note = item.get("axis") or "", item.get("score"), item.get("note") or ""
    else:
        axis = getattr(item, "axis", "") or ""
        score = getattr(item, "score", None)
        note = getattr(item, "note", "") or ""
    axis = str(axis).strip().lower().replace(" ", "_").replace("-", "_")
    if axis not in RUBRIC_AXES:
        logger.warning("judge: unknown rubric axis %r; dropping", axis)
        return None
    try:
        score_int = max(1, min(5, int(score)))
    except (TypeError, ValueError):
        logger.warning("judge: non-int score %r for axis %r; dropping", score, axis)
        return None
    return AxisScore(axis=axis, score=score_int, note=str(note).strip())


# --- T2.3 continuity --------------------------------------------------------

def find_contradictions(cast: str, chapters: dict[str, str]) -> list[Contradiction]:
    """Ask the LM for factual contradictions across the manuscript."""
    import dspy

    class FindContradictions(dspy.Signature):
        """List factual contradictions across a manuscript.

        Given the cast list and the chapters in order, identify facts that
        cannot both be true: a character dead/absent then acting; physical
        traits or relationships that flip; a timeline that doesn't add up; two
        *distinct* characters sharing one name (a collision, not a nickname); a
        location or object that teleports. Judge explicit facts only — do not
        invent contradictions from deliberate mystery or normal scene changes.
        Classify each as 'hard' (a real logical contradiction) or 'minor' (a
        forgivable wobble). Return an empty list if the manuscript is consistent.
        """

        cast: str = dspy.InputField(desc="Canonical cast list (names + descriptions).")
        chapters_text: str = dspy.InputField(desc="Chapters in order: '### <label>' then prose.")
        contradictions: list[Contradiction] = dspy.OutputField(
            desc="Factual contradictions; [] if consistent."
        )

    result = dspy.ChainOfThought(FindContradictions)(
        cast=cast, chapters_text=_format_chapters(chapters)
    )
    out: list[Contradiction] = []
    for item in result.contradictions or []:
        coerced = _coerce_contradiction(item)
        if coerced is None:
            logger.warning("judge: dropping malformed contradiction: %r", item)
            continue
        out.append(coerced)
    return out


def check_continuity(
    chapters: dict[str, str],
    cast: str,
    contradictions: list[Contradiction] | None = None,
) -> list[Finding]:
    """Hard contradictions FAIL; minor ones WARN. ``contradictions`` injects."""
    if not chapters:
        return []
    if contradictions is None:
        contradictions = find_contradictions(cast, chapters)
    findings = [
        Finding(
            check="continuity",
            severity="fail" if c.severity == "hard" else "warn",
            message=f"{c.kind}: {c.detail}",
        )
        for c in contradictions
    ]
    if not findings:
        findings.append(Finding(
            check="continuity", severity="info", message="no contradictions found"
        ))
    return findings


# --- T2.4 premise fidelity --------------------------------------------------

def judge_premise_fidelity(idea: str, chapters: dict[str, str]) -> PremiseVerdict:
    """Ask the LM whether the manuscript dramatizes the user's idea."""
    import dspy

    class JudgePremiseFidelity(dspy.Signature):
        """Decide whether a manuscript dramatizes the user's original idea.

        Given the one-line story idea the user asked for and the full
        manuscript, decide whether the story dramatizes *that premise* — its
        central conflict, world, and stakes — versus drifting into a generic
        tale that ignores the idea. Detail may diverge; what matters is that the
        user's core idea is on the page and load-bearing. Return ``dramatizes``
        (True/False), a 1–5 ``score`` (5 = the idea is the spine of the story;
        1 = absent), and a one-phrase note.
        """

        idea: str = dspy.InputField(desc="The user's original one-line story idea.")
        manuscript: str = dspy.InputField(desc="The full manuscript prose.")
        verdict: PremiseVerdict = dspy.OutputField(desc="dramatizes / score 1-5 / note.")

    result = dspy.ChainOfThought(JudgePremiseFidelity)(
        idea=idea, manuscript=_format_chapters(chapters)
    )
    return _coerce_premise(result.verdict)


def check_premise_fidelity(
    idea: str,
    chapters: dict[str, str],
    verdict: PremiseVerdict | None = None,
) -> list[Finding]:
    """Off-premise (not dramatized or score < threshold) FAILs. ``verdict`` injects."""
    if not chapters:
        return []
    if verdict is None:
        verdict = judge_premise_fidelity(idea, chapters)
    if verdict.dramatizes and verdict.score >= PREMISE_MIN_SCORE:
        return [Finding(
            check="premise_fidelity",
            severity="info",
            message=f"dramatizes premise (score {verdict.score}/5){_note(verdict.note)}",
        )]
    return [Finding(
        check="premise_fidelity",
        severity="fail",
        message=f"drifts off-premise (score {verdict.score}/5){_note(verdict.note)}",
    )]


# --- Tier-3 rubric ----------------------------------------------------------

def _score_rubric_once(chapters: dict[str, str]) -> list[AxisScore]:
    import dspy

    class ScoreRubric(dspy.Signature):
        """Score a manuscript on six craft axes, 1 (worst) to 5 (best).

        Judge the *narration* (ignore the quality of quoted dialogue). Score
        each axis independently against its anchors:
        - arc_structure: 1 episodic/shapeless ... 5 setup->rising->climax->resolution
        - ending: 1 stops/fizzles/deus-ex-machina ... 5 earned, lands the promise
        - character_agency: 1 events happen to passive figures ... 5 choices drive plot
        - scene_vs_summary: 1 tells/summarizes ... 5 dramatizes key beats in scene
        - prose_quality: 1 flat/repetitive/garbled ... 5 varied, controlled, few tics
        - cohesion: 1 threads dropped/setups unpaid ... 5 setups pay off, threads resolve

        Be calibrated and critical: reserve 5 for genuinely excellent work and 1
        for broken work. Return one entry per axis with a one-phrase note.
        """

        manuscript: str = dspy.InputField(desc="The full manuscript prose.")
        scores: list[AxisScore] = dspy.OutputField(
            desc="One {axis, score 1-5, note} per rubric axis."
        )

    result = dspy.ChainOfThought(ScoreRubric)(manuscript=_format_chapters(chapters))
    out: list[AxisScore] = []
    for item in result.scores or []:
        coerced = _coerce_axis(item)
        if coerced is not None:
            out.append(coerced)
    return out


def _median_axes(runs: list[list[AxisScore]]) -> list[AxisScore]:
    """Aggregate repeated rubric runs by per-axis median (variance control)."""
    from statistics import median

    scores: dict[str, list[int]] = {}
    notes: dict[str, str] = {}
    for run in runs:
        for axis_score in run:
            scores.setdefault(axis_score.axis, []).append(axis_score.score)
            notes.setdefault(axis_score.axis, axis_score.note)
    return [
        AxisScore(axis=axis, score=max(1, min(5, round(median(scores[axis])))), note=notes.get(axis, ""))
        for axis in RUBRIC_AXES
        if axis in scores
    ]


def score_rubric(chapters: dict[str, str], samples: int = 1) -> list[AxisScore]:
    """Score the rubric; with ``samples`` > 1, take the per-axis median."""
    if not chapters:
        return []
    runs = [_score_rubric_once(chapters) for _ in range(max(1, samples))]
    return runs[0] if len(runs) == 1 else _median_axes(runs)


def check_rubric(
    chapters: dict[str, str],
    scores: list[AxisScore] | None = None,
    samples: int = 1,
    min_avg: float = RUBRIC_MIN_AVG,
    min_axis: int = RUBRIC_MIN_AXIS,
) -> list[Finding]:
    """FAIL if rubric average < ``min_avg`` or any axis < ``min_axis``."""
    if not chapters:
        return []
    if scores is None:
        scores = score_rubric(chapters, samples=samples)
    if not scores:
        return [Finding(check="rubric", severity="warn", message="rubric judge returned no scores")]
    by_axis = {s.axis: s.score for s in scores}
    avg = sum(by_axis.values()) / len(by_axis)
    below = sorted(axis for axis, score in by_axis.items() if score < min_axis)
    detail = ", ".join(f"{s.axis}={s.score}" for s in scores)
    if avg < min_avg or below:
        return [Finding(
            check="rubric",
            severity="fail",
            message=f"avg {avg:.1f} (need >= {min_avg}); below {min_axis}: {below or 'none'} [{detail}]",
        )]
    return [Finding(check="rubric", severity="info", message=f"avg {avg:.1f} [{detail}]")]
