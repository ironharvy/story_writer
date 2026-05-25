"""Tier 3 — qualitative rubric (LLM judge, 1–5 per axis)."""
from __future__ import annotations

from . import Check, PASS, FAIL
from .judge import Judge, build_view
from .parse import Manuscript

AXES = {
    "arc_structure": "Arc & structure: clear setup -> rising action -> climax -> resolution",
    "ending": "Ending: earned and lands the premise's promise (not a fizzle / deus ex machina)",
    "character_agency": "Character agency: choices drive the plot; characters want things",
    "scene_vs_summary": "Scene vs summary: dramatizes key beats in scene, not just telling",
    "prose_quality": "Prose quality: varied, controlled, few tics",
    "cohesion": "Cohesion: setups pay off; threads resolve",
}
THRESHOLD_AVG = 4.0
MIN_AXIS = 3


def rubric(m: Manuscript, idea: str, judge: Judge) -> tuple[Check, dict]:
    axes_desc = "\n".join(f"- {k}: {v}" for k, v in AXES.items())
    prompt = (
        f"The author's original idea:\n\"\"\"\n{idea}\n\"\"\"\n\n"
        "Score the manuscript below from 1 (worst) to 5 (best) on each axis. Use the "
        "full range; be honest. Axes:\n" + axes_desc + "\n\n"
        'Return JSON: {"scores": {'
        + ", ".join(f'"{k}": <1-5>' for k in AXES)
        + '}, "justifications": {'
        + ", ".join(f'"{k}": "<one line>"' for k in AXES)
        + "}}\n\n" + build_view(m, 1000)
    )
    d = judge.ask(prompt, trace_name="t3-rubric", max_tokens=1600)
    raw = d.get("scores", d) if isinstance(d, dict) else {}
    scores: dict[str, int] = {}
    for ax in AXES:
        try:
            v = int(round(float(raw.get(ax, 0))))
        except (TypeError, ValueError):
            v = 0
        scores[ax] = max(0, min(5, v))
    valid = [v for v in scores.values() if v > 0]
    avg = round(sum(valid) / len(valid), 2) if valid else 0.0
    below = {ax: v for ax, v in scores.items() if 0 < v < MIN_AXIS}
    missing = [ax for ax, v in scores.items() if v == 0]

    rubric_data = {
        "scores": scores,
        "average": avg,
        "justifications": d.get("justifications", {}) if isinstance(d, dict) else {},
        "threshold_avg": THRESHOLD_AVG,
        "min_axis": MIN_AXIS,
    }
    ok = avg >= THRESHOLD_AVG and not below and not missing
    sev = PASS if ok else FAIL
    msg = f"Rubric average {avg} (need >= {THRESHOLD_AVG}); "
    if missing:
        msg += f"missing scores for {', '.join(missing)}; "
    if below:
        msg += "below floor: " + ", ".join(f"{k}={v}" for k, v in below.items())
    else:
        msg += "no axis below " + str(MIN_AXIS)
    return Check("T3", "Rubric", sev, msg, rubric_data), rubric_data
