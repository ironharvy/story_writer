"""Run all tiers, assemble the JSON scorecard, and decide ship: true|false.

Ship gate (STORY_EVAL_SPEC.md):
  1. zero Tier-1 FAIL (T1.1 length floor, T1.2 character presence, T1.7 ending)
  2. zero Tier-2 FAIL (T2.1 POV, T2.2 naming, T2.3 contradictions, T2.4 fidelity)
  3. Tier-3 rubric average >= 4.0, no axis < 3
  4. soft budgets: name-drift <= 2, phrase-reuse <= 5, no over-repetition flagged
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .. import config
from . import FAIL, Check
from . import tier1, tier2, tier3
from .judge import Judge
from .parse import Manuscript, parse_manuscript

NAME_DRIFT_BUDGET = tier1.NAME_DRIFT_BUDGET
PHRASE_REUSE_BUDGET = tier1.PHRASE_REUSE_BUDGET


def _soft_budget_blockers(checks: dict[str, Check]) -> list[str]:
    blockers = []
    t13 = checks.get("T1.3")
    if t13 and len(t13.details.get("flags", [])) > NAME_DRIFT_BUDGET:
        blockers.append(f"T1.3 name-drift {len(t13.details['flags'])} > {NAME_DRIFT_BUDGET}")
    t14 = checks.get("T1.4")
    if t14 and t14.details.get("count", 0) > PHRASE_REUSE_BUDGET:
        blockers.append(f"T1.4 phrase-reuse {t14.details['count']} > {PHRASE_REUSE_BUDGET}")
    t15 = checks.get("T1.5")
    if t15 and t15.details.get("flagged"):
        blockers.append(f"T1.5 over-repetition: {len(t15.details['flagged'])} word(s)")
    return blockers


def evaluate(
    md_text: str,
    idea: str,
    *,
    draft_model: str,
    judge_model: str,
    tier1_only: bool = False,
    judge_cfg: config.RunConfig | None = None,
    source: str = "",
) -> dict:
    m: Manuscript = parse_manuscript(md_text)
    checks: list[Check] = list(tier1.run_tier1(m))

    rubric_data: dict = {}
    judge_ran = False
    if not tier1_only:
        jcfg = judge_cfg or config.RunConfig(
            draft_model=draft_model, judge_model=judge_model, temperature=0.1, think=False
        )
        judge = Judge(model=judge_model, cfg=jcfg)
        checks.extend(tier2.run_tier2(m, idea, judge))
        rubric_check, rubric_data = tier3.rubric(m, idea, judge)
        checks.append(rubric_check)
        judge_ran = True

    by_id = {c.id: c for c in checks}

    # --- ship gate ---
    blockers: list[str] = []
    for gate in ("T1.1", "T1.2", "T1.7"):
        c = by_id.get(gate)
        if c and c.severity == FAIL:
            blockers.append(f"{gate} FAIL: {c.message}")
    if judge_ran:
        for gate in ("T2.1", "T2.2", "T2.3", "T2.4"):
            c = by_id.get(gate)
            if c and c.severity == FAIL:
                blockers.append(f"{gate} FAIL: {c.message}")
        t3 = by_id.get("T3")
        if t3 and t3.severity == FAIL:
            blockers.append(f"T3 FAIL: {t3.message}")
    else:
        blockers.append("Tier-2/3 not run (tier1_only); cannot ship.")
    blockers.extend(_soft_budget_blockers(by_id))

    ship = not blockers and judge_ran
    wc = sum(c.word_count for c in m.chapters)
    return {
        "source": source,
        "title": m.title,
        "idea": idea,
        "draft_model": draft_model,
        "judge_model": judge_model if judge_ran else None,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "word_count": wc,
        "chapters": len(m.chapters),
        "checks": [c.to_dict() for c in checks],
        "rubric": rubric_data,
        "ship": ship,
        "ship_blockers": blockers,
    }


def evaluate_file(path: str | Path, idea: str, **kw) -> dict:
    p = Path(path)
    return evaluate(p.read_text(), idea, source=str(p), **kw)


def summarize(scorecard: dict) -> str:
    """Human-readable one-screen summary of a scorecard."""
    lines = [
        f"{'SHIP ✅' if scorecard['ship'] else 'NO-SHIP ❌'}  "
        f"\"{scorecard['title']}\"  "
        f"({scorecard['chapters']} ch, {scorecard['word_count']}w)  "
        f"draft={scorecard['draft_model']} judge={scorecard['judge_model']}",
    ]
    sev_mark = {"pass": "✓", "warn": "▲", "fail": "✗"}
    for c in scorecard["checks"]:
        lines.append(f"  {sev_mark.get(c['severity'], '?')} {c['id']:5} {c['name']:22} {c['message']}")
    r = scorecard.get("rubric") or {}
    if r.get("scores"):
        scores = "  ".join(f"{k}={v}" for k, v in r["scores"].items())
        lines.append(f"  rubric avg={r.get('average')}: {scores}")
    if scorecard["ship_blockers"]:
        lines.append("  blockers:")
        lines.extend(f"    - {b}" for b in scorecard["ship_blockers"])
    return "\n".join(lines)


def write_scorecard(scorecard: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(scorecard, indent=2, ensure_ascii=False))
