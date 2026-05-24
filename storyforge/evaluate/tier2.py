"""Tier 2 — LLM-judge gates: POV, protagonist naming, continuity, premise fidelity."""
from __future__ import annotations

from collections import Counter

from . import FAIL, PASS, WARN, Check
from .judge import Judge, build_view, truncate_words
from .parse import Manuscript

_POV_LABELS = {"first_person", "third_person", "mixed", "other"}


def t2_1_pov(m: Manuscript, judge: Judge) -> Check:
    """Classify each chapter's narration POV; FAIL on within-chapter mix or a
    decisive disagreement with the book's dominant POV."""
    blocks = "\n\n".join(
        f"[Chapter {i}] {c.label}\n{truncate_words(c.prose, 1400)}"
        for i, c in enumerate(m.chapters, 1)
    )
    prompt = (
        "For EACH chapter below, classify the dominant point of view of the "
        "NARRATION ONLY — ignore POV inside dialogue, letters, journals, or reported "
        "speech. Labels: first_person, third_person, mixed (the narration's own POV "
        "shifts WITHIN the chapter), other.\n\n"
        'Return JSON: {"chapters": [{"chapter": <int>, "pov": "<label>", "note": "<short>"}]}'
        " with one entry per chapter, in order.\n\n" + blocks
    )
    data = judge.ask(prompt, trace_name="t2.1-pov", max_tokens=1500)
    rows = data.get("chapters", data) if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        return Check("T2.1", "POV consistency", WARN, "Judge returned no POV rows.",
                     {"raw": str(data)[:300]})
    povs = []
    for i, row in enumerate(rows, 1):
        pov = str(row.get("pov", "other")).lower().strip() if isinstance(row, dict) else "other"
        if pov not in _POV_LABELS:
            pov = "other"
        povs.append({"chapter": row.get("chapter", i) if isinstance(row, dict) else i,
                     "pov": pov, "note": (row.get("note", "") if isinstance(row, dict) else "")})

    mixed = [p for p in povs if p["pov"] == "mixed"]
    decisive = [p["pov"] for p in povs if p["pov"] in ("first_person", "third_person")]
    dominant = Counter(decisive).most_common(1)[0][0] if decisive else None
    disagree = [p for p in povs if p["pov"] in ("first_person", "third_person")
                and p["pov"] != dominant]

    details = {"dominant": dominant, "per_chapter": povs}
    if mixed:
        return Check("T2.1", "POV consistency", FAIL,
                     f"{len(mixed)} chapter(s) shift POV within the chapter (mixed): "
                     + ", ".join(str(p["chapter"]) for p in mixed), details)
    if disagree:
        return Check("T2.1", "POV consistency", FAIL,
                     f"Dominant POV is {dominant}, but chapter(s) "
                     + ", ".join(str(p["chapter"]) for p in disagree)
                     + " disagree.", details)
    return Check("T2.1", "POV consistency", PASS,
                 f"Consistent {dominant or 'single'} POV across all chapters.", details)


def t2_2_protagonist(m: Manuscript, judge: Judge) -> Check:
    """Protagonist must be named (not a placeholder) and named consistently."""
    prompt = (
        "Analyze how the PROTAGONIST is referred to in the narration.\n"
        "(a) Is the protagonist referred to primarily by a PROPER NAME, or by a "
        "placeholder such as 'the protagonist', 'the child', 'the boy', 'the girl', "
        "'the woman', 'the man'? Judge the PRIMARY referent across chapters.\n"
        "(b) Is the protagonist called by two or more DIFFERENT proper names across "
        "the book (a real inconsistency, not a nickname clearly introduced as such)?\n\n"
        'Return JSON: {"protagonist": "<name or description>", '
        '"primary_referent": "name" | "placeholder", '
        '"placeholder_examples": ["..."], "multiple_names": true|false, '
        '"names_used": ["..."], "note": "<short>"}\n\n' + build_view(m, 700)
    )
    d = judge.ask(prompt, trace_name="t2.2-protagonist", max_tokens=900)
    primary = str(d.get("primary_referent", "name")).lower()
    multiple = bool(d.get("multiple_names", False))
    details = {k: d.get(k) for k in ("protagonist", "primary_referent",
                                     "placeholder_examples", "multiple_names", "names_used")}
    if primary == "placeholder":
        return Check("T2.2", "Protagonist naming", FAIL,
                     "Protagonist is referred to primarily by a placeholder, not a name: "
                     + ", ".join(d.get("placeholder_examples", []) or ["(see note)"]), details)
    if multiple:
        return Check("T2.2", "Protagonist naming", FAIL,
                     "Protagonist is called by multiple different names: "
                     + ", ".join(d.get("names_used", []) or []), details)
    return Check("T2.2", "Protagonist naming", PASS,
                 f"Protagonist '{d.get('protagonist','?')}' is named consistently.", details)


def t2_3_continuity(m: Manuscript, judge: Judge) -> Check:
    """List factual contradictions; hard contradictions FAIL, minor wobble WARN."""
    prompt = (
        "List FACTUAL CONTRADICTIONS across this manuscript. Look for: a character "
        "dead or gone in one chapter who acts later; eye/hair color, names, ages, or "
        "relationships that flip; an impossible timeline; TWO DISTINCT characters who "
        "share one name; a location that teleports. Only report genuine contradictions "
        "supported by the text.\n\n"
        'Return JSON: {"contradictions": [{"type": "<kind>", "detail": "<what flips>", '
        '"severity": "hard" | "minor", "chapters": "<where>"}]}. '
        "Empty list if there are none.\n\n" + build_view(m, 1000)
    )
    d = judge.ask(prompt, trace_name="t2.3-continuity", max_tokens=1500)
    items = d.get("contradictions", []) if isinstance(d, dict) else (d if isinstance(d, list) else [])
    hard = [c for c in items if isinstance(c, dict) and str(c.get("severity", "")).lower() == "hard"]
    minor = [c for c in items if isinstance(c, dict) and str(c.get("severity", "")).lower() != "hard"]
    details = {"contradictions": items}
    if hard:
        return Check("T2.3", "Continuity", FAIL,
                     f"{len(hard)} hard contradiction(s): "
                     + "; ".join(c.get("detail", "?") for c in hard[:4]), details)
    if minor:
        return Check("T2.3", "Continuity", WARN,
                     f"{len(minor)} minor continuity wobble(s): "
                     + "; ".join(c.get("detail", "?") for c in minor[:4]), details)
    return Check("T2.3", "Continuity", PASS, "No contradictions found.", details)


def t2_4_premise_fidelity(m: Manuscript, idea: str, judge: Judge) -> Check:
    """Does the story dramatize the user's actual idea, or drift to a generic tale?"""
    prompt = (
        f"The author's original story idea was:\n\"\"\"\n{idea}\n\"\"\"\n\n"
        "Does the manuscript below actually DRAMATIZE this idea, or did it drift into "
        "a generic or unrelated story? Judge fidelity to the core premise, not prose "
        "quality.\n\n"
        'Return JSON: {"fidelity": "high" | "medium" | "low", "drift": true|false, '
        '"note": "<short reason>"}\n\n' + build_view(m, 800)
    )
    d = judge.ask(prompt, trace_name="t2.4-premise", max_tokens=700)
    fidelity = str(d.get("fidelity", "medium")).lower()
    details = {"fidelity": fidelity, "drift": d.get("drift"), "note": d.get("note", "")}
    if fidelity == "low" or (d.get("drift") and fidelity not in ("high",)):
        sev = FAIL if fidelity == "low" else WARN
        return Check("T2.4", "Premise fidelity", sev,
                     f"Fidelity {fidelity}: {d.get('note','')}", details)
    if fidelity == "medium":
        return Check("T2.4", "Premise fidelity", WARN,
                     f"Partial fidelity: {d.get('note','')}", details)
    return Check("T2.4", "Premise fidelity", PASS,
                 f"Story dramatizes the original idea. {d.get('note','')}", details)


def run_tier2(m: Manuscript, idea: str, judge: Judge) -> list[Check]:
    return [t2_1_pov(m, judge), t2_2_protagonist(m, judge),
            t2_3_continuity(m, judge), t2_4_premise_fidelity(m, idea, judge)]
