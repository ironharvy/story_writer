"""Unit tests for the LLM-judge eval gates.

The model calls are isolated behind small functions in each gate module, so
these tests monkeypatch those and exercise the pure aggregation/severity logic
without touching ollama.
"""
from __future__ import annotations

import pytest

from core.types import WorldBible
from evals import (
    bible_consistency,
    bible_internal,
    comprehension,
    continuity,
    premise_fidelity,
)
from evals.bible_consistency import BibleIssue
from evals.bible_internal import InternalIssue
from evals.comprehension import ComprehensionQuestion, QuestionVerdict
from evals.continuity import ChapterDigest, Contradiction
from evals.premise_fidelity import PremiseGap
from evals.types import GateReport, EvalFinding, worst_severity


@pytest.fixture
def bible() -> WorldBible:
    return WorldBible(
        story_title="Mages Die Young",
        rules_of_the_world=["Casting spills the caster's blood and ages them."],
        characters=["Sera, a young battle-mage who hoards her years."],
        locations=["The Bleeding Field, a war front where rain never washes the mud clean."],
        timeline=["The war began a generation ago."],
    )


# --- types -------------------------------------------------------------------

def test_worst_severity_and_report_helpers():
    findings = [
        EvalFinding("x", "info", "i"),
        EvalFinding("x", "warn", "w", fix="do w"),
        EvalFinding("x", "fail", "f", fix="do f", locus="Ch1"),
    ]
    assert worst_severity(findings) == "fail"
    assert worst_severity([]) == "info"
    report = GateReport(gate="x", findings=findings)
    assert not report.passed()
    assert len(report.blocking) == 1
    assert len(report.warnings) == 1
    brief = report.revision_brief()
    assert "MUST FIX" in brief and "SHOULD FIX" in brief
    assert "[Ch1]" in brief
    assert "(info)" not in brief.lower()  # info findings excluded from brief


# --- bible_consistency -------------------------------------------------------

def test_bible_consistency_severity_mapping(monkeypatch, bible):
    monkeypatch.setattr(bible_consistency, "_judge", lambda *_a, **k: [
        BibleIssue(entity="The Bleeding Field", kind="location", severity="contradiction",
                   explanation="Described as a clean meadow", fix="Restore the bloodied mud"),
        BibleIssue(entity="Sera", kind="character", severity="drift",
                   explanation="Spends years freely", fix="Show her hoarding"),
        BibleIssue(entity="casting", kind="rule", severity="omission",
                   explanation="A storm is summoned with no blood/aging", fix="Show the cost"),
        BibleIssue(entity="x", kind="rule", severity="minor", explanation="tiny", fix="tweak"),
    ])
    report = bible_consistency.check(bible, "Chapter 1", "prose...")
    sevs = sorted(f.severity for f in report.findings)
    assert sevs == ["fail", "info", "warn", "warn"]
    assert not report.passed()


def test_bible_consistency_clean(monkeypatch, bible):
    monkeypatch.setattr(bible_consistency, "_judge", lambda *_a, **k: [])
    report = bible_consistency.check(bible, "Chapter 1", "prose")
    assert report.passed()
    assert report.findings == []


def test_bible_consistency_judge_error_is_swallowed(monkeypatch, bible):
    def boom(*_a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(bible_consistency, "_judge", boom)
    report = bible_consistency.check(bible, "Chapter 1", "prose")
    assert report.passed()  # failure degrades to "no findings", never crashes


# --- bible_internal ----------------------------------------------------------

def test_bible_internal_surface_contradiction_and_clarifications(monkeypatch, bible):
    monkeypatch.setattr(bible_internal, "_judge", lambda wb: [
        InternalIssue(entities=["Kael Thorne", "Elara Vance"], kind="naming",
                      severity="contradiction",
                      explanation="Siblings given different surnames",
                      resolution="Both siblings use the surname Thorne"),
        InternalIssue(entities=["Commander Vane"], kind="age", severity="tension",
                      explanation="Age vaguely fits timeline",
                      resolution="Vane is ~180 chronological years old"),
    ])
    report, clarifications = bible_internal.check(bible)
    assert not report.passed()
    assert len(report.blocking) == 1
    assert clarifications == [
        "Both siblings use the surname Thorne",
        "Vane is ~180 chronological years old",
    ]


def test_bible_internal_dedups_clarifications(monkeypatch, bible):
    monkeypatch.setattr(bible_internal, "_judge", lambda wb: [
        InternalIssue(entities=["A", "B"], kind="naming", severity="contradiction",
                      explanation="x", resolution="Use surname Thorne"),
        InternalIssue(entities=["A", "C"], kind="naming", severity="tension",
                      explanation="y", resolution="Use surname Thorne"),
    ])
    _, clarifications = bible_internal.check(bible)
    assert clarifications == ["Use surname Thorne"]


def test_bible_internal_error_is_swallowed(monkeypatch, bible):
    monkeypatch.setattr(bible_internal, "_judge", lambda wb: (_ for _ in ()).throw(RuntimeError("x")))
    report, clarifications = bible_internal.check(bible)
    assert report.passed() and clarifications == []


# --- comprehension -----------------------------------------------------------

def test_comprehension_essential_missing_is_blocking(monkeypatch, bible):
    monkeypatch.setattr(comprehension, "generate_questions", lambda *_a, **k: [
        ComprehensionQuestion(question="Is Sera named?", importance="essential", why="intro"),
        ComprehensionQuestion(question="What does casting cost?", importance="essential", why="rule"),
        ComprehensionQuestion(question="What is the weather?", importance="supporting", why="mood"),
    ])
    monkeypatch.setattr(comprehension, "answer_and_grade", lambda prose, _qs: [
        QuestionVerdict(question="Is Sera named?", answer="Yes, Sera", verdict="answered", note=""),
        QuestionVerdict(question="What does casting cost?", answer="NOT IN TEXT", verdict="missing",
                        note="Show the blood/aging cost"),
        QuestionVerdict(question="What is the weather?", answer="vague", verdict="partial", note="clarify"),
    ])
    report = comprehension.check("Chapter 1", "beats", bible, "prose", "")
    # essential missing -> fail; supporting partial -> warn; answered -> dropped
    assert not report.passed()
    assert len(report.blocking) == 1
    assert len(report.warnings) == 1
    assert len(report.findings) == 2


def test_comprehension_supporting_missing_is_warn(monkeypatch, bible):
    monkeypatch.setattr(comprehension, "generate_questions", lambda *_a, **k: [
        ComprehensionQuestion(question="Minor detail?", importance="supporting", why="x"),
    ])
    monkeypatch.setattr(comprehension, "answer_and_grade", lambda prose, _qs: [
        QuestionVerdict(question="Minor detail?", answer="NOT IN TEXT", verdict="missing", note="add"),
    ])
    report = comprehension.check("Chapter 1", "beats", bible, "prose", "")
    assert report.passed()  # no essential gap
    assert len(report.warnings) == 1


def test_comprehension_all_answered_is_clean(monkeypatch, bible):
    monkeypatch.setattr(comprehension, "generate_questions", lambda *_a, **k: [
        ComprehensionQuestion(question="Q?", importance="essential", why="x"),
    ])
    monkeypatch.setattr(comprehension, "answer_and_grade", lambda prose, _qs: [
        QuestionVerdict(question="Q?", answer="A", verdict="answered", note=""),
    ])
    report = comprehension.check("Chapter 1", "beats", bible, "prose", "")
    assert report.passed()
    assert report.findings == []


# --- continuity --------------------------------------------------------------

def test_continuity_flags_hard_contradiction(monkeypatch):
    monkeypatch.setattr(continuity, "_digest_chapter",
                        lambda t, p: ChapterDigest(characters_present=["Sera"]))
    monkeypatch.setattr(continuity, "_find", lambda digests: [
        Contradiction(chapters=["Chapter 3", "Chapter 5"], kind="death", severity="hard",
                      explanation="Sera dies in 3 but acts in 5", fix="Resurrect or revise ch5"),
        Contradiction(chapters=["Chapter 2", "Chapter 4"], kind="trait", severity="minor",
                      explanation="eye colour wobble", fix="pick one"),
    ])
    report = continuity.check({"Chapter 3": "p", "Chapter 5": "p"})
    assert not report.passed()
    assert len(report.blocking) == 1
    assert len(report.warnings) == 1


def test_continuity_single_chapter_is_noop(monkeypatch):
    monkeypatch.setattr(continuity, "_find", lambda d: (_ for _ in ()).throw(AssertionError("should not run")))
    report = continuity.check({"only": "prose"})
    assert report.passed()


def test_continuity_render_digest_includes_reconciled_and_facts():
    from evals.continuity import _render_digest, ChapterDigest
    d = ChapterDigest(
        characters_present=["Kael"],
        hard_facts=["Kael: chronological 19, looks 17, hoarded reserve ~40 years"],
        reconciled=["3000-year figure is the Ledger's lie; true age ~300"],
    )
    rendered = _render_digest("Chapter 1", d)
    assert "hoarded reserve ~40 years" in rendered
    assert "reconciled (do not re-flag):" in rendered
    assert "Ledger's lie" in rendered


# --- phrase_freshness (deterministic, no model) ------------------------------

def test_phrase_freshness_first_chapter_is_clean():
    from evals import phrase_freshness
    report = phrase_freshness.check("any prose here", prior_chapters=[], world_bible=None)
    assert report.passed() and report.findings == []


def test_phrase_freshness_flags_recurring_stock_phrases(bible):
    from evals import phrase_freshness
    stock = "his heart hammered against his ribs as he felt the weight of his hoard"
    # the crutch phrase recurs across TWO prior chapters -> qualifies as stock
    prior = [
        f"In the trench {stock}. The mud was cold and grey.",
        f"On the wall {stock}. The wind tore at the banners overhead.",
    ]
    cur = f"Later that night {stock}. Nothing else seemed to matter at all."
    report = phrase_freshness.check(cur, prior, bible, n=5)
    assert report.findings, "expected recurring stock phrases to be flagged"
    assert report.findings[0].check == "phrase_freshness"
    assert "Rewrite" in report.findings[0].fix


def test_phrase_freshness_ignores_oneoff_overlap(bible):
    from evals import phrase_freshness
    stock = "his heart hammered against his ribs as he felt the weight of his hoard"
    # appears in only ONE prior chapter -> not yet a crutch, not flagged
    prior = [f"In the trench {stock}. The mud was cold and grey."]
    cur = f"Later that night {stock}. Nothing else seemed to matter at all."
    report = phrase_freshness.check(cur, prior, bible, n=5)
    assert report.passed()


def test_phrase_freshness_fresh_prose_passes(bible):
    from evals import phrase_freshness
    prior = ["The dawn broke red over the shattered spires of the citadel."]
    cur = "A cold wind threaded between the tents, carrying the smell of rust and rain."
    report = phrase_freshness.check(cur, prior, bible, n=5)
    assert report.passed()


def test_phrase_freshness_ignores_character_names(bible):
    from evals import phrase_freshness
    # Repeating the protagonist's name across chapters must not count as reuse.
    prior = ["Sera walked. Sera spoke. Sera fought. Sera fell. Sera rose again here."]
    cur = "Sera walked. Sera spoke. Sera fought. Sera fell. Sera rose again here too."
    report = phrase_freshness.check(cur, prior, bible, n=5)
    # the only overlaps are name-dominated grams -> filtered out
    assert report.passed()


# --- premise_fidelity --------------------------------------------------------

def test_premise_fidelity_score_and_gaps(monkeypatch):
    class _Res:
        fidelity_score = 4
        gaps = [
            PremiseGap(element="aging cost", severity="missing",
                       explanation="No one visibly ages", fix="Show it in the climax"),
            PremiseGap(element="old start wars", severity="weak",
                       explanation="Implied only", fix="Add a council scene"),
        ]
    monkeypatch.setattr(premise_fidelity, "_judge", lambda *_a, **k: _Res())
    report = premise_fidelity.check("idea", "premise", "spine", "summary")
    # first finding is the info score line
    assert report.findings[0].severity == "info"
    assert "4/5" in report.findings[0].message
    assert any(f.severity == "warn" for f in report.findings)  # missing -> warn
    assert any(f.severity == "info" and "weak" in f.message for f in report.findings)
