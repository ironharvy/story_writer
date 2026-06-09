"""Tests for the LM-judge canon metrics — no LM is called; verdicts are injected
(exactly as pov_check tests pass pre-classified ChapterPOV objects)."""
import canon as canon_mod
from bench import judge
from bench.judge import Verdict

CANON = canon_mod.load_canon(canon_mod.DEFAULT_CANON_PATH)


def _story(body: str) -> str:
    return ("# Story\n\n## World bible\n\n### Characters\n\n## Final Story\n\n"
            f"### Chapter 12: Test\n\n{body}\n")


def test_appearance_described_fail_and_consistency_pass():
    text = _story("He moved through the hall.")
    findings = judge.judge_appearance(
        text, CANON,
        described=Verdict(holds=False, evidence="no physical detail"),
        consistent=Verdict(holds=True),
    )
    by = {f.check: f for f in findings}
    assert by["appearance_described"].severity == "fail"
    assert by["appearance_vs_spec"].severity == "info"


def test_appearance_vs_spec_fail():
    text = _story("His blue eyes shone.")
    findings = judge.judge_appearance(
        text, CANON,
        described=Verdict(holds=True),
        consistent=Verdict(holds=False, evidence="blue eyes contradict grey"),
    )
    by = {f.check: f for f in findings}
    assert by["appearance_vs_spec"].severity == "fail"


def test_invariants_produce_one_finding_per_rule():
    text = _story("Azazel walked the surface market in the noon sun.")
    verdicts = {
        "azazel_maw": Verdict(holds=False, evidence="Azazel on the surface in ch.12"),
        "leash_fray": Verdict(holds=True),
        "kira_not_strawman": Verdict(holds=True),
    }
    findings = judge.judge_invariants(text, CANON, verdicts=verdicts)
    by = {f.check: f.severity for f in findings}
    assert by["azazel_maw"] == "fail"
    assert by["leash_fray"] == "info"
    assert by["kira_not_strawman"] == "info"


def test_dangling_payoff_is_warn():
    text = _story("A sealed letter appeared and was never opened.")
    findings = judge.judge_dangling_payoff(text, verdict=Verdict(holds=False))
    assert findings[0].check == "dangling_payoff"
    assert findings[0].severity == "warn"


def test_run_all_uses_injected_verdicts():
    text = _story("He moved through the hall.")
    verdicts = {
        "appearance_described": Verdict(holds=True),
        "appearance_vs_spec": Verdict(holds=True),
        "azazel_maw": Verdict(holds=True),
        "leash_fray": Verdict(holds=True),
        "kira_not_strawman": Verdict(holds=True),
        "dangling_payoff": Verdict(holds=True),
    }
    findings = judge.run_all(text, CANON, verdicts=verdicts)
    assert findings and all(f.severity == "info" for f in findings)
