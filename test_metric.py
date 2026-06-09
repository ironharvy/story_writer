"""Tests for the composite story-quality metric (deterministic; no LM)."""
import canon as canon_mod
from bench import metric
from bench.judge import Verdict
from bench import judge

CANON = canon_mod.load_canon(canon_mod.DEFAULT_CANON_PATH)


def _story(body: str, title: str = "Chapter 1: Open") -> str:
    return ("# Story\n\n## World bible\n\n### Characters\n\n## Final Story\n\n"
            f"### {title}\n\n{body}\n")


def test_clean_story_scores_one():
    text = _story("Snow fell on the courtyard; his grey eyes watched the gate.")
    assert metric.score_text(text, CANON) == 1.0


def test_each_violation_lowers_score():
    clean = _story("Snow fell; his grey eyes watched the gate quietly.")
    dirty = _story("His blue eyes. Ch.19. Over the next eight years.")
    assert metric.score_text(dirty, CANON) < metric.score_text(clean, CANON)
    assert metric.score_text(dirty, CANON) >= 0.0


def test_score_is_clamped_to_zero():
    # Pile on enough violations that raw penalty would exceed 1.0.
    text = _story(
        "His blue eyes. His green eyes. Ch.19. Beat 3. The reader. "
        "Over the next eight years.",
        title="Chapter 4: Rite",  # also missing required hymn artifact
    )
    assert 0.0 <= metric.score_text(text, CANON) <= 1.0


def test_judge_findings_count_against_score():
    text = _story("He moved through the hall.")
    bad_judge = judge.judge_appearance(
        text, CANON,
        described=Verdict(holds=False),
        consistent=Verdict(holds=True),
    )
    with_judge = metric.score_text(text, CANON, judge_findings=bad_judge)
    without = metric.score_text(text, CANON)
    assert with_judge < without


def test_make_story_metric_threshold_returns_bool():
    m = metric.make_story_metric(CANON, threshold=0.9)
    clean = _story("Snow fell; his grey eyes watched the gate.")
    dirty = _story("His blue eyes opened on Ch.19.")
    assert m(object(), clean) is True
    assert m(object(), dirty) is False


def test_make_story_metric_float_default():
    m = metric.make_story_metric(CANON)
    val = m(object(), _story("Snow fell; his grey eyes watched."))
    assert isinstance(val, float) and val == 1.0
