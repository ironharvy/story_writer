"""Wiring tests for the canon optimizer harness — no LM is configured, so we
verify the trainset/program/metric/optimizer/evaluator all BUILD and the
per-chapter metric scores deterministically. The live compile is not exercised.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import dspy

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import canon as canon_mod  # noqa: E402
import optimize_canon_pipeline as opt  # noqa: E402
from bench import metric  # noqa: E402

CANON = canon_mod.load_canon(canon_mod.DEFAULT_CANON_PATH)


def test_build_trainset_has_examples_with_metric_fields():
    trainset = opt.build_trainset(CANON)
    assert trainset
    ex = trainset[0]
    assert "story_idea" in ex.inputs()
    assert "canon_directives" in ex.inputs()
    assert isinstance(ex.chapter_number, int)
    # The required-artifact chapter (4) should be among the seeded focus chapters.
    assert any(e.chapter_number == 4 for e in trainset)


def test_build_program_is_a_module():
    program = opt.ChapterDrafter()
    assert isinstance(program, dspy.Module)


def test_build_optimizer_and_evaluator_construct():
    optimizer = opt.build_optimizer(CANON, threshold=1.0)
    assert hasattr(optimizer, "compile")
    evaluator = opt.build_evaluator(opt.build_trainset(CANON), CANON)
    assert callable(evaluator)


def test_chapter_metric_rewards_clean_prose():
    m = metric.make_chapter_metric(CANON)
    clean = SimpleNamespace(prose="Snow fell; his grey eyes watched the gate.")
    dirty = SimpleNamespace(prose="His blue eyes opened. He had no Ch.19.")
    ex = SimpleNamespace(chapter_number=2, chapter_title="X", canon=CANON)
    assert m(ex, clean) == 1.0
    assert m(ex, dirty) < 1.0


def test_chapter_metric_threshold_returns_bool():
    m = metric.make_chapter_metric(CANON, threshold=1.0)
    ex = SimpleNamespace(chapter_number=4, chapter_title="Rite", canon=CANON)
    # Chapter 4 requires the hymn; prose without it should fall below threshold.
    assert m(ex, SimpleNamespace(prose="They gathered and dispersed in silence.")) is False


def test_main_without_model_skips_compile(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    assert opt.main(["--canon", str(canon_mod.DEFAULT_CANON_PATH)]) == 0
