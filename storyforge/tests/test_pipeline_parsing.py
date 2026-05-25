"""Regression tests: planning steps must tolerate odd JSON shapes from the model.

Models occasionally return a bare JSON list (or scalar) where an object was
requested. The pipeline must degrade gracefully instead of crashing with
`'list' object has no attribute 'get'` mid-run.
"""
from __future__ import annotations

from storyforge import config, llm, pipeline
from storyforge.models import Premise, StorySpec


def _pipe(tmp_path):
    return pipeline.Pipeline(draft_model="x", run_dir=tmp_path,
                             cfg=config.RunConfig(draft_model="x", judge_model="x"))


def test_gen_bible_accepts_bare_list(tmp_path, monkeypatch):
    # Model returns the characters array directly, with no enclosing object.
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: [
        {"name": "Mara", "role": "protagonist", "want": "find the map"},
        {"name": "Tomas", "role": "ally"},
    ])
    bible = _pipe(tmp_path).gen_bible("idea", StorySpec(), Premise(title="T"))
    assert [c.name for c in bible.characters] == ["Mara", "Tomas"]
    assert any(c.role == "protagonist" for c in bible.characters)


def test_gen_spec_tolerates_non_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: ["genre?", "tone?"])
    spec = _pipe(tmp_path).gen_spec("idea", {"genre": "noir"})
    # overrides still applied; pov falls back to a valid default rather than crashing.
    assert spec.genre == "noir"
    assert spec.pov == "third_person_limited"


def test_gen_premise_tolerates_non_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: "oops not json object")
    prem = _pipe(tmp_path).gen_premise("idea", StorySpec())
    assert prem.title == "Untitled"
