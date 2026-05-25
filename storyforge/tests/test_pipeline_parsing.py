"""Regression tests: planning steps must tolerate odd JSON shapes from the model.

Models occasionally return a bare JSON list (or scalar) where an object was
requested. The pipeline must degrade gracefully instead of crashing with
`'list' object has no attribute 'get'` mid-run.
"""
from __future__ import annotations

from storyforge import config, llm, pipeline
from storyforge.models import Character, Premise, StorySpec, WorldBible


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


def _bible():
    return WorldBible(characters=[Character(name="Mara", role="protagonist"),
                                  Character(name="Tomas", role="ally")])


def _spine_rows(roles):
    return {"chapters": [
        {"number": i, "title": f"Ch{i}", "summary": "s", "beats": ["b"],
         "structural_role": role} for i, role in enumerate(roles, 1)]}


def test_gen_spine_records_structural_role(tmp_path, monkeypatch):
    roles = ["setup", "rising", "midpoint", "lowest_point", "climax"]
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: _spine_rows(roles))
    spine = _pipe(tmp_path).gen_spine(StorySpec(), Premise(title="T"), _bible())
    assert [p.structural_role for p in spine] == roles


def test_gen_spine_forces_exactly_one_climax_when_model_omits_it(tmp_path, monkeypatch):
    # Model assigned no climax at all -> the final chapter must become the climax.
    roles = ["setup", "rising", "midpoint", "lowest_point", "rising"]
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: _spine_rows(roles))
    spine = _pipe(tmp_path).gen_spine(StorySpec(), Premise(title="T"), _bible())
    climaxes = [p.number for p in spine if p.structural_role == "climax"]
    assert climaxes == [spine[-1].number]


def test_gen_spine_collapses_multiple_climaxes(tmp_path, monkeypatch):
    # Two climaxes -> exactly one remains (the last chapter).
    roles = ["setup", "climax", "midpoint", "climax", "resolution"]
    monkeypatch.setattr(llm, "complete_json", lambda *a, **k: _spine_rows(roles))
    spine = _pipe(tmp_path).gen_spine(StorySpec(), Premise(title="T"), _bible())
    assert sum(1 for p in spine if p.structural_role == "climax") == 1
