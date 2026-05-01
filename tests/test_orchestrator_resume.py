from __future__ import annotations

from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from story_writer.models.qa import RuleResult
from story_writer.models.run import Manifest
from story_writer.models.story import (
    Chapter,
    ChapterBeat,
    ChapterPlan,
    Character,
    EnhancementGuide,
    EnhancementNotes,
    Location,
    PlannedChapter,
    Premise,
    Spine,
    StoryLength,
    WorldBible,
)
from story_writer.orchestrator import (
    _remove_unknown_names,
    _revise_hard_qa_failures,
    resume_run,
)
from story_writer.run_store import RunStore
from story_writer.stages import StageContext


def test_resume_skips_existing_generation_artifacts_and_runs_qa(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.create("demo", "Lila searches.", Manifest(slug="demo", length="short"))
    store.write_json("demo", "clarify", [])
    store.write_model(
        "demo",
        "premise",
        Premise(
            protagonist="Lila",
            want="find Elias",
            obstacle="flood",
            stakes="memory",
            genre="fantasy",
            length=StoryLength.SHORT,
            summary="Lila searches the Flooded Valley.",
        ),
    )
    store.write_model(
        "demo",
        "spine",
        Spine(
            once_upon_a_time="Lila lived in the Flooded Valley.",
            every_day="The water erased memory.",
            until_one_day="She found a clue.",
            because_of_that="She followed it.",
            until_finally="She chose what to remember.",
            ever_since_then="The valley changed.",
        ),
    )
    store.write_model(
        "demo",
        "world_bible",
        WorldBible(
            logline="Lila searches the Flooded Valley.",
            characters=[Character(name="Lila", role="protagonist")],
            locations=[
                Location(name="Flooded Valley", description="A drowned valley.")
            ],
        ),
    )
    beat = ChapterBeat(order=1, summary="Lila enters the water.")
    store.write_model(
        "demo",
        "chapter_plan",
        ChapterPlan(
            length=StoryLength.SHORT,
            chapters=[
                PlannedChapter(
                    number=1,
                    title="The Water",
                    purpose="Open the search.",
                    spine_alignment="setup",
                    beats=[beat],
                )
            ],
        ),
    )
    store.write_model(
        "demo",
        "enhancement",
        EnhancementGuide(
            notes=[
                EnhancementNotes(
                    chapter_number=1,
                    tension="rising water",
                    mystery="forgotten brother",
                    theme_alignment="memory",
                    setup_payoff="locket",
                    emotional_curve="unease to resolve",
                )
            ]
        ),
    )
    store.write_model(
        "demo",
        "chapter:1",
        Chapter(
            number=1,
            title="The Water",
            beats=[beat],
            prose="Lila entered the Flooded Valley.",
        ),
    )

    resume_run("demo", non_interactive=True, store=store)

    qa = store.read_json("demo", "qa:1")
    assert [item["rule_id"] for item in qa] == [
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "R6",
        "R7",
        "R8",
        "R9",
    ]


def test_resume_rejects_invalid_profile_before_model_work(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    store.create("demo", "Lila searches.", Manifest(slug="demo", length="short"))

    try:
        resume_run("demo", profile="invalid", store=store)
    except ValueError as exc:
        assert "profile must be one of" in str(exc)
    else:
        raise AssertionError("invalid profile should fail")


def test_hard_qa_revision_loop_rewrites_failed_chapter(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    manifest = Manifest(slug="demo")
    store.create("demo", "Lila searches.", manifest)
    world_bible = WorldBible(
        logline="Lila searches the Flooded Valley.",
        characters=[Character(name="Lila", role="protagonist")],
        locations=[Location(name="Flooded Valley", description="A drowned valley.")],
    )
    context = StageContext(
        slug="demo",
        idea="Lila searches.",
        length=StoryLength.SHORT,
        embellish_probability=0,
        allow_paid=False,
        artifacts={"world_bible": world_bible},
    )
    failed = Chapter(
        number=1,
        title="Bad",
        prose="In the room, Mara entered the Flooded Valley.",
    )
    fixed = Chapter(
        number=1,
        title="Bad",
        prose="Lila entered the Flooded Valley.",
    )

    def fake_run_stage(*args: Any, **kwargs: Any) -> Chapter:
        _ = args, kwargs
        return fixed

    monkeypatch.setattr("story_writer.orchestrator._run_stage", fake_run_stage)

    revised = _revise_hard_qa_failures(
        failed,
        prose_stage=None,
        context=context,
        manifest=manifest,
        store=store,
        prior_chapters=[],
    )

    assert revised.prose == fixed.prose
    assert context.artifacts["qa_feedback"] == ""


def test_hard_qa_revision_loop_gets_second_attempt(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    store = RunStore(tmp_path)
    manifest = Manifest(slug="demo")
    store.create("demo", "Lila searches.", manifest)
    world_bible = WorldBible(
        logline="Lila searches the Flooded Valley.",
        characters=[Character(name="Lila", role="protagonist")],
        locations=[Location(name="Flooded Valley", description="A drowned valley.")],
    )
    context = StageContext(
        slug="demo",
        idea="Lila searches.",
        length=StoryLength.SHORT,
        embellish_probability=0,
        allow_paid=False,
        artifacts={"world_bible": world_bible},
    )
    failed = Chapter(number=1, title="Bad", prose="In the room, Mara waited.")
    still_failed = Chapter(number=1, title="Bad", prose="In the room, Mara stayed.")
    fixed = Chapter(number=1, title="Bad", prose="Lila waited in the Flooded Valley.")
    attempts = [still_failed, fixed]

    def fake_run_stage(*args: Any, **kwargs: Any) -> Chapter:
        _ = args, kwargs
        return attempts.pop(0)

    monkeypatch.setattr("story_writer.orchestrator._run_stage", fake_run_stage)

    revised = _revise_hard_qa_failures(
        failed,
        prose_stage=None,
        context=context,
        manifest=manifest,
        store=store,
        prior_chapters=[],
    )

    assert revised.prose == fixed.prose
    assert attempts == []


def test_unknown_name_fallback_removes_remaining_r5_spans() -> None:
    chapter = Chapter(
        number=1,
        title="Bad",
        prose="The mist whispered Elias before Lila could answer.",
    )

    revised = _remove_unknown_names(
        chapter,
        [
            RuleResult(
                rule_id="R5",
                passed=False,
                severity="hard",
                message="Unknown proper nouns found.",
                spans=["Elias"],
            )
        ],
    )

    assert "Elias" not in revised.prose
    assert "a name she could not read" in revised.prose
