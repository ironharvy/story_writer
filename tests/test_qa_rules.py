from __future__ import annotations

from story_writer.models.story import Chapter, Character, Location, WorldBible
from story_writer.qa.rules import detect_chapter, has_hard_failures


def _world_bible() -> WorldBible:
    return WorldBible(
        logline="Lila searches the Flooded Valley.",
        characters=[Character(name="Lila", role="protagonist")],
        locations=[Location(name="Flooded Valley", description="A drowned home.")],
    )


def test_detects_assistant_artifacts_and_unknown_names() -> None:
    chapter = Chapter(
        number=1,
        title="Opening",
        prose="Certainly! In the room, Mara enters the Flooded Valley.",
    )

    results = detect_chapter(chapter, _world_bible())

    assert has_hard_failures(results)
    assert next(result for result in results if result.rule_id == "R4").passed is False
    assert "Mara" in next(result for result in results if result.rule_id == "R5").spans


def test_clean_chapter_passes_hard_rules() -> None:
    chapter = Chapter(
        number=1,
        title="Opening",
        prose="She waited. Lila stepped into the Flooded Valley.",
    )

    results = detect_chapter(chapter, _world_bible())

    assert not has_hard_failures(results)


def test_repeated_opening_is_flagged_separately() -> None:
    prior = Chapter(
        number=1,
        title="Prior",
        prose="Lila stepped into the Flooded Valley and listened. The reeds shook.",
    )
    chapter = Chapter(
        number=2,
        title="Repeat",
        prose="Lila stepped into the Flooded Valley and listened. The rain began.",
    )

    results = detect_chapter(chapter, _world_bible(), [prior])

    repeated_opening = next(result for result in results if result.rule_id == "R9")
    assert repeated_opening.passed is False
    assert repeated_opening.severity == "soft"
