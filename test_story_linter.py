from story_linter import (
    Replacement,
    apply,
    replacements_to_json,
    validate,
)


SAMPLE_STORY = """# Story

## Generation Parameters

- model: `test`

## World bible

### Characters

1. Cinder, the protagonist
2. Renn, an ally

#### Character 1

The protagonist of the story.

## Final Story

### Chapter 1: The Cold Open

The protagonist crouched by the embers. The child had not slept.

### Chapter 2: Naming

Cinder rose at dawn. Cindar, as the others sometimes called her, said nothing.
"""


def test_apply_replaces_only_in_final_story_region():
    # Matching is case-sensitive on purpose; the linter is expected to emit a
    # separate entry per casing observed in the prose.
    replacements = [
        Replacement(find="the protagonist", replace="Cinder", reason="placeholder"),
        Replacement(find="The protagonist", replace="Cinder", reason="placeholder"),
        Replacement(find="The child", replace="Cinder", reason="placeholder"),
        Replacement(find="Cindar", replace="Cinder", reason="misspelling"),
    ]
    rewritten, counts = apply(SAMPLE_STORY, replacements)

    # World-bible / Characters section is untouched.
    bible_marker = "1. Cinder, the protagonist"
    assert bible_marker in rewritten
    assert "The protagonist of the story." in rewritten

    # Chapter 1 placeholders are replaced.
    assert "The protagonist crouched" not in rewritten
    assert "Cinder crouched by the embers." in rewritten
    assert "Cinder had not slept." in rewritten

    # Chapter 2 misspelling fixed.
    assert "Cindar" not in rewritten.split("## Final Story", 1)[1]

    counts_by_find = {r.find: n for r, n in counts}
    assert counts_by_find["The protagonist"] == 1
    assert counts_by_find["the protagonist"] == 0  # lowercase doesn't occur in ch1
    assert counts_by_find["The child"] == 1
    assert counts_by_find["Cindar"] == 1


def test_apply_word_boundary_does_not_clobber_substrings():
    story = (
        "# Story\n\n## Final Story\n\n### Chapter 1\n\n"
        "Cind was a nickname; Cinder was her real name.\n"
    )
    rewritten, counts = apply(
        story,
        [Replacement(find="Cind", replace="Cinder", reason="x")],
    )
    # The substring "Cind" inside "Cinder" must not match.
    assert "Cindererererer" not in rewritten
    assert "Cinderer" not in rewritten
    assert rewritten.count("Cinder") == 2  # one for the original "Cinder", one rewritten "Cind"
    assert counts[0][1] == 1


def test_apply_no_final_story_heading_is_noop():
    story = "# Story\n\nNo Final Story section here.\n"
    rewritten, counts = apply(
        story,
        [Replacement(find="No", replace="Yes", reason="x")],
    )
    assert rewritten == story
    assert counts[0][1] == 0


def test_apply_stops_at_next_h2_after_final_story():
    story = (
        "# Story\n\n## Final Story\n\n"
        "### Chapter 1\n\nThe protagonist walked.\n\n"
        "## Notes\n\nThe protagonist appendix should NOT be rewritten.\n"
    )
    rewritten, _ = apply(
        story,
        [Replacement(find="The protagonist", replace="Cinder", reason="x")],
    )
    assert "Cinder walked." in rewritten
    assert "The protagonist appendix should NOT be rewritten." in rewritten


def test_validate_drops_empty_and_self_replacements():
    chapter_prose = "Cinder walked. Cindar later."
    out = validate(
        [
            Replacement(find="", replace="X", reason=""),
            Replacement(find="X", replace="", reason=""),
            Replacement(find="Cinder", replace="Cinder", reason="self"),
            Replacement(find="Cindar", replace="Cinder", reason="ok"),
            Replacement(find="Nowhere", replace="Cinder", reason="not in prose"),
        ],
        chapter_prose,
    )
    assert [r.find for r in out] == ["Cindar"]


def test_validate_deduplicates_identical_pairs():
    chapter_prose = "the protagonist and the protagonist"
    out = validate(
        [
            Replacement(find="the protagonist", replace="Cinder", reason="a"),
            Replacement(find="the protagonist", replace="Cinder", reason="b"),
        ],
        chapter_prose,
    )
    assert len(out) == 1


def test_replacements_to_json_with_counts():
    payload = replacements_to_json(
        [Replacement(find="x", replace="y", reason="z")],
        counts=[(Replacement(find="x", replace="y", reason="z"), 3)],
    )
    assert '"applied": 3' in payload
    assert '"find": "x"' in payload
