from qa import (
    PROPER_NOUN_RE,
    canonical_name,
    character_bullets,
    check_chapter_band,
    check_chapter_length,
    check_character_presence,
    check_name_drift,
    check_placeholder_protagonist,
    check_structure,
    split_chapters,
)


def _story(chapters: dict[str, str], characters_block: str = "") -> str:
    """Build a minimal story.md fixture."""
    parts = [
        "# Story",
        "",
        "## World bible",
        "",
        "### Characters",
        "",
        characters_block,
        "",
        "## Final Story",
        "",
    ]
    for title, body in chapters.items():
        parts.append(f"### {title}")
        parts.append("")
        parts.append(body)
        parts.append("")
    return "\n".join(parts)


# --- canonical_name + character_bullets -------------------------------------

def test_canonical_name_strips_surrounding_bold():
    assert canonical_name("**Cinder** — the protagonist") == "Cinder"
    assert canonical_name("**Cinder**, the protagonist") == "Cinder"
    assert canonical_name("*Cinder*, the protagonist") == "Cinder"
    # Plain name unchanged.
    assert canonical_name("Cinder, the protagonist") == "Cinder"
    # No delimiter, whole bullet treated as name.
    assert canonical_name("**Cinder**") == "Cinder"


def test_character_bullets_skips_bold_only_motivations():
    story = _story(
        {"Chapter 1: Open": "Cinder walked."},
        characters_block="\n".join([
            "1. **Reclaim Identity**",
            "2. **Protect the Nursery**",
            "3. Cinder, the protagonist",
            "4. Renn, an ally",
        ]),
    )
    bullets = character_bullets(story)
    assert bullets == ["Cinder, the protagonist", "Renn, an ally"]


def test_character_bullets_keeps_bold_name_with_description():
    story = _story(
        {"Chapter 1": "Cinder walked."},
        characters_block="\n".join([
            "1. **Cinder** — the protagonist",
            "2. **Renn**, an ally",
        ]),
    )
    bullets = character_bullets(story)
    assert bullets == ["**Cinder** — the protagonist", "**Renn**, an ally"]
    names = [canonical_name(b) for b in bullets]
    assert names == ["Cinder", "Renn"]


# --- proper-noun matcher ----------------------------------------------------

def test_proper_noun_regex_does_not_span_paragraph_break():
    text = "Renn paused.\n\nThe Sanctum was empty."
    matches = [m.group(1) for m in PROPER_NOUN_RE.finditer(text)]
    # 'Renn' on its own and 'The Sanctum' on its own, never merged.
    assert "Renn" in matches
    assert "The Sanctum" in matches
    assert "Renn The Sanctum" not in matches
    assert "Renn\n\nThe Sanctum" not in matches


def test_proper_noun_regex_still_matches_multiword_names():
    text = "Soren Vael drew his blade. High Cardinal Vane watched."
    matches = [m.group(1) for m in PROPER_NOUN_RE.finditer(text)]
    assert "Soren Vael" in matches
    assert "High Cardinal Vane" in matches


# --- check_name_drift -------------------------------------------------------

def test_name_drift_is_warn_not_fail():
    story = _story(
        {"Chapter 1": "High Clergy gathered. Soren the Ash watched."},
        characters_block="\n".join([
            "1. High Cardinal Vane, the antagonist",
            "2. Soren Vael, the warrior",
        ]),
    )
    findings = check_name_drift(story)
    drifts = [f for f in findings if f.check == "name_drift" and f.severity != "info"]
    assert drifts, "expected at least one drift finding"
    for f in drifts:
        assert f.severity == "warn"


def test_name_drift_clean_story_reports_info():
    story = _story(
        {"Chapter 1": "Cinder walked. Renn followed."},
        characters_block="\n".join([
            "1. Cinder, the protagonist",
            "2. Renn, an ally",
        ]),
    )
    findings = check_name_drift(story)
    severities = {f.severity for f in findings}
    assert "fail" not in severities
    assert "warn" not in severities


# --- check_character_presence -----------------------------------------------

def test_character_presence_ignores_bold_only_motivations():
    # The motivation bullet would previously be reported as missing.
    story = _story(
        {"Chapter 1": "Cinder walked. Renn followed."},
        characters_block="\n".join([
            "1. **Reclaim Identity**",
            "2. Cinder, the protagonist",
            "3. Renn, an ally",
        ]),
    )
    findings = check_character_presence(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert fails == []


def test_character_presence_strips_bold_for_lookup():
    story = _story(
        {"Chapter 1": "Cinder walked. Renn followed."},
        characters_block="\n".join([
            "1. **Cinder** — the protagonist",
            "2. **Renn**, an ally",
        ]),
    )
    findings = check_character_presence(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert fails == []


# --- check_chapter_length ---------------------------------------------------

def _make_prose(words: int) -> str:
    return " ".join(["alpha"] * words) + "."


def test_chapter_length_passes_when_all_long_enough():
    body = _make_prose(120)
    story = _story({"Chapter 1": body, "Chapter 2": body})
    findings = check_chapter_length(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert fails == []
    assert any(f.severity == "info" for f in findings)


def test_chapter_length_fails_empty_chapter():
    story = _story({
        "Chapter 1": _make_prose(120),
        "Chapter 2": "",
        "Chapter 3": _make_prose(120),
    })
    findings = check_chapter_length(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "Chapter 2" in fails[0].message


def test_chapter_length_fails_short_chapter():
    story = _story({
        "Chapter 1": _make_prose(120),
        "Chapter 2": _make_prose(40),  # under threshold
    })
    findings = check_chapter_length(story, min_words=80)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "40 words" in fails[0].message


def test_chapter_length_no_chapters_no_finding():
    # No chapters at all -> no info finding, no fail finding.
    story = "# Story\n\n## World bible\n\n## Final Story\n"
    findings = check_chapter_length(story)
    assert findings == []


# --- split_chapters sanity --------------------------------------------------

def test_split_chapters_round_trip():
    body1 = _make_prose(120)
    body2 = _make_prose(150)
    story = _story({"Chapter 1: Open": body1, "Chapter 2: Mid": body2})
    chapters = split_chapters(story)
    assert list(chapters.keys()) == ["Chapter 1: Open", "Chapter 2: Mid"]
    assert chapters["Chapter 1: Open"].startswith("alpha")


# --- check_structure --------------------------------------------------------

def test_structure_passes_complete_story():
    story = _story({"Chapter 1": _make_prose(120)})
    findings = check_structure(story)
    assert [f.severity for f in findings] == ["info"]


def test_structure_fails_missing_final_story():
    story = "\n".join([
        "# Story",
        "",
        "## World bible",
        "",
        "### Characters",
        "",
        "1. Cinder, the protagonist",
    ])
    findings = check_structure(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "Final Story" in fails[0].message


def test_structure_fails_final_story_without_chapters():
    story = _story({})  # has the headings but no '### Chapter' bodies
    findings = check_structure(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert any("no '### Chapter'" in f.message for f in fails)


# --- check_placeholder_protagonist ------------------------------------------

def test_placeholder_protagonist_fails_on_literal():
    story = _story({"Chapter 1": "The protagonist drew a blade. " + _make_prose(80)})
    findings = check_placeholder_protagonist(story)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "Chapter 1" in fails[0].message


def test_placeholder_protagonist_passes_clean_prose():
    story = _story({"Chapter 1": "Cinder drew a blade. " + _make_prose(80)})
    findings = check_placeholder_protagonist(story)
    assert [f.severity for f in findings] == ["info"]


# --- check_chapter_band -----------------------------------------------------

def test_chapter_band_warns_thin_chapter():
    story = _story({"Chapter 1": _make_prose(120)})  # >= 80 floor, < 300 band
    findings = check_chapter_band(story)
    warns = [f for f in findings if f.severity == "warn"]
    assert len(warns) == 1
    assert "thin" in warns[0].message


def test_chapter_band_warns_bloated_chapter():
    story = _story({"Chapter 1": _make_prose(2600)})
    findings = check_chapter_band(story)
    warns = [f for f in findings if f.severity == "warn"]
    assert len(warns) == 1
    assert "bloated" in warns[0].message


def test_chapter_band_passes_in_band():
    story = _story({"Chapter 1": _make_prose(400), "Chapter 2": _make_prose(900)})
    findings = check_chapter_band(story)
    assert [f.severity for f in findings] == ["info"]


def test_chapter_band_skips_below_hard_floor():
    # A 40-word chapter is a hard FAIL elsewhere; the band must not also warn
    # it as "thin" (no double-reporting).
    story = _story({"Chapter 1": _make_prose(40)})
    findings = check_chapter_band(story)
    assert [f.severity for f in findings] == ["info"]
