"""Unit tests for the deterministic canon metrics.

Seeded with the REAL bugs found in the manual continuity review as gold
negatives (each must be flagged by exactly the metric named in the brief), plus
good cases that must stay clean.
"""
import canon as canon_mod
import canon_metrics as cm

CANON = canon_mod.load_canon(canon_mod.DEFAULT_CANON_PATH)


def _story(chapters: dict[str, str]) -> str:
    """Build a minimal story.md fixture with the chapters under ## Final Story."""
    parts = ["# Story", "", "## World bible", "", "### Characters", "", "## Final Story", ""]
    for title, body in chapters.items():
        parts += [f"### {title}", "", body, ""]
    return "\n".join(parts)


def _fails(findings, check):
    return [f for f in findings if f.check == check and f.severity == "fail"]


# --- scaffolding leak --------------------------------------------------------

def test_scaffolding_leak_flags_chapter_reference():
    # "The breaking was Ch.19's. He had no Ch.19." -> scaffolding-leak FAIL
    story = _story({"Chapter 18: The Break": "The breaking was Ch.19's. He had no Ch.19."})
    findings = cm.check_scaffolding_leak(story, CANON)
    assert _fails(findings, "scaffolding_leak")


def test_scaffolding_leak_flags_pov_and_beat():
    story = _story({"Chapter 2: Plan": "She kept a clear POV. Beat 3 arrived early."})
    assert _fails(cm.check_scaffolding_leak(story, CANON), "scaffolding_leak")


def test_scaffolding_leak_allows_device_phrase():
    # GOOD: "The reader sees her stand." is sanctioned via device_allowlist
    story = _story({"Chapter 5: Stand": "The reader sees her stand. She did not move."})
    findings = cm.check_scaffolding_leak(story, CANON)
    assert not _fails(findings, "scaffolding_leak")
    assert any(f.severity == "info" for f in findings)


def test_scaffolding_leak_clean_prose_is_info():
    story = _story({"Chapter 1: Open": "Snow fell on the quiet courtyard and she waited."})
    findings = cm.check_scaffolding_leak(story, CANON)
    assert not _fails(findings, "scaffolding_leak")


# --- name-arc ----------------------------------------------------------------

def test_name_arc_flags_marta_saying_elias_before_ch17():
    # Marta says "Elias" in a pre-Ch.17 fixture -> name-arc FAIL
    story = _story({
        "Chapter 10: The Warning": 'Marta caught his sleeve. "Elias," she whispered, "run."',
    })
    assert _fails(cm.check_name_arc(story, CANON), "name_arc")


def test_name_arc_flags_name_before_earliest_narration():
    # Severin may not appear until ch.11
    story = _story({"Chapter 3: Early": "Severin walked the long hall alone."})
    assert _fails(cm.check_name_arc(story, CANON), "name_arc")


def test_name_arc_allows_name_after_earliest():
    story = _story({"Chapter 11: Named": "Severin walked the long hall alone."})
    assert not _fails(cm.check_name_arc(story, CANON), "name_arc")


def test_name_arc_allows_elias_in_narration_after_ch7_when_marta_silent():
    story = _story({"Chapter 8: After": "Elias crossed the bridge. No one spoke his name."})
    assert not _fails(cm.check_name_arc(story, CANON), "name_arc")


# --- canon-fact (locked appearance) -----------------------------------------

def test_canon_fact_flags_blue_eyes():
    # "his blue eyes" -> canon-fact FAIL (locked eyes=grey)
    story = _story({"Chapter 1: Open": "She studied his blue eyes for a long moment."})
    assert _fails(cm.check_canon_fact(story, CANON), "canon_fact")


def test_canon_fact_flags_eyes_were_green():
    story = _story({"Chapter 2: Look": "His eyes were green in the lamplight."})
    assert _fails(cm.check_canon_fact(story, CANON), "canon_fact")


def test_canon_fact_allows_no_horns_near_protagonist():
    # GOOD: "No horns" near protagonist -> appearance OK (no transformation)
    story = _story({"Chapter 1: Open": "No horns, no wings — his grey eyes, the same as ever."})
    findings = cm.check_canon_fact(story, CANON)
    assert not _fails(findings, "canon_fact")


def test_canon_fact_allows_correct_grey_eyes():
    story = _story({"Chapter 1: Open": "His grey eyes caught the light; brown hair at his nape."})
    assert not _fails(cm.check_canon_fact(story, CANON), "canon_fact")


# --- required artifact -------------------------------------------------------

def test_required_artifact_flags_missing_hymn_in_ch4():
    story = _story({"Chapter 4: The Rite": "They gathered in silence and then dispersed."})
    assert _fails(cm.check_required_artifact(story, CANON), "required_artifact")


def test_required_artifact_passes_when_hymn_present():
    story = _story({"Chapter 4: The Rite": "A low hymn rose from the stalls as they gathered."})
    assert not _fails(cm.check_required_artifact(story, CANON), "required_artifact")


def test_required_artifact_ignores_absent_chapter():
    # Chapter 4 not in the manuscript -> not a failure.
    story = _story({"Chapter 1: Open": "Nothing required here."})
    assert not _fails(cm.check_required_artifact(story, CANON), "required_artifact")


# --- timeline / age ----------------------------------------------------------

def test_timeline_flags_eight_year_span():
    # "...the next eight years" (span=5) -> timeline FAIL
    story = _story({"Chapter 9: After": "Over the next eight years, the city forgot her."})
    assert _fails(cm.check_timeline_age(story, CANON), "timeline_age")


def test_timeline_allows_five_year_span():
    story = _story({"Chapter 9: After": "Five years from the flight to the end, no longer."})
    assert not _fails(cm.check_timeline_age(story, CANON), "timeline_age")


def test_timeline_allows_digit_span_within_bounds():
    story = _story({"Chapter 9: After": "It had been 3 years since the flight."})
    assert not _fails(cm.check_timeline_age(story, CANON), "timeline_age")


# --- runner ------------------------------------------------------------------

def test_run_all_aggregates_and_separates_good_from_bad():
    bad = _story({
        "Chapter 4: The Rite": "His blue eyes. The next eight years. He had no Ch.19.",
    })
    findings = cm.run_all(bad, CANON)
    flagged = {f.check for f in findings if f.severity == "fail"}
    assert {"scaffolding_leak", "canon_fact", "required_artifact", "timeline_age"} <= flagged


def test_empty_canon_metrics_are_noops():
    empty = canon_mod.Canon()
    story = _story({"Chapter 1: Open": "His blue eyes. Ch.19. The next eight years."})
    assert cm.run_all(story, empty) == []
