from pov_check import ChapterPOV, _coerce, check_pov_consistency


def _chapters(*titles: str) -> dict[str, str]:
    return {t: f"prose for {t}" for t in titles}


# --- _coerce ----------------------------------------------------------------

def test_coerce_from_dataclass_dict_and_object():
    assert _coerce(ChapterPOV("Chapter 1", "first_person", "uses I")) == ChapterPOV(
        "Chapter 1", "first_person", "uses I"
    )
    assert _coerce({"chapter": "Chapter 2", "pov": "third_person", "note": ""}) == ChapterPOV(
        "Chapter 2", "third_person", ""
    )

    class _Obj:
        chapter = "Chapter 3"
        pov = "Third Person"  # normalised to third_person
        note = "names + she/he"

    assert _coerce(_Obj()) == ChapterPOV("Chapter 3", "third_person", "names + she/he")


def test_coerce_normalises_and_defaults_unknown_label():
    assert _coerce({"chapter": "C1", "pov": "First-Person"}).pov == "first_person"
    assert _coerce({"chapter": "C1", "pov": "omniscient"}).pov == "other"


def test_coerce_drops_item_without_chapter():
    assert _coerce({"pov": "first_person"}) is None
    assert _coerce({"chapter": "", "pov": "first_person"}) is None


# --- check_pov_consistency --------------------------------------------------

def test_uniform_third_person_passes():
    chapters = _chapters("Chapter 1", "Chapter 2", "Chapter 3")
    cls = [ChapterPOV(t, "third_person") for t in chapters]
    findings = check_pov_consistency(chapters, cls)
    assert [f.severity for f in findings] == ["info"]
    assert "third_person" in findings[0].message


def test_uniform_first_person_passes():
    chapters = _chapters("Chapter 1", "Chapter 2")
    cls = [ChapterPOV(t, "first_person") for t in chapters]
    findings = check_pov_consistency(chapters, cls)
    assert all(f.severity == "info" for f in findings)


def test_single_first_person_outlier_fails_that_chapter():
    chapters = _chapters("Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4")
    cls = [
        ChapterPOV("Chapter 1", "third_person"),
        ChapterPOV("Chapter 2", "third_person"),
        ChapterPOV("Chapter 3", "first_person", "narrator says 'I'"),
        ChapterPOV("Chapter 4", "third_person"),
    ]
    findings = check_pov_consistency(chapters, cls)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "Chapter 3" in fails[0].message
    assert "first_person" in fails[0].message
    assert "dominantly third_person" in fails[0].message
    assert "narrator says 'I'" in fails[0].message


def test_mixed_chapter_fails_on_its_own():
    chapters = _chapters("Chapter 1", "Chapter 2", "Chapter 3")
    cls = [
        ChapterPOV("Chapter 1", "third_person"),
        ChapterPOV("Chapter 2", "third_person"),
        ChapterPOV("Chapter 3", "mixed", "starts third, ends first"),
    ]
    findings = check_pov_consistency(chapters, cls)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 1
    assert "Chapter 3" in fails[0].message
    assert "shifts narrative POV" in fails[0].message


def test_other_chapters_are_not_flagged():
    chapters = _chapters("Chapter 1", "Chapter 2", "Chapter 3")
    cls = [
        ChapterPOV("Chapter 1", "third_person"),
        ChapterPOV("Chapter 2", "third_person"),
        ChapterPOV("Chapter 3", "other", "too short to tell"),
    ]
    findings = check_pov_consistency(chapters, cls)
    assert [f.severity for f in findings] == ["info"]


def test_unclassified_chapter_warns():
    chapters = _chapters("Chapter 1", "Chapter 2", "Chapter 3")
    cls = [
        ChapterPOV("Chapter 1", "third_person"),
        ChapterPOV("Chapter 2", "third_person"),
        # Chapter 3 missing from the model's output.
    ]
    findings = check_pov_consistency(chapters, cls)
    warns = [f for f in findings if f.severity == "warn"]
    assert len(warns) == 1
    assert "Chapter 3" in warns[0].message
    assert "not classified" in warns[0].message


def test_all_other_reports_undetermined():
    chapters = _chapters("Chapter 1", "Chapter 2")
    cls = [ChapterPOV(t, "other") for t in chapters]
    findings = check_pov_consistency(chapters, cls)
    assert [f.severity for f in findings] == ["info"]
    assert "undetermined" in findings[0].message


def test_empty_classifications_warns():
    chapters = _chapters("Chapter 1")
    findings = check_pov_consistency(chapters, [])
    assert [f.severity for f in findings] == ["warn"]


def test_no_chapters_no_findings():
    assert check_pov_consistency({}, []) == []


def test_tie_picks_a_dominant_and_flags_the_other_side():
    # 2 first, 2 third — Counter.most_common is deterministic by insertion order,
    # so the first-seen label wins; the opposite-POV chapters become outliers.
    chapters = _chapters("Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4")
    cls = [
        ChapterPOV("Chapter 1", "third_person"),
        ChapterPOV("Chapter 2", "third_person"),
        ChapterPOV("Chapter 3", "first_person"),
        ChapterPOV("Chapter 4", "first_person"),
    ]
    findings = check_pov_consistency(chapters, cls)
    fails = [f for f in findings if f.severity == "fail"]
    assert len(fails) == 2
