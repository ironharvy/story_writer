"""Tests for the generation-time canon injection + self-correction.

The critic (chapter_violations) and the slice renderer are deterministic and
tested without an LM. The revise step (revise_chapter) needs an LM and is
exercised only indirectly via draft_with_canon_guard's no-violation short
circuit and its non-LM failure handling.
"""
import canon as canon_mod
import canon_guard as cg

CANON = canon_mod.load_canon(canon_mod.DEFAULT_CANON_PATH)


# --- critic ------------------------------------------------------------------

def test_chapter_violations_flags_scaffolding_and_appearance():
    prose = "His blue eyes opened. He had no Ch.19 left in him."
    viols = cg.chapter_violations(prose, CANON, chapter_number=18, chapter_title="Break")
    checks = {f.check for f in viols}
    assert "scaffolding_leak" in checks
    assert "canon_fact" in checks


def test_chapter_violations_flags_name_before_allowed():
    prose = "Severin crossed the courtyard."
    viols = cg.chapter_violations(prose, CANON, chapter_number=3, chapter_title="Early")
    assert any(f.check == "name_arc" for f in viols)


def test_chapter_violations_clean_chapter_has_none():
    prose = "Snow fell on the courtyard. His grey eyes watched the gate, and he waited."
    viols = cg.chapter_violations(prose, CANON, chapter_number=2, chapter_title="Wait")
    assert viols == []


def test_chapter_violations_required_artifact_missing():
    prose = "They gathered in silence and then dispersed without a sound."
    viols = cg.chapter_violations(prose, CANON, chapter_number=4, chapter_title="Rite")
    assert any(f.check == "required_artifact" for f in viols)


# --- guard loop --------------------------------------------------------------

def test_guard_returns_draft_unchanged_when_clean():
    clean = "Snow fell. His grey eyes watched the gate; brown hair at his nape."
    out = cg.draft_with_canon_guard(
        lambda: clean, CANON, chapter_number=2, chapter_title="Wait",
    )
    assert out == clean


def test_guard_noop_under_empty_canon():
    # Empty canon -> no constraints -> draft returned verbatim even if "dirty".
    dirty = "His blue eyes. Ch.19. The next eight years."
    out = cg.draft_with_canon_guard(
        lambda: dirty, canon_mod.Canon(), chapter_number=1, chapter_title="X",
    )
    assert out == dirty


def test_guard_returns_original_when_revise_unavailable():
    # No LM configured here, so revise_chapter raises; the guard must swallow it
    # and return the (still-violating) draft rather than crash the run.
    dirty = "His blue eyes opened."
    out = cg.draft_with_canon_guard(
        lambda: dirty, CANON, chapter_number=1, chapter_title="X", max_revisions=1,
    )
    assert out == dirty


# --- slice renderer ----------------------------------------------------------

def test_render_slice_includes_appearance_pov_and_blocked_names():
    slice_text = canon_mod.render_chapter_slice(CANON, chapter=3)
    assert "grey" in slice_text                  # locked appearance
    assert "POINT OF VIEW" in slice_text         # pov rule
    assert "Severin" in slice_text               # not yet allowed by ch.3
    assert "SCAFFOLDING" in slice_text


def test_render_slice_lists_required_artifact_for_its_chapter():
    assert "hymn" in canon_mod.render_chapter_slice(CANON, chapter=4)
    assert "hymn" not in canon_mod.render_chapter_slice(CANON, chapter=5)


def test_render_slice_empty_canon_is_blank():
    assert canon_mod.render_chapter_slice(canon_mod.Canon(), chapter=1) == ""


def test_names_allowed_helpers():
    assert "Severin" in CANON.names_not_yet_allowed(10)
    assert "Severin" in CANON.names_allowed(11)
    assert CANON.required_artifact(12)["name"] == "bard song"
