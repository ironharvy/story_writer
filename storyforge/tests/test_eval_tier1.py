"""Regression tests for the deterministic (Tier-1) eval checks.

Each defect listed in STORY_EVAL_SPEC.md's regression list must keep failing the
right check. Tests call individual checks so a fixture's targeted defect is
asserted without other checks masking it.
"""
from __future__ import annotations

from pathlib import Path

from storyforge.evaluate import FAIL, PASS, WARN
from storyforge.evaluate import parse, tier1

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> parse.Manuscript:
    return parse.parse_manuscript((FIX / name).read_text())


# --- parser -------------------------------------------------------------------

def test_parser_basic_shape():
    m = load("clean_small.md")
    assert m.title == "The Clockmaker's Apprentice"
    assert "apprentice who can hear" in m.premise
    assert [c.canonical for c in m.characters] == ["Tamsin", "Orrin"]
    assert len(m.chapters) == 2
    assert m.chapters[0].label.startswith("Chapter 1")
    assert m.chapters[0].word_count > 40


def test_canonical_name_rules():
    assert parse.canonical_name("1. **Cinder**, a runaway") == "Cinder"
    assert parse.canonical_name("- Kaelen Vey — a soldier") == "Kaelen Vey"
    assert parse.canonical_name("* _Mira_: the bellringer") == "Mira"
    # motivation-as-name (fully emphasized, no description) is skipped
    assert parse.canonical_name("3. **Reclaim Identity**") is None


# --- T1.1 length --------------------------------------------------------------

def test_t1_1_empty_chapter_fails():
    m = load("empty_chapter.md")
    c = tier1.t1_1_length(m)
    assert c.severity == FAIL
    counts = c.details["counts"]
    # the empty chapter is present and has zero words
    assert any(v == 0 for v in counts.values())


# --- T1.2 character presence --------------------------------------------------

def test_t1_2_missing_character_fails():
    m = load("missing_character.md")
    # the motivation bullet must not be parsed as a character
    names = [c.canonical for c in m.characters]
    assert "Wren" in names and "Bramble" in names
    assert not any("Reclaim" in n for n in names)
    c = tier1.t1_2_presence(m)
    assert c.severity == FAIL
    assert "Bramble" in c.details["missing"]
    assert "Wren" not in c.details["missing"]


def test_t1_2_clean_passes():
    assert tier1.t1_2_presence(load("clean_small.md")).severity == PASS


# --- T1.3 name drift ----------------------------------------------------------

def test_t1_3_name_drift_warns():
    c = tier1.t1_3_name_drift(load("name_drift.md"))
    assert c.severity == WARN
    variants = {f["variant"] for f in c.details["flags"]}
    assert "Cindar" in variants


def test_t1_3_clean_passes():
    assert tier1.t1_3_name_drift(load("clean_small.md")).severity == PASS


# --- T1.4 phrase reuse --------------------------------------------------------

def test_t1_4_phrase_reuse_warns():
    c = tier1.t1_4_phrase_reuse(load("phrase_reuse.md"))
    assert c.severity == WARN
    assert c.details["count"] > tier1.PHRASE_REUSE_BUDGET


def test_t1_4_clean_passes():
    assert tier1.t1_4_phrase_reuse(load("clean_small.md")).severity == PASS


# --- T1.5 over-repetition -----------------------------------------------------

def _build_over_repetition() -> str:
    """Build a manuscript where 'veins' is over-repeated but ranks below the
    exempted core vocabulary, then persist it as a fixture. Several core words
    clearly out-rank 'veins' so it falls outside the exempt set regardless of
    tie-ordering."""
    core = "held fast through gale".split()  # appear ~12x each -> dominate, get exempted
    body1 = " ".join("The lantern held fast through the gale." for _ in range(12))
    # 'veins' appears 9 times, above the floor threshold of 8 for this length
    veins = " ".join("Frost spread thin like cold veins on glass." for _ in range(9))
    md = f"""# The Frostbound Ship

## Premise
A navigator aboard a freezing ship must thaw the frozen helm before the vessel drifts past the edge of the charted world.

## World / Characters
1. Sable, the ship's navigator
2. Dunmore, the iron-handed bosun

## Manuscript

### Chapter 1: The Drift
Sable watched as Dunmore barked orders. {body1}

### Chapter 2: The Thaw
{veins} Sable freed the helm at dawn.
"""
    assert core  # documents intent: these become the exempted high-frequency core
    (FIX / "over_repetition.md").write_text(md)
    return md


def test_t1_5_over_repetition_warns():
    _build_over_repetition()
    m = load("over_repetition.md")
    c = tier1.t1_5_over_repetition(m)
    assert c.severity == WARN
    flagged = {w for w, _ in c.details["flagged"]}
    assert "veins" in flagged


def test_t1_5_clean_passes():
    assert tier1.t1_5_over_repetition(load("clean_small.md")).severity == PASS
