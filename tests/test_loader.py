"""Tests for the story-artifact loader (pure markdown parsing, no model)."""
from __future__ import annotations

from evals.loader import load_story

ARTIFACT = """# Story

## Generation Parameters

- model: `ollama/qwen3.6:27b`

## Story Title

Mages Die Young

## Core Premise

Kael hoards his lifespan in a war that ages every mage who casts.

## Spine

Beat one.
Beat two.

## World bible

### Rules of the World

- Casting spills the caster's blood.
- Every spell ages the caster.

### Locations

- The Barracks, a damp complex.
- The Scar, a ruined valley.

#### Location 1

The Barracks is a sprawling subterranean complex reeking of antiseptic and old blood.

#### Location 2

The Scar is a valley of red dust where bones line the trenches.

### Timeline

- Kael is conscripted.
- Kael executes his mentor.

### Characters

1. Kael, a young hoarder of years.
2. Elara, his sister.

#### Character 1

Kael Thorne, 19, a conscript who refuses to spend his life force.

#### Character 2

Elara, 16, his sister with latent magic.

## Final Story

### Chapter 1: The Ledger of Blood

Kael moved through the barracks like a ghost, hoarding his silence like gold.

### Chapter 2: Echoes

The Scar swallowed the light as Kael knelt in the red dust.
"""


def test_load_story_parses_all_sections(tmp_path):
    p = tmp_path / "story.md"
    p.write_text(ARTIFACT, encoding="utf-8")
    s = load_story(p)

    assert s.title == "Mages Die Young"
    assert "hoards his lifespan" in s.premise
    assert "Beat one." in s.spine and "Beat two." in s.spine

    wb = s.world_bible
    assert wb.rules_of_the_world == [
        "Casting spills the caster's blood.",
        "Every spell ages the caster.",
    ]
    assert wb.timeline == ["Kael is conscripted.", "Kael executes his mentor."]

    # enhanced blocks are preferred over the short bullets
    assert len(wb.locations) == 2
    assert "subterranean complex" in wb.locations[0]
    assert "red dust" in wb.locations[1]
    assert len(wb.characters) == 2
    assert "Kael Thorne" in wb.characters[0]

    assert list(s.chapters) == ["Chapter 1: The Ledger of Blood", "Chapter 2: Echoes"]
    assert "hoarding his silence" in s.chapters["Chapter 1: The Ledger of Blood"]


_NO_ENHANCED = """# Story

## Story Title

Mages Die Young

## World bible

### Rules of the World

- Casting spills the caster's blood.

### Locations

- The Barracks, a damp complex.
- The Scar, a ruined valley.

### Characters

1. Kael, a young hoarder of years.
2. Elara, his sister.

## Final Story

### Chapter 1: X

body text here
"""


def test_load_story_falls_back_to_bullets_without_enhanced_blocks(tmp_path):
    p = tmp_path / "story.md"
    p.write_text(_NO_ENHANCED, encoding="utf-8")
    s = load_story(p)
    # no #### blocks -> falls back to the short location bullets and character bullets
    assert s.world_bible.locations == ["The Barracks, a damp complex.", "The Scar, a ruined valley."]
    assert len(s.world_bible.characters) == 2
    assert "Kael" in s.world_bible.characters[0]
