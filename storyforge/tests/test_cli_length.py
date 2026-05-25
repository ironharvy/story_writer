"""Length presets: alias resolution and the eval-band invariant.

The presets are named after publishing word-count categories. The key
invariant: every preset's words-per-chapter must sit inside the eval's
800-2500w band, so picking a preset never auto-fails Tier-1 length (T1.1) —
bigger stories scale by chapter count, not chapter size.
"""
from __future__ import annotations

from storyforge import cli
from storyforge.evaluate.tier1 import BAND_HIGH, BAND_LOW


def test_aliases_resolve_to_canonical():
    assert cli._length_dims("standard") == cli.LENGTH_PRESETS["novelette"]
    assert cli._length_dims("long") == cli.LENGTH_PRESETS["novella"]


def test_canonical_names_resolve():
    assert cli._length_dims("novella") == cli.LENGTH_PRESETS["novella"]
    assert cli._length_dims("flash") == cli.LENGTH_PRESETS["flash"]


def test_unknown_or_empty_falls_back_to_default():
    assert cli._length_dims("zzz") == cli.LENGTH_PRESETS[cli.DEFAULT_LENGTH]
    assert cli._length_dims(None) == cli.LENGTH_PRESETS[cli.DEFAULT_LENGTH]


def test_every_preset_stays_within_eval_band():
    for name, (nc, wpc) in cli.LENGTH_PRESETS.items():
        assert nc >= 1, name
        assert BAND_LOW <= wpc <= BAND_HIGH, f"{name}: {wpc}w out of {BAND_LOW}-{BAND_HIGH} band"


def test_help_string_mentions_presets_and_word_hints():
    h = cli._length_help()
    assert "novella" in h and "epic" in h
    assert "w" in h  # word-count hints present
