from core.types import WorldState, render_world_state


def test_world_state_recent_events_defaults_empty():
    state = WorldState(
        story_clock="Opening",
        characters=["Mara: at home, knows nothing"],
        locations=["The cottage: intact"],
        plot_threads=["Who sent the letter?"],
        key_objects=["The sealed letter: in Mara's pocket"],
    )
    assert state.recent_events == []


def test_render_world_state_includes_all_sections_and_handles_empty():
    state = WorldState(
        story_clock="Day 3, dusk",
        characters=["Mara: in the city, suspicious of To"],
        locations=[],
        plot_threads=["Who sent the letter?"],
        key_objects=["The sealed letter: lost"],
        recent_events=["Ch.1: Mara left home"],
    )
    rendered = render_world_state(state)
    assert "Day 3, dusk" in rendered
    assert "Mara: in the city" in rendered
    assert "Ch.1: Mara left home" in rendered
    # Empty list renders a placeholder rather than nothing.
    assert "- (none)" in rendered
