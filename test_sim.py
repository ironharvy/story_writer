"""Tests for the multi-agent story simulation."""

import dspy

from story_sim.models import (
    CharacterAction,
    CharacterSheet,
    ClarifyingQuestion,
    PacingDecision,
    RoundRecord,
    RoundResolution,
    SceneContext,
    SimulationState,
    WorldState,
)
from story_sim.engine import (
    SimulationEngine,
    compile_story,
    finalize_chapter,
    format_actions,
    format_character_sheet,
    format_characters_summary,
    get_pacing_hint,
    get_personal_memory,
    get_recent_events_summary,
    get_story_progress,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_character(
    name: str = "Kael",
    role: str = "protagonist",
) -> CharacterSheet:
    return CharacterSheet(
        name=name,
        role=role,
        backstory=f"{name} grew up in the wilds.",
        personality="Brave and stubborn.",
        goals=["Survive", "Find the truth"],
        secrets=["Knows the king's weakness"],
        relationships={"Mira": "trusted ally"},
    )


def _make_world() -> WorldState:
    return WorldState(
        setting="A crumbling kingdom on the edge of a cursed forest.",
        genre="fantasy",
        tone="dark",
        locations=["Castle Vorn", "The Ashwood"],
        factions=["The Crown", "The Exiles"],
        key_facts=["Magic costs blood"],
        timeline=[],
    )


def _make_scene(characters: list[str] | None = None) -> SceneContext:
    return SceneContext(
        location="Castle Vorn courtyard",
        characters_present=characters or ["Kael", "Mira"],
        time_context="Dusk",
        situation="The guards are changing shift. Tension is high.",
    )


def _make_action(name: str = "Kael") -> CharacterAction:
    return CharacterAction(
        character_name=name,
        action="Slips past the guards toward the gate.",
        dialogue="Stay close.",
        internal_thought="This is our only chance.",
    )


def _make_resolution(
    pacing: PacingDecision = PacingDecision.CONTINUE_SCENE,
) -> RoundResolution:
    return RoundResolution(
        outcome="Kael slips through unnoticed. Mira follows.",
        world_changes=["Gate left unguarded for 2 minutes"],
        character_updates={"Kael": "Now outside the castle walls"},
        pacing=pacing,
        chapter_title="",
    )


def _make_round_record(
    round_number: int = 1,
    chapter_number: int = 1,
    pacing: PacingDecision = PacingDecision.CONTINUE_SCENE,
    characters: list[str] | None = None,
) -> RoundRecord:
    return RoundRecord(
        round_number=round_number,
        chapter_number=chapter_number,
        scene=_make_scene(characters),
        actions=[_make_action("Kael"), _make_action("Mira")],
        resolution=_make_resolution(pacing),
        narrator_prose=f"Round {round_number} prose.",
    )


def _make_state() -> SimulationState:
    return SimulationState(
        world=_make_world(),
        characters=[_make_character("Kael"), _make_character("Mira", "ally")],
        plot_outline="Inciting incident: castle escape. Climax: confrontation.",
    )


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------


class TestCharacterSheet:
    def test_minimal_creation(self) -> None:
        char = CharacterSheet(
            name="Test",
            role="protagonist",
            backstory="None",
            personality="Bold",
            goals=["Win"],
        )
        assert char.name == "Test"
        assert char.secrets == []
        assert char.relationships == {}

    def test_full_creation(self) -> None:
        char = _make_character()
        assert char.name == "Kael"
        assert len(char.secrets) == 1
        assert "Mira" in char.relationships


class TestWorldState:
    def test_creation(self) -> None:
        world = _make_world()
        assert world.genre == "fantasy"
        assert len(world.locations) == 2

    def test_defaults(self) -> None:
        world = WorldState(setting="test", genre="mystery", tone="noir")
        assert world.locations == []
        assert world.timeline == []


class TestPacingDecision:
    def test_all_values(self) -> None:
        assert len(PacingDecision) == 5
        assert PacingDecision.END_STORY.value == "end_story"


class TestClarifyingQuestion:
    def test_creation(self) -> None:
        q = ClarifyingQuestion(
            question="What genre?",
            proposed_answer="Fantasy",
        )
        assert q.question == "What genre?"


# ---------------------------------------------------------------------------
# Engine Helper Tests
# ---------------------------------------------------------------------------


class TestFormatCharacterSheet:
    def test_includes_all_fields(self) -> None:
        char = _make_character()
        text = format_character_sheet(char)
        assert "Kael" in text
        assert "protagonist" in text
        assert "Brave and stubborn" in text
        assert "Survive" in text
        assert "king's weakness" in text
        assert "Mira" in text

    def test_omits_empty_secrets(self) -> None:
        char = CharacterSheet(
            name="X",
            role="npc",
            backstory="",
            personality="",
            goals=[],
        )
        text = format_character_sheet(char)
        assert "Secrets" not in text


class TestFormatCharactersSummary:
    def test_multiple_characters(self) -> None:
        chars = [_make_character("A"), _make_character("B")]
        text = format_characters_summary(chars)
        assert "A" in text
        assert "B" in text
        assert text.count("Name:") == 2


class TestGetRecentEventsSummary:
    def test_empty_rounds(self) -> None:
        result = get_recent_events_summary([])
        assert "just begun" in result

    def test_limits_to_n(self) -> None:
        rounds = [_make_round_record(i) for i in range(1, 10)]
        result = get_recent_events_summary(rounds, limit=3)
        assert "Round 7" in result
        assert "Round 9" in result
        assert "Round 1" not in result


class TestGetPersonalMemory:
    def test_no_rounds(self) -> None:
        result = get_personal_memory("Kael", [])
        assert "No prior events" in result

    def test_filters_by_presence(self) -> None:
        r1 = _make_round_record(1, characters=["Kael", "Mira"])
        r2 = _make_round_record(2, characters=["Mira"])
        result = get_personal_memory("Kael", [r1, r2])
        assert "Round 1" in result
        assert "Round 2" not in result

    def test_shows_own_action(self) -> None:
        r = _make_round_record(1, characters=["Kael", "Mira"])
        result = get_personal_memory("Kael", [r])
        assert "You " in result


class TestFormatActions:
    def test_with_dialogue(self) -> None:
        actions = [_make_action("Kael")]
        result = format_actions(actions)
        assert "Kael:" in result
        assert 'Says: "Stay close."' in result

    def test_without_dialogue(self) -> None:
        action = CharacterAction(
            character_name="Mira",
            action="Watches silently.",
        )
        result = format_actions([action])
        assert "Says:" not in result


class TestGetStoryProgress:
    def test_format(self) -> None:
        state = _make_state()
        state.current_round = 5
        state.current_chapter = 2
        state.chapters = [("Ch1", "text")]
        result = get_story_progress(state)
        assert "Chapter 2" in result
        assert "Round 5" in result
        assert "Chapters completed: 1" in result


class TestGetPacingHint:
    def test_early_is_setup(self) -> None:
        state = _make_state()
        assert "setup" in get_pacing_hint(state)

    def test_mid_is_rising(self) -> None:
        state = _make_state()
        state.rounds = [_make_round_record(i) for i in range(5)]
        assert "rising" in get_pacing_hint(state)

    def test_late_is_resolution(self) -> None:
        state = _make_state()
        state.rounds = [_make_round_record(i) for i in range(35)]
        assert "resolution" in get_pacing_hint(state)


class TestFinalizeChapter:
    def test_collects_prose(self) -> None:
        state = _make_state()
        state.rounds = [
            _make_round_record(1, chapter_number=1),
            _make_round_record(2, chapter_number=1),
        ]
        finalize_chapter(state, "The Escape")
        assert len(state.chapters) == 1
        assert state.chapters[0][0] == "The Escape"
        assert "Round 1 prose" in state.chapters[0][1]
        assert "Round 2 prose" in state.chapters[0][1]
        assert state.current_chapter == 2

    def test_default_title(self) -> None:
        state = _make_state()
        state.rounds = [_make_round_record(1, chapter_number=1)]
        finalize_chapter(state, "")
        assert state.chapters[0][0] == "Chapter 1"


class TestCompileStory:
    def test_formats_chapters(self) -> None:
        state = _make_state()
        state.chapters = [
            ("The Escape", "They fled."),
            ("The Forest", "Darkness swallowed them."),
        ]
        story = compile_story(state)
        assert "### The Escape" in story
        assert "### The Forest" in story
        assert "They fled." in story

    def test_empty_chapters(self) -> None:
        state = _make_state()
        assert compile_story(state) == ""


# ---------------------------------------------------------------------------
# MockLM for simulation tests
# ---------------------------------------------------------------------------


class SimMockLM(dspy.LM):
    """Mock LM that returns valid JSON for simulation agent signatures."""

    def __init__(self) -> None:
        super().__init__(model="mock")

    def __call__(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        **kwargs: object,
    ) -> list[str]:
        content = self._extract_content(messages) if messages else (prompt or "")
        return [self._route_response(content)]

    def _extract_content(self, messages: list) -> str:
        """Combine all message content for pattern matching."""
        return "\n".join(m.get("content", "") for m in messages)

    def _route_response(self, content: str) -> str:
        if "is_chapter_start" in content or "is_chapter_end" in content:
            return self._narrate_response()
        if "story_progress" in content and "character_actions" in content:
            return self._resolution_response()
        if "personal_memory" in content and "character_sheet" in content:
            return self._action_response()
        if "pacing_hint" in content:
            return self._scene_response()
        if "plot_outline" in content and "inciting" in content.lower():
            return self._plot_response()
        if "CharacterSheet" in content and "num_characters" in content:
            return self._characters_response()
        if "WorldState" in content:
            return self._world_response()
        if "ClarifyingQuestion" in content:
            return self._questions_response()
        return '{"reasoning": "mock", "result": "fallback"}'

    def _questions_response(self) -> str:
        return (
            '{"questions": ['
            '{"question": "What genre?", "proposed_answer": "Fantasy"}, '
            '{"question": "Dark or light?", "proposed_answer": "Dark"}'
            "]}"
        )

    def _world_response(self) -> str:
        return (
            '{"reasoning": "mock", "world": {'
            '"setting": "A cursed kingdom", "genre": "fantasy", '
            '"tone": "dark", "locations": ["Castle"], '
            '"factions": ["Crown"], "key_facts": ["Magic costs blood"], '
            '"timeline": []}}'
        )

    def _characters_response(self) -> str:
        return (
            '{"reasoning": "mock", "characters": [{'
            '"name": "Kael", "role": "protagonist", '
            '"backstory": "Raised by wolves", '
            '"personality": "Brave", "goals": ["Survive"], '
            '"secrets": ["Knows the truth"], '
            '"relationships": {"Mira": "ally"}}, '
            '{"name": "Mira", "role": "ally", '
            '"backstory": "Former soldier", '
            '"personality": "Cautious", "goals": ["Protect Kael"], '
            '"secrets": [], "relationships": {"Kael": "ward"}}'
            "]}"
        )

    def _plot_response(self) -> str:
        return (
            '{"reasoning": "mock", '
            '"plot_outline": "Inciting: Castle escape. '
            'Rising: Forest journey. Climax: Confrontation."}'
        )

    def _scene_response(self) -> str:
        return (
            '{"reasoning": "mock", "scene": {'
            '"location": "Castle courtyard", '
            '"characters_present": ["Kael", "Mira"], '
            '"time_context": "Dusk", '
            '"situation": "Guards changing shift.", '
            '"gm_notes": "Setup escape."}}'
        )

    def _action_response(self) -> str:
        return (
            '{"reasoning": "mock", "action": {'
            '"character_name": "Kael", '
            '"action": "Moves toward the gate.", '
            '"dialogue": "Now.", '
            '"internal_thought": "No turning back."}}'
        )

    def _resolution_response(self) -> str:
        return (
            '{"reasoning": "mock", "resolution": {'
            '"outcome": "They escape through the gate.", '
            '"world_changes": ["Gate breached"], '
            '"character_updates": {}, '
            '"pacing": "end_story", '
            '"chapter_title": "The Escape"}}'
        )

    def _narrate_response(self) -> str:
        return (
            '{"reasoning": "mock", '
            '"prose": "The last light of dusk painted the courtyard '
            'in amber and shadow."}'
        )


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestSimulationEngineSetup:
    def test_setup_creates_valid_state(self) -> None:
        dspy.configure(lm=SimMockLM())
        engine = SimulationEngine()
        state = engine.setup(
            idea="A cursed kingdom",
            qa_context="Q: Genre?\nA: Fantasy",
            num_characters=2,
        )
        assert isinstance(state.world, WorldState)
        assert len(state.characters) >= 1
        assert state.plot_outline != ""
        assert state.current_round == 0
        assert state.current_chapter == 1


class TestSimulationEngineRound:
    def test_single_round(self) -> None:
        dspy.configure(lm=SimMockLM())
        engine = SimulationEngine()
        state = engine.setup(
            idea="A cursed kingdom",
            qa_context="Q: Genre?\nA: Fantasy",
            num_characters=2,
        )
        record = engine.run_round(state)
        assert record.round_number == 1
        assert record.chapter_number == 1
        assert len(record.actions) > 0
        assert record.narrator_prose != ""
        assert len(state.rounds) == 1


class TestSimulationEngineRun:
    def test_run_produces_story(self) -> None:
        dspy.configure(lm=SimMockLM())
        engine = SimulationEngine()
        state = engine.setup(
            idea="A cursed kingdom",
            qa_context="Q: Genre?\nA: Fantasy",
            num_characters=2,
        )
        story = engine.run(state)
        assert "###" in story
        assert len(state.chapters) >= 1

    def test_on_round_callback(self) -> None:
        dspy.configure(lm=SimMockLM())
        engine = SimulationEngine()
        state = engine.setup(
            idea="test",
            qa_context="",
            num_characters=2,
        )
        rounds_seen: list[int] = []
        story = engine.run(
            state,
            on_round=lambda r: rounds_seen.append(r.round_number),
        )
        assert len(rounds_seen) >= 1
        assert story != ""
