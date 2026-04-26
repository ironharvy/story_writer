"""Simulation engine — coordinates GM, Character, and Narrator agents."""

import logging

from story_sim.agents import (
    CharacterAgent,
    GameMasterRound,
    GameMasterSetup,
    NarratorAgent,
)
from story_sim.models import (
    CharacterAction,
    CharacterSheet,
    PacingDecision,
    RoundRecord,
    SimulationState,
    WorldState,
)

logger = logging.getLogger(__name__)

MAX_ROUNDS = 50
MAX_ROUNDS_PER_CHAPTER = 12

_RECOVERABLE_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    KeyError,
    IndexError,
)


def format_character_sheet(char: CharacterSheet) -> str:
    """Serialize a character sheet into a prompt-friendly string."""
    lines = [
        f"Name: {char.name}",
        f"Role: {char.role}",
        f"Backstory: {char.backstory}",
        f"Personality: {char.personality}",
        f"Goals: {', '.join(char.goals)}",
    ]
    if char.secrets:
        lines.append(f"Secrets: {', '.join(char.secrets)}")
    if char.relationships:
        rels = "; ".join(f"{k}: {v}" for k, v in char.relationships.items())
        lines.append(f"Relationships: {rels}")
    return "\n".join(lines)


def format_characters_summary(characters: list[CharacterSheet]) -> str:
    """Serialize all character sheets for prompt use."""
    return "\n\n".join(format_character_sheet(c) for c in characters)


def get_recent_events_summary(
    rounds: list[RoundRecord],
    limit: int = 5,
) -> str:
    """Summarize recent rounds for the GM's context."""
    if not rounds:
        return "Story has just begun. No prior events."
    recent = rounds[-limit:]
    summaries = []
    for r in recent:
        actions = "; ".join(f"{a.character_name}: {a.action}" for a in r.actions)
        summaries.append(
            f"Round {r.round_number} (Ch.{r.chapter_number}): "
            f"{r.scene.situation} | Actions: {actions} | "
            f"Outcome: {r.resolution.outcome}",
        )
    return "\n".join(summaries)


def get_personal_memory(
    character_name: str,
    rounds: list[RoundRecord],
    limit: int = 5,
) -> str:
    """Get events from this character's perspective only."""
    relevant = [r for r in rounds if character_name in r.scene.characters_present]
    if not relevant:
        return "No prior events witnessed."
    recent = relevant[-limit:]
    memories = []
    for r in recent:
        own_action = next(
            (a for a in r.actions if a.character_name == character_name),
            None,
        )
        others = [a for a in r.actions if a.character_name != character_name]
        others_text = "; ".join(f"{a.character_name} {a.action}" for a in others)
        own_text = f"You {own_action.action}" if own_action else "You observed"
        memories.append(
            f"Round {r.round_number}: {r.scene.situation}. "
            f"{own_text}. Others: {others_text}. "
            f"Result: {r.resolution.outcome}",
        )
    return "\n".join(memories)


def format_actions(actions: list[CharacterAction]) -> str:
    """Format character actions for prompt injection."""
    parts = []
    for a in actions:
        text = f"{a.character_name}: {a.action}"
        if a.dialogue:
            text += f' Says: "{a.dialogue}"'
        parts.append(text)
    return "\n".join(parts)


def get_story_progress(state: SimulationState) -> str:
    """Summarize story progress for the GM's pacing decisions."""
    return (
        f"Chapter {state.current_chapter}, Round {state.current_round}. "
        f"Chapters completed: {len(state.chapters)}. "
        f"Total rounds so far: {len(state.rounds)}."
    )


def get_pacing_hint(state: SimulationState) -> str:
    """Suggest a pacing phase based on how far the story has progressed."""
    total_rounds = len(state.rounds)
    if total_rounds < 3:
        return "setup — establish characters and initial situation"
    if total_rounds < 10:
        return "rising action — develop conflicts and deepen stakes"
    if total_rounds < 20:
        return "mid-story — complications, twists, character development"
    if total_rounds < 30:
        return "approaching climax — tensions should peak soon"
    return "late story — drive toward resolution"


def finalize_chapter(state: SimulationState, title: str) -> None:
    """Collect narrator prose from current chapter rounds into a chapter."""
    chapter_rounds = [
        r for r in state.rounds if r.chapter_number == state.current_chapter
    ]
    prose = "\n\n".join(r.narrator_prose for r in chapter_rounds if r.narrator_prose)
    chapter_title = title or f"Chapter {state.current_chapter}"
    state.chapters.append((chapter_title, prose))
    logger.info(
        "Finalized chapter %d: %s (%d rounds)",
        state.current_chapter,
        chapter_title,
        len(chapter_rounds),
    )
    state.current_chapter += 1


def compile_story(state: SimulationState) -> str:
    """Compile all chapters into final story text."""
    parts = []
    for i, (title, prose) in enumerate(state.chapters, start=1):
        clean_title = title if title else f"Chapter {i}"
        parts.append(f"### {clean_title}\n\n{prose}")
    return "\n\n".join(parts)


class SimulationEngine:
    """Orchestrates the multi-agent story simulation loop."""

    def __init__(self) -> None:
        self.gm_setup = GameMasterSetup()
        self.gm_round = GameMasterRound()
        self.character_agent = CharacterAgent()
        self.narrator = NarratorAgent()

    def setup(
        self,
        idea: str,
        qa_context: str,
        num_characters: int,
    ) -> SimulationState:
        """Run the GM setup phase and return initial simulation state."""
        world_result = self.gm_setup.generate_world(
            idea=idea,
            qa_context=qa_context,
            num_characters=num_characters,
        )
        world: WorldState = world_result.world

        chars_result = self.gm_setup.generate_characters(
            idea=idea,
            qa_context=qa_context,
            world=world.model_dump_json(),
            num_characters=num_characters,
        )
        characters: list[CharacterSheet] = chars_result.characters

        plot_result = self.gm_setup.generate_plot_outline(
            idea=idea,
            world=world.model_dump_json(),
            characters=format_characters_summary(characters),
        )

        return SimulationState(
            world=world,
            characters=characters,
            plot_outline=plot_result.plot_outline,
        )

    def _collect_character_actions(
        self,
        state: SimulationState,
        scene_situation: str,
        characters_present: list[str],
    ) -> list[CharacterAction]:
        """Have each present character take their turn."""
        actions: list[CharacterAction] = []
        for char in state.characters:
            if char.name not in characters_present:
                continue
            try:
                action_result = self.character_agent.take_action(
                    character_sheet=format_character_sheet(char),
                    scene_description=scene_situation,
                    personal_memory=get_personal_memory(
                        char.name,
                        state.rounds,
                    ),
                    other_characters_actions=format_actions(actions),
                )
                action = action_result.action
                action.character_name = char.name
                actions.append(action)
            except _RECOVERABLE_EXCEPTIONS as exc:
                logger.warning(
                    "Character %s failed to act: %s",
                    char.name,
                    exc,
                )
        return actions

    def run_round(self, state: SimulationState) -> RoundRecord:
        """Execute one simulation round: scene -> actions -> resolve -> narrate."""
        state.current_round += 1
        is_chapter_start = not any(
            r.chapter_number == state.current_chapter for r in state.rounds
        )

        scene_result = self.gm_round.create_scene(
            world_state=state.world.model_dump_json(),
            plot_outline=state.plot_outline,
            recent_events=get_recent_events_summary(state.rounds),
            current_chapter=state.current_chapter,
            pacing_hint=get_pacing_hint(state),
        )
        scene = scene_result.scene

        actions = self._collect_character_actions(
            state,
            scene.situation,
            scene.characters_present,
        )

        resolve_result = self.gm_round.resolve_round(
            world_state=state.world.model_dump_json(),
            plot_outline=state.plot_outline,
            scene=scene.situation,
            character_actions=format_actions(actions),
            story_progress=get_story_progress(state),
        )
        resolution = resolve_result.resolution

        is_chapter_end = resolution.pacing in (
            PacingDecision.NEW_CHAPTER,
            PacingDecision.END_STORY,
        )

        narrate_result = self.narrator.render_round(
            genre=state.world.genre,
            tone=state.world.tone,
            scene_description=scene.situation,
            character_actions=format_actions(actions),
            resolution=resolution.outcome,
            is_chapter_start=is_chapter_start,
            is_chapter_end=is_chapter_end,
        )

        record = RoundRecord(
            round_number=state.current_round,
            chapter_number=state.current_chapter,
            scene=scene,
            actions=actions,
            resolution=resolution,
            narrator_prose=narrate_result.prose,
        )
        state.rounds.append(record)

        for change in resolution.world_changes:
            state.world.timeline.append(change)

        if is_chapter_end:
            finalize_chapter(state, resolution.chapter_title)

        return record

    def run(
        self,
        state: SimulationState,
        on_round: "None | callable" = None,
    ) -> str:
        """Run the full simulation until the GM ends the story."""
        rounds_in_chapter = 0

        for _ in range(MAX_ROUNDS):
            record = self.run_round(state)
            rounds_in_chapter += 1

            if on_round is not None:
                on_round(record)

            logger.info(
                "Round %d (Ch.%d): %s -> %s",
                record.round_number,
                record.chapter_number,
                record.scene.situation[:80],
                record.resolution.pacing.value,
            )

            if record.resolution.pacing == PacingDecision.END_STORY:
                break
            if record.resolution.pacing in (
                PacingDecision.NEW_CHAPTER,
                PacingDecision.CLIMAX,
            ):
                rounds_in_chapter = 0

            if rounds_in_chapter >= MAX_ROUNDS_PER_CHAPTER:
                logger.warning(
                    "Forcing chapter break after %d rounds",
                    rounds_in_chapter,
                )
                finalize_chapter(state, "")
                rounds_in_chapter = 0

        self._finalize_remaining(state)
        return compile_story(state)

    def _finalize_remaining(self, state: SimulationState) -> None:
        """Finalize any chapter that wasn't closed by the GM."""
        unfinalized = [
            r for r in state.rounds if r.chapter_number == state.current_chapter
        ]
        finalized_nums = set(range(1, len(state.chapters) + 1))
        if unfinalized and state.current_chapter not in finalized_nums:
            finalize_chapter(state, "")
