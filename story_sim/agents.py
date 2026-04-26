"""DSPy agent modules for GM, Character, and Narrator roles."""

import logging
from typing import List

import dspy

from story_sim._compat import observe
from story_sim.models import (
    CharacterAction,
    CharacterSheet,
    ClarifyingQuestion,
    RoundResolution,
    SceneContext,
    WorldState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GM Setup Signatures
# ---------------------------------------------------------------------------


class GenerateClarifyingQuestionsSignature(dspy.Signature):
    """Generate clarifying questions to understand the user's story idea.

    You are a Game Master preparing a TRPG-style story simulation.
    Ask questions about genre, tone, themes, scope, and character dynamics.
    """

    idea: str = dspy.InputField(desc="The user's initial story idea.")
    num_characters: int = dspy.InputField(
        desc="How many player characters the user wants.",
    )
    questions: List[ClarifyingQuestion] = dspy.OutputField(
        desc="3-5 clarifying questions about genre, tone, themes, and scope.",
    )


class GenerateWorldStateSignature(dspy.Signature):
    """Create the initial world state for a TRPG-style story simulation.

    Build a coherent world with locations, factions, and rules that
    support the story idea and the number of characters involved.
    """

    idea: str = dspy.InputField(desc="The user's story idea.")
    qa_context: str = dspy.InputField(
        desc="Answers to clarifying questions.",
    )
    num_characters: int = dspy.InputField(
        desc="Number of player characters to create.",
    )
    world: WorldState = dspy.OutputField(desc="The initial world state.")


class GenerateCharactersSignature(dspy.Signature):
    """Create character sheets for the story simulation.

    Each character must have distinct personality, goals, and at least one
    secret. Relationships between characters should create natural tension
    and potential for interesting interactions.
    """

    idea: str = dspy.InputField(desc="The user's story idea.")
    qa_context: str = dspy.InputField(
        desc="Answers to clarifying questions.",
    )
    world: str = dspy.InputField(desc="The world state description.")
    num_characters: int = dspy.InputField(
        desc="Number of characters to create.",
    )
    characters: List[CharacterSheet] = dspy.OutputField(
        desc="Character sheets with distinct personalities, goals, and relationships.",
    )


class GeneratePlotOutlineSignature(dspy.Signature):
    """Create a private plot compass for the GM to guide pacing.

    This outline is flexible — a compass, not a railroad. Include an
    inciting incident, key tension points, conditions that should trigger
    the climax, and a general resolution direction.
    """

    idea: str = dspy.InputField(desc="The user's story idea.")
    world: str = dspy.InputField(desc="The world state.")
    characters: str = dspy.InputField(desc="Summary of all characters.")
    plot_outline: str = dspy.OutputField(
        desc=(
            "A loose plot outline: inciting incident, rising action beats, "
            "climax trigger conditions, and resolution direction."
        ),
    )


# ---------------------------------------------------------------------------
# Simulation Round Signatures
# ---------------------------------------------------------------------------


class SetSceneSignature(dspy.Signature):
    """GM sets the scene for the next round of the simulation.

    Consider the plot outline and pacing hint to decide what scene would
    best advance the story. Include which characters are present and what
    tensions or opportunities exist.
    """

    world_state: str = dspy.InputField(desc="Current world state.")
    plot_outline: str = dspy.InputField(desc="GM's private plot compass.")
    recent_events: str = dspy.InputField(desc="Summary of recent rounds.")
    current_chapter: int = dspy.InputField(desc="Current chapter number.")
    pacing_hint: str = dspy.InputField(
        desc="Current story phase (setup, rising, climax, etc.).",
    )
    scene: SceneContext = dspy.OutputField(
        desc="The scene description for this round.",
    )


class CharacterActSignature(dspy.Signature):
    """A character decides their action in the current scene.

    Stay in character. Act based on your personality, goals, and what you
    know. You may not know everything that's happening — act on your own
    knowledge and perspective only.
    """

    character_sheet: str = dspy.InputField(
        desc="This character's full sheet.",
    )
    scene_description: str = dspy.InputField(
        desc="The current scene as described by the GM.",
    )
    personal_memory: str = dspy.InputField(
        desc="This character's memory of recent events they witnessed.",
    )
    other_characters_actions: str = dspy.InputField(
        desc=(
            "Actions already taken by other characters this round "
            "(empty if acting first)."
        ),
    )
    action: CharacterAction = dspy.OutputField(
        desc="The character's action, dialogue, and thoughts.",
    )


class ResolveRoundSignature(dspy.Signature):
    """GM resolves all character actions and determines outcomes.

    Consider the world rules, character capabilities, and plot compass.
    Determine what actually happens, any world state changes, and whether
    the story should continue the current scene, start a new scene or
    chapter, or end the story.
    """

    world_state: str = dspy.InputField(desc="Current world state.")
    plot_outline: str = dspy.InputField(desc="GM's private plot compass.")
    scene: str = dspy.InputField(desc="The current scene description.")
    character_actions: str = dspy.InputField(
        desc="All character actions this round.",
    )
    story_progress: str = dspy.InputField(
        desc="Brief summary of story progress (chapter count, key events).",
    )
    resolution: RoundResolution = dspy.OutputField(
        desc="Outcome of the round, world changes, and pacing decision.",
    )


class NarrateRoundSignature(dspy.Signature):
    """Render a simulation round into polished story prose.

    Write rich, immersive narrative with natural dialogue, character
    interiority, and vivid description. Match the genre and tone.
    If this is a chapter start, set the scene. If a chapter end,
    provide closure or a hook.
    """

    genre: str = dspy.InputField(desc="Story genre for tone matching.")
    tone: str = dspy.InputField(desc="Desired narrative tone.")
    scene_description: str = dspy.InputField(
        desc="The scene as set by the GM.",
    )
    character_actions: str = dspy.InputField(
        desc="What each character did and said.",
    )
    resolution: str = dspy.InputField(
        desc="How the GM resolved the round.",
    )
    is_chapter_start: bool = dspy.InputField(
        desc="Whether this is the first round of a new chapter.",
    )
    is_chapter_end: bool = dspy.InputField(
        desc="Whether this is the last round of the current chapter.",
    )
    prose: str = dspy.OutputField(
        desc=(
            "Polished narrative prose for this round. Rich description, "
            "natural dialogue, character interiority."
        ),
    )


# ---------------------------------------------------------------------------
# Agent Modules
# ---------------------------------------------------------------------------


class GameMasterSetup(dspy.Module):
    """GM agent for the setup phase: questions, world, characters, plot."""

    def __init__(self) -> None:
        super().__init__()
        self.ask_questions = dspy.Predict(GenerateClarifyingQuestionsSignature)
        self.create_world = dspy.ChainOfThought(GenerateWorldStateSignature)
        self.create_characters = dspy.ChainOfThought(
            GenerateCharactersSignature,
        )
        self.create_plot = dspy.ChainOfThought(GeneratePlotOutlineSignature)

    @observe()
    def generate_questions(
        self,
        idea: str,
        num_characters: int,
    ) -> dspy.Prediction:
        """Ask clarifying questions about the user's story idea."""
        return self.ask_questions(idea=idea, num_characters=num_characters)

    @observe()
    def generate_world(
        self,
        idea: str,
        qa_context: str,
        num_characters: int,
    ) -> dspy.Prediction:
        """Create the initial world state."""
        return self.create_world(
            idea=idea,
            qa_context=qa_context,
            num_characters=num_characters,
        )

    @observe()
    def generate_characters(
        self,
        idea: str,
        qa_context: str,
        world: str,
        num_characters: int,
    ) -> dspy.Prediction:
        """Create character sheets."""
        return self.create_characters(
            idea=idea,
            qa_context=qa_context,
            world=world,
            num_characters=num_characters,
        )

    @observe()
    def generate_plot_outline(
        self,
        idea: str,
        world: str,
        characters: str,
    ) -> dspy.Prediction:
        """Create the GM's private plot compass."""
        return self.create_plot(idea=idea, world=world, characters=characters)


class GameMasterRound(dspy.Module):
    """GM agent for running a simulation round."""

    def __init__(self) -> None:
        super().__init__()
        self.set_scene = dspy.ChainOfThought(SetSceneSignature)
        self.resolve = dspy.ChainOfThought(ResolveRoundSignature)

    @observe()
    def create_scene(
        self,
        world_state: str,
        plot_outline: str,
        recent_events: str,
        current_chapter: int,
        pacing_hint: str,
    ) -> dspy.Prediction:
        """Set the scene for the next round."""
        return self.set_scene(
            world_state=world_state,
            plot_outline=plot_outline,
            recent_events=recent_events,
            current_chapter=current_chapter,
            pacing_hint=pacing_hint,
        )

    @observe()
    def resolve_round(
        self,
        world_state: str,
        plot_outline: str,
        scene: str,
        character_actions: str,
        story_progress: str,
    ) -> dspy.Prediction:
        """Resolve a round's character actions."""
        return self.resolve(
            world_state=world_state,
            plot_outline=plot_outline,
            scene=scene,
            character_actions=character_actions,
            story_progress=story_progress,
        )


class CharacterAgent(dspy.Module):
    """A character agent that decides actions in-character."""

    def __init__(self) -> None:
        super().__init__()
        self.act = dspy.ChainOfThought(CharacterActSignature)

    @observe()
    def take_action(
        self,
        character_sheet: str,
        scene_description: str,
        personal_memory: str,
        other_characters_actions: str,
    ) -> dspy.Prediction:
        """Decide an in-character action for the current scene."""
        return self.act(
            character_sheet=character_sheet,
            scene_description=scene_description,
            personal_memory=personal_memory,
            other_characters_actions=other_characters_actions,
        )


class NarratorAgent(dspy.Module):
    """Narrator agent that renders rounds into polished prose."""

    def __init__(self) -> None:
        super().__init__()
        self.narrate = dspy.ChainOfThought(NarrateRoundSignature)

    @observe()
    def render_round(
        self,
        genre: str,
        tone: str,
        scene_description: str,
        character_actions: str,
        resolution: str,
        *,
        is_chapter_start: bool,
        is_chapter_end: bool,
    ) -> dspy.Prediction:
        """Render a round into polished narrative prose."""
        return self.narrate(
            genre=genre,
            tone=tone,
            scene_description=scene_description,
            character_actions=character_actions,
            resolution=resolution,
            is_chapter_start=is_chapter_start,
            is_chapter_end=is_chapter_end,
        )
