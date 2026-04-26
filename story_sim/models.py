"""Data models for multi-agent story simulation."""

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field


class PacingDecision(str, Enum):
    """GM's decision about what happens next in the story structure."""

    CONTINUE_SCENE = "continue_scene"
    NEW_SCENE = "new_scene"
    NEW_CHAPTER = "new_chapter"
    CLIMAX = "climax"
    END_STORY = "end_story"


class CharacterSheet(BaseModel):
    """A character's identity and traits, known to the character agent."""

    name: str = Field(description="Character's full name.")
    role: str = Field(
        description="Role in the story (protagonist, antagonist, ally, etc.).",
    )
    backstory: str = Field(description="Character's background and history.")
    personality: str = Field(
        description="Core personality traits and temperament.",
    )
    goals: list[str] = Field(
        description="What the character wants to achieve.",
    )
    secrets: list[str] = Field(
        default_factory=list,
        description="Things only this character (and GM) knows.",
    )
    relationships: dict[str, str] = Field(
        default_factory=dict,
        description="Relationships to other characters (name -> description).",
    )


class WorldState(BaseModel):
    """Compact world representation maintained by the GM."""

    setting: str = Field(description="The world's setting description.")
    genre: str = Field(description="Story genre.")
    tone: str = Field(
        description="Narrative tone (dark, lighthearted, gritty, etc.).",
    )
    locations: list[str] = Field(
        default_factory=list,
        description="Known locations in the world.",
    )
    factions: list[str] = Field(
        default_factory=list,
        description="Major groups, organizations, or factions.",
    )
    key_facts: list[str] = Field(
        default_factory=list,
        description="Important world facts and rules.",
    )
    timeline: list[str] = Field(
        default_factory=list,
        description="Major events that have occurred, in order.",
    )


class SceneContext(BaseModel):
    """Current scene information provided by the GM each round."""

    location: str = Field(description="Where the scene takes place.")
    characters_present: list[str] = Field(
        description="Which characters are in the scene.",
    )
    time_context: str = Field(
        description="Time of day, how much time has passed, etc.",
    )
    situation: str = Field(
        description="What's happening — tensions, events, atmosphere.",
    )
    gm_notes: str = Field(
        default="",
        description="Private GM notes about this scene's purpose in the plot.",
    )


class CharacterAction(BaseModel):
    """A character's response during a round."""

    character_name: str = Field(description="Who is acting.")
    action: str = Field(
        description="What the character does (physical actions, movement).",
    )
    dialogue: str = Field(
        default="",
        description="What the character says, if anything.",
    )
    internal_thought: str = Field(
        default="",
        description="What the character is thinking (for narrator's use).",
    )


class RoundResolution(BaseModel):
    """GM's resolution of a round after all characters act."""

    outcome: str = Field(
        description="What actually happens as a result of character actions.",
    )
    world_changes: list[str] = Field(
        default_factory=list,
        description="Updates to world state.",
    )
    character_updates: dict[str, str] = Field(
        default_factory=dict,
        description="Per-character consequences or state changes.",
    )
    pacing: PacingDecision = Field(
        description="What happens next in terms of story structure.",
    )
    chapter_title: str = Field(
        default="",
        description="If pacing is NEW_CHAPTER or END_STORY, the chapter title.",
    )


class RoundRecord(BaseModel):
    """Complete record of one simulation round."""

    round_number: int
    chapter_number: int
    scene: SceneContext
    actions: list[CharacterAction]
    resolution: RoundResolution
    narrator_prose: str = Field(
        default="",
        description="The narrator's rendered prose for this round.",
    )


class ClarifyingQuestion(BaseModel):
    """A GM clarifying question paired with a suggested answer."""

    question: str = Field(description="A question to clarify the story idea.")
    proposed_answer: str = Field(description="A suggested default answer.")


@dataclass
class SimulationConfig:
    """User-provided configuration for the simulation."""

    idea: str
    num_characters: int = 3
    qa_context: str = ""


@dataclass
class SimulationState:
    """Mutable state tracking the full simulation."""

    world: WorldState
    characters: list[CharacterSheet]
    plot_outline: str
    rounds: list[RoundRecord] = field(default_factory=list)
    current_chapter: int = 1
    current_round: int = 0
    chapters: list[tuple[str, str]] = field(default_factory=list)
