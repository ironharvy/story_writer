"""Data contracts for the headless story generation pipeline.

These types replace the ad-hoc locals scattered across ``main.py`` with one
serializable state object that can live in a DB row / Redis job payload.

Design notes:
- ``StoryState`` accumulates outputs across stages. Each stage reads what it
  needs and writes its own slice.
- ``QAPair`` mirrors ``story_modules.QuestionWithAnswer`` but adds the user's
  accepted answer (or ``None`` meaning "accept the proposed answer").
- ``PipelineOptions`` captures the per-run toggles that used to live on the
  argparse ``Namespace``.
- ``StageName`` is the stable identifier used for persistence and routing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class StageName(str, Enum):
    """Canonical stage identifiers.

    These are persisted (in DB rows, job queues, logs) so the string values
    are part of the public contract — do not rename without a migration.
    """

    IDEATE = "ideate"
    PREMISE = "premise"
    SPINE = "spine"
    WORLD_BIBLE_QUESTIONS = "world_bible_questions"
    WORLD_BIBLE = "world_bible"
    CHARACTER_VISUALS = "character_visuals"
    CHARACTER_PORTRAITS = "character_portraits"
    STORY = "story"
    INPAINT = "inpaint"
    SCENE_IMAGES = "scene_images"
    SIMILARITY_CHECK = "similarity_check"


@dataclass
class QAPair:
    """One interrogative question + a proposed answer + the user's choice.

    ``user_answer`` is the authoritative answer. When ``None``, the caller
    should treat ``proposed_answer`` as accepted.
    """

    question: str
    proposed_answer: str
    user_answer: Optional[str] = None

    @property
    def effective_answer(self) -> str:
        """The answer to feed downstream stages."""
        return self.user_answer if self.user_answer is not None else self.proposed_answer

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAPair":
        return cls(
            question=data["question"],
            proposed_answer=data["proposed_answer"],
            user_answer=data.get("user_answer"),
        )


@dataclass
class CharacterVisualState:
    """Serializable mirror of ``story_modules.CharacterVisual`` for persistence.

    We avoid depending on the Pydantic model directly so ``StoryState`` stays
    free of DSPy / Pydantic imports and remains trivially JSON-roundtrippable.
    """

    name: str
    reference_mix: str
    distinguishing_features: str
    full_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharacterVisualState":
        return cls(
            name=data["name"],
            reference_mix=data.get("reference_mix", ""),
            distinguishing_features=data.get("distinguishing_features", ""),
            full_prompt=data.get("full_prompt", ""),
        )


@dataclass
class StoryState:
    """Accumulating pipeline state.

    Every stage reads what it needs, writes its slice, and returns the mutated
    state. The whole struct is JSON-serializable via :meth:`to_dict`.
    """

    idea: str = ""
    ideation_qa: list[QAPair] = field(default_factory=list)
    core_premise: str = ""
    spine_template: str = ""
    world_bible_qa: list[QAPair] = field(default_factory=list)
    world_bible: str = ""

    # Image pipeline (optional)
    character_visuals: list[CharacterVisualState] = field(default_factory=list)
    character_portrait_paths: dict[str, str] = field(default_factory=dict)

    # Story pipeline outputs
    arc_outline: str = ""
    chapter_plan: str = ""
    enhancers_guide: str = ""
    story_text: str = ""
    # ``final_story_text`` mirrors ``story_text`` unless inpainting ran.
    final_story_text: str = ""

    # Scene illustrations keyed by 1-based chapter index.
    scene_image_paths: dict[int, str] = field(default_factory=dict)

    # Post-processing
    similarity_report: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation of the state."""
        return {
            "idea": self.idea,
            "ideation_qa": [qa.to_dict() for qa in self.ideation_qa],
            "core_premise": self.core_premise,
            "spine_template": self.spine_template,
            "world_bible_qa": [qa.to_dict() for qa in self.world_bible_qa],
            "world_bible": self.world_bible,
            "character_visuals": [cv.to_dict() for cv in self.character_visuals],
            "character_portrait_paths": dict(self.character_portrait_paths),
            "arc_outline": self.arc_outline,
            "chapter_plan": self.chapter_plan,
            "enhancers_guide": self.enhancers_guide,
            "story_text": self.story_text,
            "final_story_text": self.final_story_text,
            # Serialize int keys as strings for JSON compatibility.
            "scene_image_paths": {str(k): v for k, v in self.scene_image_paths.items()},
            "similarity_report": self.similarity_report,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoryState":
        scene_images_raw = data.get("scene_image_paths", {}) or {}
        scene_images: dict[int, str] = {}
        for key, value in scene_images_raw.items():
            try:
                scene_images[int(key)] = value
            except (TypeError, ValueError):
                # Tolerate non-integer keys by skipping; data that round-trips
                # through ``to_dict`` will always be coercible.
                continue

        return cls(
            idea=data.get("idea", ""),
            ideation_qa=[QAPair.from_dict(x) for x in data.get("ideation_qa", [])],
            core_premise=data.get("core_premise", ""),
            spine_template=data.get("spine_template", ""),
            world_bible_qa=[QAPair.from_dict(x) for x in data.get("world_bible_qa", [])],
            world_bible=data.get("world_bible", ""),
            character_visuals=[
                CharacterVisualState.from_dict(x)
                for x in data.get("character_visuals", [])
            ],
            character_portrait_paths=dict(data.get("character_portrait_paths", {})),
            arc_outline=data.get("arc_outline", ""),
            chapter_plan=data.get("chapter_plan", ""),
            enhancers_guide=data.get("enhancers_guide", ""),
            story_text=data.get("story_text", ""),
            final_story_text=data.get("final_story_text", ""),
            scene_image_paths=scene_images,
            similarity_report=data.get("similarity_report"),
        )


@dataclass
class PipelineOptions:
    """Per-run toggles. Mirrors the subset of CLI flags that drives pipeline
    behavior (not cosmetic concerns like logging verbosity)."""

    enable_images: bool = False
    replicate_api_token: Optional[str] = None

    inpaint_chapters: bool = False
    inpaint_ratio: float = 1.35

    check_similar: bool = True
    similar_threshold: float = 0.65

    use_optimized: bool = False
    optimized_manifest: Optional[str] = None

    def validate(self) -> None:
        """Raise ``ValueError`` for invalid combinations.

        Callers should run this before kicking off a pipeline so that the
        failure surfaces at job-submission time rather than mid-generation.
        """
        if self.inpaint_chapters and self.inpaint_ratio <= 1.0:
            raise ValueError(
                f"inpaint_ratio must be > 1.0 when inpaint_chapters=True, "
                f"got {self.inpaint_ratio}"
            )
        if self.enable_images and not self.replicate_api_token:
            raise ValueError(
                "enable_images=True requires replicate_api_token to be set."
            )
        if not (0.0 <= self.similar_threshold <= 1.0):
            raise ValueError(
                f"similar_threshold must be in [0.0, 1.0], got {self.similar_threshold}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineOptions":
        return cls(**data)
