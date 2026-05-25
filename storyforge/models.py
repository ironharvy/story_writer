"""Typed artifacts for the generation pipeline + resumable run state.

Everything is a dataclass with dict (de)serialization so each artifact can be
written to disk the moment it is produced and a run can resume after a crash.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StorySpec:
    genre: str = "literary"
    tone: str = "evocative"
    pov: str = "third_person_limited"  # first_person | third_person_limited | third_person_omniscient
    pov_character: str = ""             # name of the POV/protagonist character
    ending: str = "earned and bittersweet"
    audience: str = "adult"
    num_chapters: int = 7
    words_per_chapter: int = 1200

    @property
    def pov_label(self) -> str:
        return {"first_person": "first person",
                "third_person_limited": "third person limited",
                "third_person_omniscient": "third person omniscient"}.get(self.pov, self.pov)


@dataclass
class Character:
    name: str
    role: str = ""          # protagonist | antagonist | ally | mentor | ...
    description: str = ""
    want: str = ""          # external goal
    need: str = ""          # internal lesson
    arc: str = ""


@dataclass
class Location:
    name: str
    description: str = ""


@dataclass
class Premise:
    title: str = ""
    logline: str = ""
    paragraph: str = ""     # one-paragraph three-act summary
    theme: str = ""


@dataclass
class WorldBible:
    characters: list[Character] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)

    def names(self) -> list[str]:
        return [c.name for c in self.characters]


@dataclass
class ChapterPlan:
    number: int
    title: str = ""
    pov_character: str = ""
    summary: str = ""
    beats: list[str] = field(default_factory=list)
    goal: str = ""           # the POV character's scene goal
    conflict: str = ""       # what opposes it
    outcome: str = ""        # how it shifts the story (and the want/need)
    characters_present: list[str] = field(default_factory=list)
    word_target: int = 1200
    # setup | rising | midpoint | lowest_point | climax | resolution
    structural_role: str = ""


@dataclass
class Chapter:
    number: int
    title: str
    prose: str
    summary: str = ""


@dataclass
class RunState:
    idea: str
    spec: StorySpec = field(default_factory=StorySpec)
    premise: Premise | None = None
    bible: WorldBible | None = None
    spine: list[ChapterPlan] = field(default_factory=list)
    chapters: list[Chapter] = field(default_factory=list)
    status: str = "init"
    draft_model: str = ""

    # --- serialization ---
    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "RunState":
        spec = StorySpec(**d.get("spec", {}))
        premise = Premise(**d["premise"]) if d.get("premise") else None
        bible = None
        if d.get("bible"):
            b = d["bible"]
            bible = WorldBible(
                characters=[Character(**c) for c in b.get("characters", [])],
                locations=[Location(**l) for l in b.get("locations", [])],
                rules=list(b.get("rules", [])),
            )
        spine = [ChapterPlan(**p) for p in d.get("spine", [])]
        chapters = [Chapter(**c) for c in d.get("chapters", [])]
        return RunState(
            idea=d["idea"], spec=spec, premise=premise, bible=bible, spine=spine,
            chapters=chapters, status=d.get("status", "init"),
            draft_model=d.get("draft_model", ""),
        )

    def drafted_numbers(self) -> set[int]:
        return {c.number for c in self.chapters}

    def synopsis(self) -> str:
        """Rolling synopsis of drafted chapters for continuity context."""
        return "\n".join(f"Ch{c.number} ({c.title}): {c.summary}" for c in self.chapters
                         if c.summary)
