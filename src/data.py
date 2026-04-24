

from dataclasses import dataclass, field


@dataclass
class QuestionWithAnswer:
    question: str
    answer: str

@dataclass
class Character:
    name: str
    appearance: str
    role: str
    background: str

@dataclass
class Location:
    name: str
    description: str
    climate: str
    significance: str

@dataclass
class Chapter:
    title: str
    summary: str
    beats: list[str] = field(default_factory=list)
    random_detail: str | None = None
    enhancements: str | None = None
    prose: str | None = None

@dataclass
class Plan:
    act_1: list[Chapter] = field(default_factory=list)
    act_2: list[Chapter] = field(default_factory=list)
    act_3: list[Chapter] = field(default_factory=list)

@dataclass
class Story:
    idea: str
    core_premise: str | None = None
    spine: str | None = None
    characters: list[Character] = field(default_factory=list)
    rules: list[str] = field(default_factory=list)
    locations: list[Location] = field(default_factory=list)
    timeline: str | None = None
    plan: Plan | None = None
