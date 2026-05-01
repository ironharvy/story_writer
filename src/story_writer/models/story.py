from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class StoryLength(StrEnum):
    SHORT = "short"
    NOVELLA = "novella"
    NOVEL = "novel"


class Clarification(BaseModel):
    question: str
    suggested_answer: str
    answer: str = ""
    source: str = "model"


class Premise(BaseModel):
    protagonist: str
    want: str
    obstacle: str
    stakes: str
    genre: str = "unspecified"
    length: StoryLength = StoryLength.SHORT
    summary: str

    @field_validator("length", mode="before")
    @classmethod
    def normalize_length(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class Spine(BaseModel):
    once_upon_a_time: str
    every_day: str
    until_one_day: str
    because_of_that: str
    until_finally: str
    ever_since_then: str

    def beats(self) -> list[str]:
        return [
            self.once_upon_a_time,
            self.every_day,
            self.until_one_day,
            self.because_of_that,
            self.until_finally,
            self.ever_since_then,
        ]


class Character(BaseModel):
    name: str
    role: str
    traits: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


class Location(BaseModel):
    name: str
    description: str
    facts: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    order: int
    event: str


class WorldBible(BaseModel):
    logline: str
    characters: list[Character] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    continuity_constraints: list[str] = Field(default_factory=list)

    def known_names(self) -> set[str]:
        names = {character.name for character in self.characters}
        names.update(location.name for location in self.locations)
        return {name for name in names if name}


class ChapterBeat(BaseModel):
    order: int
    summary: str


class PlannedChapter(BaseModel):
    number: int
    title: str
    purpose: str
    spine_alignment: str
    beats: list[ChapterBeat] = Field(default_factory=list)

    @field_validator("number")
    @classmethod
    def number_is_positive(cls, value: int) -> int:
        if value < 1:
            msg = "chapter number must be positive"
            raise ValueError(msg)
        return value


class ChapterPlan(BaseModel):
    length: StoryLength
    chapters: list[PlannedChapter]

    @field_validator("length", mode="before")
    @classmethod
    def normalize_length(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class EnhancementNotes(BaseModel):
    chapter_number: int
    tension: str
    mystery: str
    theme_alignment: str
    setup_payoff: str
    emotional_curve: str
    easter_egg: str = ""


class EnhancementGuide(BaseModel):
    notes: list[EnhancementNotes]


class Embellishment(BaseModel):
    chapter_number: int
    triggered: bool
    detail: str = ""


class Chapter(BaseModel):
    number: int
    title: str
    beats: list[ChapterBeat] = Field(default_factory=list)
    enhancement_notes: EnhancementNotes | None = None
    embellishment: Embellishment | None = None
    prose: str

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_chapter(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        notes = normalized.get("enhancement_notes")
        if isinstance(notes, list):
            normalized["enhancement_notes"] = notes[0] if notes else None
        embellishment = normalized.get("embellishment")
        if isinstance(embellishment, str):
            normalized["embellishment"] = {
                "chapter_number": normalized.get("number", 0),
                "triggered": bool(embellishment.strip()),
                "detail": embellishment,
            }
        return normalized
