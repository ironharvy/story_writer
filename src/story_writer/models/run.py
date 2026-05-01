from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class StageRecord(BaseModel):
    status: str = "pending"
    model: str = ""
    provider: str = "ollama"
    temperature: float | None = None
    max_tokens: int | None = None
    context_window: int | None = None
    estimated_input_tokens: int | None = None
    context_warning: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str = ""

    @field_validator("started_at", "finished_at", mode="before")
    @classmethod
    def blank_datetime_is_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


class Manifest(BaseModel):
    schema_version: int = 1
    slug: str
    created_at: datetime = Field(default_factory=utc_now)
    length: str = "short"
    embellish_probability: float = 0.15
    allow_paid: bool = False
    stages: dict[str, StageRecord] = Field(default_factory=dict)
