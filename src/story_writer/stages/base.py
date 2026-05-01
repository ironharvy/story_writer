from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from story_writer.config import ProviderConfig
from story_writer.models.story import StoryLength
from story_writer.providers import stage_lm

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass
class StageContext:
    slug: str
    idea: str
    length: StoryLength
    embellish_probability: float
    allow_paid: bool
    artifacts: dict[str, Any] = field(default_factory=dict)
    feedback: str = ""


class StageBase(ABC):
    name: str
    output_model: type[BaseModel]

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def run(self, context: StageContext) -> BaseModel:
        with stage_lm(self.config, allow_paid=context.allow_paid):
            return self._execute(context)

    @abstractmethod
    def _execute(self, context: StageContext) -> BaseModel:
        raise NotImplementedError


def parse_json_model(raw: Any, model: type[ModelT]) -> ModelT:
    if isinstance(raw, model):
        return raw
    if isinstance(raw, dict):
        return model.model_validate(raw)
    text = str(raw).strip()
    try:
        return model.model_validate_json(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return model.model_validate_json(match.group(0))


def parse_json_list(raw: Any, item_model: type[ModelT]) -> list[ModelT]:
    if isinstance(raw, list):
        return [
            item if isinstance(item, item_model) else item_model.model_validate(item)
            for item in raw
        ]
    text = str(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    return [
        item if isinstance(item, item_model) else item_model.model_validate(item)
        for item in data
    ]


def artifact_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, list):
        return json.dumps(
            [
                item.model_dump() if isinstance(item, BaseModel) else item
                for item in value
            ],
            indent=2,
        )
    return json.dumps(value, indent=2, default=str)
