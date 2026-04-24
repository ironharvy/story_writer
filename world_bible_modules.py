"""DSPy modules for generating world-bible components."""

import re
from typing import List

import dspy

from _compat import observe
from story_modules import QuestionWithAnswer
from world_bible import WorldBible


_act_heading_re = re.compile(
    r"^\s*(?:#+\s*)?Act\s+(\d+)\b(?:\s*[-:–—]\s*(.*))?\s*$",
    re.IGNORECASE,
)


def _normalize_plot_timeline(plot_timeline: str) -> str:
    lines = (plot_timeline or "").splitlines()
    normalized_lines: list[str] = []
    last_emitted_act_number: str | None = None

    for raw_line in lines:
        line = raw_line.strip()
        match = _act_heading_re.match(line)
        if match:
            act_number = match.group(1)
            if act_number == last_emitted_act_number:
                continue
            normalized_lines.append(raw_line.rstrip())
            last_emitted_act_number = act_number
            continue

        normalized_lines.append(raw_line.rstrip())
        if line:
            last_emitted_act_number = None

    return "\n".join(normalized_lines).strip()


class GenerateWorldBibleQuestionsSignature(dspy.Signature):
    """Generate follow-up questions that enrich world-bible inputs."""

    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(
        desc=(
            "Up to 3 follow-up interrogative questions with proposed answers "
            "to help flesh out the World Bible."
        ),
    )


class WorldBibleQuestionGenerator(dspy.Module):
    """Generate world-bible clarification questions from premise and spine."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateWorldBibleQuestionsSignature)

    @observe()
    def forward(self, core_premise: str, spine_template: str):
        """Generate clarification questions and proposed answers."""
        return self.generate(core_premise=core_premise, spine_template=spine_template)


class GenerateWorldRulesSignature(dspy.Signature):
    """Generate rules governing the world and its systems."""

    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(
        desc="Additional answers or details provided by the user.",
    )
    world_rules: str = dspy.OutputField(
        desc="Rules of the world, including magic, science, etiquette, and law.",
    )


class GenerateCharactersSignature(dspy.Signature):
    """Generate character bios and relevant relationship details."""

    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(
        desc="Additional answers or details provided by the user.",
    )
    world_rules: str = dspy.InputField(desc="Rules of the world.")
    characters: str = dspy.OutputField(desc="Character descriptions and biographies.")


class GenerateLocationsSignature(dspy.Signature):
    """Generate key places and location context for the setting."""

    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(
        desc="Additional answers or details provided by the user.",
    )
    world_rules: str = dspy.InputField(desc="Rules of the world.")
    characters: str = dspy.InputField(desc="Character descriptions and biographies.")
    locations: str = dspy.OutputField(desc="Places and locations in the world.")


class GeneratePlotTimelineSignature(dspy.Signature):
    """A plot timeline."""

    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(
        desc="Additional answers or details provided by the user.",
    )
    world_rules: str = dspy.InputField(desc="Rules of the world.")
    characters: str = dspy.InputField(desc="Character descriptions and biographies.")
    locations: str = dspy.InputField(desc="Places and locations in the world.")
    plot_timeline: str = dspy.OutputField(desc="A plot timeline.")


class WorldBibleGenerator(dspy.Module):
    """Compose world rules, characters, locations, and timeline into one bible."""

    def __init__(self):
        super().__init__()
        self.generate_rules = dspy.ChainOfThought(GenerateWorldRulesSignature)
        self.generate_characters = dspy.ChainOfThought(GenerateCharactersSignature)
        self.generate_locations = dspy.ChainOfThought(GenerateLocationsSignature)
        self.generate_timeline = dspy.ChainOfThought(GeneratePlotTimelineSignature)

    @observe()
    def forward(self, core_premise: str, spine_template: str, user_additions: str = ""):
        """Generate all world-bible sections and return a consolidated prediction."""
        rules_result = self.generate_rules(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
        )

        characters_result = self.generate_characters(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
            world_rules=rules_result.world_rules,
        )

        locations_result = self.generate_locations(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
            world_rules=rules_result.world_rules,
            characters=characters_result.characters,
        )

        timeline_result = self.generate_timeline(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
            world_rules=rules_result.world_rules,
            characters=characters_result.characters,
            locations=locations_result.locations,
        )

        normalized_plot_timeline = _normalize_plot_timeline(
            timeline_result.plot_timeline,
        )
        world_bible = WorldBible(
            rules=rules_result.world_rules,
            characters=characters_result.characters,
            locations=locations_result.locations,
            plot_timeline=normalized_plot_timeline,
        )

        return dspy.Prediction(
            world_bible=world_bible.full_text,
            world_bible_structured=world_bible,
            world_rules=world_bible.rules,
            characters=world_bible.characters,
            locations=world_bible.locations,
            plot_timeline=world_bible.plot_timeline,
        )
