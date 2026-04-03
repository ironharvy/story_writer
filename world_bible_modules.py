import dspy
from typing import List
from story_modules import QuestionWithAnswer
try:
    from langfuse import observe
except ImportError:
    def observe():
        def decorator(fn):
            return fn
        return decorator

class GenerateWorldBibleQuestionsSignature(dspy.Signature):
    """Generates a few follow-up questions to ask the user to help flesh out the world bible before generating the final version."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(desc="Up to 3 follow-up interrogative questions with proposed answers to help flesh out the World Bible.")

class WorldBibleQuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateWorldBibleQuestionsSignature)

    @observe()
    def forward(self, core_premise: str, spine_template: str):
        return self.generate(core_premise=core_premise, spine_template=spine_template)

class GenerateWorldRulesSignature(dspy.Signature):
    """Rules of the world. Magic or a highly specialized science? What about highly involved rules of etiquette or law? If so, how does one use said magic or science? What are the intricacies of the rules or laws, including any loopholes your characters might avail themselves of?"""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(desc="Additional answers or details provided by the user.")
    world_rules: str = dspy.OutputField(desc="Rules of the world, including magic, science, etiquette, and law.")

class GenerateCharactersSignature(dspy.Signature):
    """Character descriptions and biographies. List of characters’ full names and any facts about them that are relevant (e.g., physical description, relationships to other characters, job titles, aspirations). Note that entries for main characters are likely to be longer than for minor characters."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(desc="Additional answers or details provided by the user.")
    world_rules: str = dspy.InputField(desc="Rules of the world.")
    characters: str = dspy.OutputField(desc="Character descriptions and biographies.")

class GenerateLocationsSignature(dspy.Signature):
    """Places and locations in the world. What is their usual climate? Who lives there generally? Where are they in relation to one another?"""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(desc="Additional answers or details provided by the user.")
    world_rules: str = dspy.InputField(desc="Rules of the world.")
    characters: str = dspy.InputField(desc="Character descriptions and biographies.")
    locations: str = dspy.OutputField(desc="Places and locations in the world.")

class GeneratePlotTimelineSignature(dspy.Signature):
    """A plot timeline."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(desc="Additional answers or details provided by the user.")
    world_rules: str = dspy.InputField(desc="Rules of the world.")
    characters: str = dspy.InputField(desc="Character descriptions and biographies.")
    locations: str = dspy.InputField(desc="Places and locations in the world.")
    plot_timeline: str = dspy.OutputField(desc="A plot timeline.")

class WorldBibleGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_rules = dspy.ChainOfThought(GenerateWorldRulesSignature)
        self.generate_characters = dspy.ChainOfThought(GenerateCharactersSignature)
        self.generate_locations = dspy.ChainOfThought(GenerateLocationsSignature)
        self.generate_timeline = dspy.ChainOfThought(GeneratePlotTimelineSignature)

    @observe()
    def forward(self, core_premise: str, spine_template: str, user_additions: str = ""):
        rules_result = self.generate_rules(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions
        )

        characters_result = self.generate_characters(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
            world_rules=rules_result.world_rules
        )

        locations_result = self.generate_locations(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
            world_rules=rules_result.world_rules,
            characters=characters_result.characters
        )

        timeline_result = self.generate_timeline(
            core_premise=core_premise,
            spine_template=spine_template,
            user_additions=user_additions,
            world_rules=rules_result.world_rules,
            characters=characters_result.characters,
            locations=locations_result.locations
        )

        world_bible = (
            f"### Rules of the World\n{rules_result.world_rules}\n\n"
            f"### Characters\n{characters_result.characters}\n\n"
            f"### Locations\n{locations_result.locations}\n\n"
            f"### Plot Timeline\n{timeline_result.plot_timeline}"
        )

        return dspy.Prediction(
            world_bible=world_bible,
            world_rules=rules_result.world_rules,
            characters=characters_result.characters,
            locations=locations_result.locations,
            plot_timeline=timeline_result.plot_timeline
        )
