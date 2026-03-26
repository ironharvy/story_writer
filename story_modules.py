import dspy
from typing import List
from pydantic import BaseModel, Field

class QuestionWithAnswer(BaseModel):
    question: str = Field(description="The interrogative question.")
    proposed_answer: str = Field(description="A proposed answer for the user to potentially accept.")

class GenerateQuestionsSignature(dspy.Signature):
    """Generates 5 interrogative questions to interrogate the user's idea to generate a 'Core Premise'. Each question must include a proposed answer."""
    idea: str = dspy.InputField(desc="The initial idea or prompt for the story.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(desc="5 interrogative questions with proposed answers.")

class QuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQuestionsSignature)

    def forward(self, idea: str):
        return self.generate(idea=idea)


class GenerateCorePremiseSignature(dspy.Signature):
    """Synthesizes the user's idea, the questions, and the user's answers into a Core Premise."""
    idea: str = dspy.InputField(desc="The initial idea or prompt for the story.")
    qa_pairs: str = dspy.InputField(desc="The interrogative questions and the user's accepted or provided answers.")
    core_premise: str = dspy.OutputField(desc="A detailed Core Premise summarizing the foundation of the story.")

class CorePremiseGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateCorePremiseSignature)

    def forward(self, idea: str, qa_pairs: str):
        return self.generate(idea=idea, qa_pairs=qa_pairs)


class GenerateSpineTemplateSignature(dspy.Signature):
    """Creates a narrative spine template based on the Core Premise."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.OutputField(desc="A narrative spine template (e.g., Once upon a time... Every day... One day... Because of that... Because of that... Until finally...).")

class SpineTemplateGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSpineTemplateSignature)

    def forward(self, core_premise: str):
        return self.generate(core_premise=core_premise)


class GenerateWorldBibleSignature(dspy.Signature):
    """Generates a World Bible, fleshing out details like setting, lore, characters, etc. Uses the existing context and optionally asks the user questions."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    user_additions: str = dspy.InputField(desc="Additional answers or details provided by the user.")
    world_bible: str = dspy.OutputField(desc="A comprehensive World Bible containing setting, lore, and characters.")

class WorldBibleGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateWorldBibleSignature)

    def forward(self, core_premise: str, spine_template: str, user_additions: str = ""):
        return self.generate(core_premise=core_premise, spine_template=spine_template, user_additions=user_additions)

class GenerateWorldBibleQuestionsSignature(dspy.Signature):
    """Generates a few follow-up questions to ask the user to help flesh out the world bible before generating the final version."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(desc="Up to 3 follow-up interrogative questions with proposed answers to help flesh out the World Bible.")

class WorldBibleQuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateWorldBibleQuestionsSignature)

    def forward(self, core_premise: str, spine_template: str):
        return self.generate(core_premise=core_premise, spine_template=spine_template)


class GeneratePlotSignature(dspy.Signature):
    """Generates the plot including Level 1 (Arc Outline), Level 2 (Chapter Plan), and Level 3 (Scenes)."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    arc_outline: str = dspy.OutputField(desc="Level 1: Arc Outline (5-10 major events).")
    chapter_plan: str = dspy.OutputField(desc="Level 2: Chapter Plan (Each arc broken into chapters).")
    scenes: str = dspy.OutputField(desc="Level 3: Scenes (Actual writing/detailed scene plans).")

class PlotGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GeneratePlotSignature)

    def forward(self, core_premise: str, spine_template: str, world_bible: str):
        return self.generate(core_premise=core_premise, spine_template=spine_template, world_bible=world_bible)
