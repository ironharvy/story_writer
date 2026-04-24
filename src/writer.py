from collections.abc import Callable

import dspy

from .data import QuestionWithAnswer, Character, Story

AskQuestionsCallback = Callable[[list[QuestionWithAnswer]], list[QuestionWithAnswer]]


class ClarifyIdea(dspy.Signature):
    """Come up with questions and proposed answers to clarify the user's idea."""

    idea: str = dspy.InputField(desc="The user's initial story idea")
    questions_with_answers: list[QuestionWithAnswer] = dspy.OutputField(
        desc="A list of questions with proposed answers",
    )

class DefineCorePremise(dspy.Signature):
    """Define the core premise of the story based on the answered questions."""
    
    idea: str = dspy.InputField(desc="The user's initial story idea")
    answered_questions: list[QuestionWithAnswer] = dspy.InputField(desc="List of answered questions")
    core_premise: str = dspy.OutputField(desc="The core premise of the story")

class DefineSpine(dspy.Signature):
    """Define the spline of the story based on the core premise."""
    
    core_premise: str = dspy.InputField(desc="The core premise of the story")
    spine: str = dspy.OutputField(desc="The spine of the story")

class DefineCharacters(dspy.Signature):
    """Define the characters of the story based on the spine."""
    
    spine: str = dspy.InputField(desc="The spine of the story")
    characters: list[Character] = dspy.OutputField(desc="The characters of the story")


class Writer:
    def __init__(self, config: dict[str, str]) -> None:
        provider_str = f"{config['provider']}:{config['model']}"
        self.lm = dspy.LM(
            provider_str,
            api_key=config["api_key"],
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000),
        )
        self.clarify_idea = dspy.ChainOfThought(ClarifyIdea)
        self.define_core_premise = dspy.ChainOfThought(DefineCorePremise)
        self.define_spine = dspy.ChainOfThought(DefineSpine)

    def compose(
        self,
        idea: str,
        ask_questions: AskQuestionsCallback,
    ) -> Story:
        story = Story(idea=idea)
        result = self.clarify_idea(idea=idea)
        answered = ask_questions(result.questions_with_answers)
        
        core_premise = self.define_core_premise(idea=idea, answered_questions=answered).core_premise
        qa = QuestionWithAnswer(question="What is the core premise of the story?", answer=core_premise)
        answered = ask_questions([qa])

        story.core_premise = answered[0].answer

        spine = self.define_spine(core_premise=core_premise).spine
        qa = QuestionWithAnswer(question="What is the spine of the story?", answer=spine)
        answered = ask_questions([qa])

        story.spine = answered[0].answer


        
        
        return core_premise