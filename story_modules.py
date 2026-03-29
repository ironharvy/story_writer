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


class CharacterVisual(BaseModel):
    name: str = Field(description="The character's name.")
    reference_mix: str = Field(
        description="A visual anchor describing who the character looks like, "
        "e.g. 'a mix of Cinderella and Princess Pingyang'. "
        "Use well-known fictional or historical figures."
    )
    distinguishing_features: str = Field(
        description="Specific distinguishing visual traits: hair color/style, "
        "eye color, clothing, accessories, scars, etc."
    )
    full_prompt: str = Field(
        description="A complete, self-contained anime image-generation prompt "
        "for this character's portrait, combining the reference mix and "
        "distinguishing features into a single descriptive paragraph."
    )


class GenerateCharacterVisualsSignature(dspy.Signature):
    """Extracts the main characters from the world bible and generates a
    structured visual description for each one.  Each character's appearance
    must be anchored to a mix of 2-3 well-known fictional or historical
    figures so that an anime image generator can produce consistent results.
    """
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    character_visuals: List[CharacterVisual] = dspy.OutputField(
        desc="A list of visual descriptions, one per main character."
    )


class CharacterVisualDescriber(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateCharacterVisualsSignature)

    def forward(self, world_bible: str):
        return self.generate(world_bible=world_bible)


class GenerateSceneImagePromptSignature(dspy.Signature):
    """Generates a single anime image-generation prompt that depicts the most
    important scene from the given chapter.  The prompt must reference the
    characters by their visual descriptors (the reference_mix and
    distinguishing_features) so the image stays visually consistent with
    their portraits.  Output ONLY the image prompt, nothing else.
    """
    chapter_text: str = dspy.InputField(desc="The full text of the chapter.")
    character_visuals_summary: str = dspy.InputField(
        desc="A summary of all character visual descriptors to reference."
    )
    image_prompt: str = dspy.OutputField(
        desc="A detailed anime scene image-generation prompt."
    )


class SceneImagePromptGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSceneImagePromptSignature)

    def forward(self, chapter_text: str, character_visuals_summary: str):
        return self.generate(
            chapter_text=chapter_text,
            character_visuals_summary=character_visuals_summary,
        )


class GenerateArcOutlineSignature(dspy.Signature):
    """Generates Level 1: Arc Outline (5-10 major events)."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(desc="The narrative spine template.")
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    arc_outline: str = dspy.OutputField(desc="Arc Outline (5-10 major events).")

class GenerateChapterPlanSignature(dspy.Signature):
    """Generates Level 2: Chapter Plan (Each arc broken into chapters)."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    arc_outline: str = dspy.InputField(desc="Arc Outline (5-10 major events).")
    chapter_plan: str = dspy.OutputField(desc="Chapter Plan (Each arc broken into chapters).")

class GenerateEnhancersSignature(dspy.Signature):
    """Evaluates the chapter plan and determines which story enhancers are needed for specific scenes/chapters.
    Story Enhancers include: Tension Module, Mystery Module, Theme Alignment, Setup/Payoff Tracker, Emotional Curve, Twist Generator, Easter Egg Injector."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    chapter_plan: str = dspy.InputField(desc="Chapter Plan (Each arc broken into chapters).")
    enhancers_guide: str = dspy.OutputField(desc="A guide evaluating which story enhancers (e.g., Tension, Mystery, Twists) are needed for specific scenes or chapters and how to apply them.")

class GenerateSingleChapterSignature(dspy.Signature):
    """Writes a full, detailed chapter based on the world bible and the specific chapter goal."""
    world_bible: str = dspy.InputField()
    chapter_plan: str = dspy.InputField(desc="The full plan for context.")
    current_chapter_description: str = dspy.InputField(desc="The specific event to write now.")
    previous_chapters_summary: str = dspy.InputField(desc="Brief summary of what happened so far.")
    enhancers_guide: str = dspy.InputField(desc="A guide evaluating which story enhancers (e.g., Tension, Mystery, Twists) are needed for specific scenes or chapters and how to apply them.")
    title: str = dspy.OutputField(desc="The title of the chapter.")
    chapter_text: str = dspy.OutputField(desc="A long, immersive chapter with dialogue and description.")

class StoryGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_arc_outline = dspy.ChainOfThought(GenerateArcOutlineSignature)
        self.generate_chapter_plan = dspy.ChainOfThought(GenerateChapterPlanSignature)
        self.generate_enhancers = dspy.ChainOfThought(GenerateEnhancersSignature)
        self.write_chapter = dspy.ChainOfThought(GenerateSingleChapterSignature)

    def forward(self, core_premise: str, spine_template: str, world_bible: str):
        arc_outline_result = self.generate_arc_outline(
            core_premise=core_premise,
            spine_template=spine_template,
            world_bible=world_bible
        )
        chapter_plan_result = self.generate_chapter_plan(
            core_premise=core_premise,
            world_bible=world_bible,
            arc_outline=arc_outline_result.arc_outline
        )
        enhancers_result = self.generate_enhancers(
            world_bible=world_bible,
            chapter_plan=chapter_plan_result.chapter_plan
        )

        chapters_to_write = [line.strip() for line in chapter_plan_result.chapter_plan.split('\n') if line.strip()]

        full_story = ""
        previous_chapters_summary = ""

        for i, chapter_desc in enumerate(chapters_to_write):
            try:
                result = self.write_chapter(
                    world_bible=world_bible,
                    chapter_plan=chapter_plan_result.chapter_plan,
                    current_chapter_description=chapter_desc,
                    previous_chapters_summary=previous_chapters_summary,
                    enhancers_guide=enhancers_result.enhancers_guide
                )

                chapter_text = result.chapter_text
                full_story += f"\n\n### Chapter {i+1}: {result.title}\n\n" + chapter_text
                previous_chapters_summary += f"Chapter {i+1}: {chapter_desc}\n"
            except Exception as e:
                print(f"Error writing chapter {i+1}: {e}")
                break

        return dspy.Prediction(
            arc_outline=arc_outline_result.arc_outline,
            chapter_plan=chapter_plan_result.chapter_plan,
            enhancers_guide=enhancers_result.enhancers_guide,
            story=full_story.strip()
        )
