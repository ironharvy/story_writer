import dspy
import logging
import random
import re
from typing import List
from pydantic import BaseModel, Field, model_validator
from typing import Any
from _compat import observe

# Probability that a chapter receives a random creative flourish (0.0 – 1.0).
RANDOM_DETAIL_PROBABILITY = 0.35

# The kinds of random detail the LLM can be asked to invent.
_RANDOM_DETAIL_TYPES = [
    "an unusually long and vivid description of a piece of scenery or environment",
    "a quirky or unexpected object placed naturally in the scene (e.g. a paper unicorn on a desk, a rusted music-box on a windowsill)",
    "a strange but fitting atmospheric detail involving sounds, smells, or textures",
    "an unusual yet revealing character habit, nervous tic, or physical detail",
    "a brief, surprising background element that enriches the world without derailing the plot",
]

logger = logging.getLogger(__name__)

# Regex to strip leading "Chapter <number>:" / "Chapter <number> -" from LLM-generated titles
_chapter_prefix_re = re.compile(
    r'^(chapter\s+\d+\s*[:\-\.]\s*)',
    re.IGNORECASE,
)


def _clean_chapter_title(raw_title: str) -> str:
    title = (raw_title or "").strip()
    title = re.sub(r'^\s*#+\s*', '', title)
    title = title.strip()
    title = title.strip('"\'')

    if (title.startswith("**") and title.endswith("**")) or (
        title.startswith("__") and title.endswith("__")
    ):
        title = title[2:-2].strip()

    title = _chapter_prefix_re.sub('', title).strip()
    logger.debug("Cleaned chapter title: %s", title)
    return title.strip('"\'')


def _normalize_chapter_plan_entries(chapters: list[str]) -> list[str]:
    normalized: list[str] = []
    for index, chapter in enumerate(chapters, start=1):
        chapter_text = (chapter or "").strip()
        chapter_text = _clean_chapter_title(chapter_text)
        if not chapter_text:
            chapter_text = f"Untitled Chapter {index}"
        normalized.append(f"Chapter {index}: {chapter_text}")

    logger.debug("Normalized chapter plan entries: %s", normalized)
    return normalized

class QuestionWithAnswer(BaseModel):
    question: str = Field(description="The interrogative question.")
    proposed_answer: str = Field(description="A proposed answer for the user to potentially accept.")

    @model_validator(mode='before')
    @classmethod
    def fix_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        def normalize_key(key: str) -> str:
            return ''.join(ch for ch in key.lower() if ch.isalnum())

        question_aliases = {"question", "q", "prompt", "query"}
        answer_aliases = {"proposedanswer", "answer", "response", "a"}

        # Preserve already-correct keys first.
        if "question" not in normalized:
            if "" in normalized and isinstance(normalized[""], str):
                normalized["question"] = normalized.pop("")
            else:
                for key in list(normalized.keys()):
                    if key in {"question", "proposed_answer"}:
                        continue
                    if normalize_key(key) in question_aliases and isinstance(normalized[key], str):
                        normalized["question"] = normalized.pop(key)
                        break

        if "proposed_answer" not in normalized:
            for key in list(normalized.keys()):
                if key in {"question", "proposed_answer"}:
                    continue
                if normalize_key(key) in answer_aliases and isinstance(normalized[key], str):
                    normalized["proposed_answer"] = normalized.pop(key)
                    break

        return normalized

class GenerateQuestionsSignature(dspy.Signature):
    """Generates 5 interrogative questions to interrogate the user's idea to generate a 'Core Premise'. Each question must include a proposed answer."""
    idea: str = dspy.InputField(desc="The initial idea or prompt for the story.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(desc="5 interrogative questions with proposed answers.")

class QuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQuestionsSignature)

    @observe()
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

    @observe()
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

    @observe()
    def forward(self, core_premise: str):
        return self.generate(core_premise=core_premise)


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

    @model_validator(mode='before')
    @classmethod
    def normalize_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)
        description = normalized.get("description") or normalized.get("visual_description")

        if isinstance(description, str):
            if "reference_mix" not in normalized:
                normalized["reference_mix"] = description
            if "distinguishing_features" not in normalized:
                normalized["distinguishing_features"] = description
            if "full_prompt" not in normalized:
                subject = normalized.get("name", "character")
                normalized["full_prompt"] = f"anime portrait of {subject}, {description}"

        return normalized


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

    @observe()
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

    @observe()
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
    act: str = dspy.InputField(desc="The act of the story.")
    arc_outline: list[str] = dspy.OutputField(desc="Arc Outline (5-10 major events of the act).")

class GenerateChapterPlanSignature(dspy.Signature):
    """Generates Level 2: Chapter Plan (Each arc broken into chapters)."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    act: str = dspy.InputField(desc="The act of the story.")
    #arc: str = dspy.InputField(desc="The arc of the story.")
    chapter_plan: list[str] = dspy.OutputField(desc="Chapter Plan for the arc (5-10 major events of the act)")

class GenerateEnhancersSignature(dspy.Signature):
    """Evaluates the chapter plan and determines which story enhancers are needed for specific scenes/chapters.
    Story Enhancers include: Tension Module, Mystery Module, Theme Alignment, Setup/Payoff Tracker, Emotional Curve, Twist Generator, Easter Egg Injector."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    chapter_plan: str = dspy.InputField(desc="Chapter Plan (Each arc broken into chapters).")
    enhancers_guide: str = dspy.OutputField(desc="A guide evaluating which story enhancers (e.g., Tension, Mystery, Twists) are needed for specific scenes or chapters and how to apply them.")

class GenerateRandomDetailSignature(dspy.Signature):
    """Invents a single concrete, contextually-appropriate creative flourish to be woven into a chapter.
    The detail must fit the story's world and genre—no anachronisms or setting violations.
    Output only the detail itself as a short, vivid description (2-5 sentences)."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible describing the story's setting, rules, and characters.")
    current_chapter_description: str = dspy.InputField(desc="The specific chapter event that the detail should fit into.")
    detail_type: str = dspy.InputField(desc="The category of creative flourish to invent.")
    random_detail: str = dspy.OutputField(desc="A concrete, vivid creative detail (2-5 sentences) that fits the world and can be naturally woven into the chapter.")


class GenerateSingleChapterSignature(dspy.Signature):
    """Writes a full, detailed chapter based on the world bible and the specific chapter goal."""
    world_bible: str = dspy.InputField()
    chapter_plan: str = dspy.InputField(desc="The full plan for context.")
    current_chapter_description: str = dspy.InputField(desc="The specific event to write now.")
    previous_chapters_summary: str = dspy.InputField(desc="Brief summary of what happened so far.")
    enhancers_guide: str = dspy.InputField(desc="A guide evaluating which story enhancers (e.g., Tension, Mystery, Twists) are needed for specific scenes or chapters and how to apply them.")
    random_detail: str = dspy.InputField(desc="An optional creative flourish to weave naturally into the chapter. Empty string means no special detail is required.")
    title: str = dspy.OutputField(desc="The title of the chapter.")
    chapter_text: str = dspy.OutputField(desc="A long, immersive chapter with dialogue and description.")

class StoryGenerator(dspy.Module):
    def __init__(self, random_detail_probability: float = RANDOM_DETAIL_PROBABILITY):
        if not (0.0 <= random_detail_probability <= 1.0):
            raise ValueError(
                f"random_detail_probability must be in [0.0, 1.0], got {random_detail_probability}"
            )
        super().__init__()
        self.random_detail_probability = random_detail_probability
        #self.generate_arc_outline = dspy.ChainOfThought(GenerateArcOutlineSignature)
        self.generate_chapter_plan = dspy.ChainOfThought(GenerateChapterPlanSignature)
        self.generate_enhancers = dspy.ChainOfThought(GenerateEnhancersSignature)
        self.generate_random_detail = dspy.Predict(GenerateRandomDetailSignature)
        self.write_chapter = dspy.ChainOfThought(GenerateSingleChapterSignature)

    def _maybe_generate_random_detail(self, world_bible: str, chapter_desc: str) -> str:
        """Roll the dice and, if triggered, generate a contextual creative flourish."""
        if random.random() >= self.random_detail_probability:
            return ""
        detail_type = random.choice(_RANDOM_DETAIL_TYPES)
        logger.debug("Probabilistic detail triggered for chapter (type: %s)", detail_type)
        try:
            result = self.generate_random_detail(
                world_bible=world_bible,
                current_chapter_description=chapter_desc,
                detail_type=detail_type,
            )
            return result.random_detail
        except Exception as e:
            logger.warning("Failed to generate random detail: %s", e)
            return ""

    @observe()
    def forward(self, core_premise: str, spine_template: str, world_bible: str):
        chapters_to_write = []
        for act in ["Act 1", "Act 2", "Act 3"]:
            logger.debug("Generating chapter plan for %s...", act)
            chapter_plan_result = self.generate_chapter_plan(
                core_premise=core_premise,
                world_bible=world_bible,
                act=act,
            )
            chapters_to_write.extend(chapter_plan_result.chapter_plan)
            logger.debug("Added chapters: %s", chapter_plan_result.chapter_plan)


        chapters_to_write = _normalize_chapter_plan_entries(chapters_to_write)

        chapter_plan_text = "\n".join(chapters_to_write)

        enhancers_result = self.generate_enhancers(
            world_bible=world_bible,
            chapter_plan=chapter_plan_text
        )

        full_story = ""
        previous_chapters_summary = ""

        for i, chapter_desc in enumerate(chapters_to_write):
            try:
                random_detail = self._maybe_generate_random_detail(world_bible, chapter_desc)
                if random_detail:
                    logger.info("Chapter %d: injecting probabilistic detail.", i + 1)

                result = self.write_chapter(
                    world_bible=world_bible,
                    chapter_plan=chapter_plan_text,
                    current_chapter_description=chapter_desc,
                    previous_chapters_summary=previous_chapters_summary,
                    enhancers_guide=enhancers_result.enhancers_guide,
                    random_detail=random_detail,
                )

                chapter_text = result.chapter_text
                # Strip markdown wrappers and redundant "Chapter N:" prefixes.
                clean_title = _clean_chapter_title(result.title)
                if not clean_title:
                    clean_title = _clean_chapter_title(chapter_desc)
                full_story += f"\n\n### Chapter {i+1}: {clean_title}\n\n" + chapter_text
                previous_chapters_summary += f"Chapter {i+1}: {chapter_desc}\n"
            except Exception as e:
                logger.error(f"Error writing chapter {i+1}: {e}", exc_info=True)
                break

        if not full_story.strip():
            logger.warning(
                "Story generation produced no chapter content. "
                "chapters_to_write=%d, chapter_plan excerpt=%.500s",
                len(chapters_to_write),
                chapter_plan_result.chapter_plan,
            )

        return dspy.Prediction(
            arc_outline="[REMOVED]",#arc_outline=arc_outline_result.arc_outline,
            chapter_plan=chapter_plan_text,
            enhancers_guide=enhancers_result.enhancers_guide,
            story=full_story.strip()
        )
