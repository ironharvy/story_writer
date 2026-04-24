"""Core DSPy modules for story generation and chapter post-processing."""

import logging
import random
import re
from typing import Any, List

import dspy
from pydantic import BaseModel, Field, model_validator

from _compat import observe
from exceptions import RECOVERABLE_MODEL_EXCEPTIONS

# Probability that a chapter receives a random creative flourish (0.0 – 1.0).
RANDOM_DETAIL_PROBABILITY = 0.35

# The kinds of random detail the LLM can be asked to invent.
_RANDOM_DETAIL_TYPES = [
    "an unusually long and vivid description of a piece of scenery or environment",
    (
        "a quirky or unexpected object placed naturally in the scene "
        "(e.g. a paper unicorn on a desk, a rusted music-box on a windowsill)"
    ),
    "a strange but fitting atmospheric detail involving sounds, smells, or textures",
    "an unusual yet revealing character habit, nervous tic, or physical detail",
    "a brief, surprising background element that enriches the world without derailing the plot",
]

_ACT_SEQUENCE = [
    "Act 1 - Setup",
    "Act 2 - Confrontation",
    "Act 3 - Resolution",
]

_chapter_heading_re = re.compile(r"^###\s+Chapter\s+\d+:.*$", re.MULTILINE)

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


def _split_story_into_chapters(story_text: str) -> list[tuple[str, str]]:
    chapter_matches = list(_chapter_heading_re.finditer(story_text or ""))
    if not chapter_matches:
        return []

    chapters: list[tuple[str, str]] = []
    for index, match in enumerate(chapter_matches):
        body_start = match.end()
        body_end = (
            chapter_matches[index + 1].start()
            if index + 1 < len(chapter_matches)
            else len(story_text)
        )
        header = match.group(0).strip()
        body = story_text[body_start:body_end].strip()
        chapters.append((header, body))

    return chapters


def _compose_story_from_chapters(chapters: list[tuple[str, str]]) -> str:
    chapter_blocks = [
        f"{chapter_header}\n\n{chapter_body}".strip()
        for chapter_header, chapter_body in chapters
    ]
    return "\n\n".join(chapter_blocks).strip()

class QuestionWithAnswer(BaseModel):
    """A generated follow-up question paired with a suggested answer."""

    question: str = Field(description="The interrogative question.")
    proposed_answer: str = Field(
        description="A proposed answer for the user to potentially accept.",
    )

    @model_validator(mode='before')
    @classmethod
    def fix_keys(cls, data: Any) -> Any:
        """Normalize loosely-structured model output keys to canonical fields."""
        _ = cls
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
    """Generate 5 interrogative questions for premise clarification.

    Each question includes a proposed answer.
    """

    idea: str = dspy.InputField(desc="The initial idea or prompt for the story.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(
        desc="5 interrogative questions with proposed answers.",
    )

class QuestionGenerator(dspy.Module):
    """Generate interactive clarification questions for the user's idea."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateQuestionsSignature)

    @observe()
    def forward(self, idea: str):
        """Generate question/answer suggestions from the initial idea."""
        return self.generate(idea=idea)


class GenerateCorePremiseSignature(dspy.Signature):
    """Synthesizes the user's idea, the questions, and the user's answers into a Core Premise."""
    idea: str = dspy.InputField(desc="The initial idea or prompt for the story.")
    qa_pairs: str = dspy.InputField(
        desc="The interrogative questions and the user's accepted or provided answers.",
    )
    core_premise: str = dspy.OutputField(
        desc="A detailed Core Premise summarizing the foundation of the story.",
    )

class CorePremiseGenerator(dspy.Module):
    """Synthesize idea context into a single core premise statement."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateCorePremiseSignature)

    @observe()
    def forward(self, idea: str, qa_pairs: str):
        """Build the core premise from the idea and Q/A refinements."""
        return self.generate(idea=idea, qa_pairs=qa_pairs)


class GenerateSpineTemplateSignature(dspy.Signature):
    """Creates a narrative spine template based on the Core Premise."""
    idea: str = dspy.InputField(desc="The original story idea.")
    qa_pairs: str = dspy.InputField(desc="Questions and answers to flesh out the story.")
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.OutputField(
        desc=(
            "A narrative spine template (e.g., Once upon a time... Every day... "
            "One day... Because of that... Because of that... Until finally...)."
        ),
    )

class SpineTemplateGenerator(dspy.Module):
    """Generate a narrative spine scaffold for downstream planning."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSpineTemplateSignature)

    @observe()
    def forward(self, idea: str, qa_pairs: str, core_premise: str):
        """Create a spine template from ideation and premise context."""
        return self.generate(idea=idea, qa_pairs=qa_pairs, core_premise=core_premise)


class CharacterVisual(BaseModel):
    """Structured visual profile used for character portrait generation."""

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
        """Backfill legacy fields into the current character-visual schema."""
        _ = cls
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
    """Extract and describe the story's major character visuals."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateCharacterVisualsSignature)

    @observe()
    def forward(self, world_bible: str):
        """Generate normalized visual descriptors from a world bible."""
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
    """Produce chapter-level scene prompts for image generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict(GenerateSceneImagePromptSignature)

    @observe()
    def forward(self, chapter_text: str, character_visuals_summary: str):
        """Generate an image prompt for a chapter's most salient scene."""
        return self.generate(
            chapter_text=chapter_text,
            character_visuals_summary=character_visuals_summary,
        )


class GenerateChapterPlanSignature(dspy.Signature):
    """Generates Level 2: Chapter Plan for one act, continuing from prior acts.

    Plan only the current act. Previously planned chapters from earlier acts are
    provided as context — continue the narrative from where they leave off,
    without repeating beats, scenes, or chapter titles.
    """
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    spine_template: str = dspy.InputField(
        desc=(
            "The Spine Template describing the overall narrative shape "
            "(Once upon a time / Every day / One day / Because of that / "
            "Until finally). Use it to anchor the act boundary."
        ),
    )
    characters: str = dspy.InputField(
        desc="Character bios and relationship context most relevant to chapter planning.",
    )
    plot_timeline: str = dspy.InputField(
        desc="Plot beats and act progression for the story.",
    )
    rules: str = dspy.InputField(
        desc="Rules of the world that constrain or enable plot events.",
    )
    previous_chapters: str = dspy.InputField(
        desc=(
            "Chapters already planned in earlier acts, one per line. "
            "Continue from these — do not repeat or restate any beat, scene, "
            "or chapter title. Empty string means this is the first act."
        ),
    )
    act: str = dspy.InputField(desc="The act of the story to plan chapters for.")
    chapter_plan: list[str] = dspy.OutputField(
        desc=(
            "Chapter Plan for this act (5-10 major events). Each entry must "
            "advance the story past previous_chapters and must not duplicate "
            "any prior beat or chapter title."
        ),
    )

class GenerateEnhancersSignature(dspy.Signature):
    """Evaluate chapter plan needs for story-enhancer techniques.

    Story enhancers include tension, mystery, theme alignment,
    setup/payoff tracking, emotional curve shaping, twists, and easter eggs.
    """

    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    chapter_plan: str = dspy.InputField(desc="Chapter Plan (Each arc broken into chapters).")
    enhancers_guide: str = dspy.OutputField(
        desc=(
            "A guide evaluating which story enhancers (e.g., Tension, Mystery, "
            "Twists) are needed for specific scenes or chapters and how to apply "
            "them."
        ),
    )

class GenerateRandomDetailSignature(dspy.Signature):
    """Invent a concrete, contextual creative flourish for a chapter.

    The detail must fit the story's world and genre.
    """

    world_bible: str = dspy.InputField(
        desc=(
            "The comprehensive World Bible describing the story's setting, "
            "rules, and characters."
        ),
    )
    current_chapter_description: str = dspy.InputField(
        desc="The specific chapter event that the detail should fit into.",
    )
    detail_type: str = dspy.InputField(desc="The category of creative flourish to invent.")
    random_detail: str = dspy.OutputField(
        desc=(
            "A concrete, vivid creative detail (2-5 sentences) that fits the "
            "world and can be naturally woven into the chapter."
        ),
    )


class GenerateSingleChapterSignature(dspy.Signature):
    """Writes a full, detailed chapter based on the world bible and the specific chapter goal."""
    characters: str = dspy.InputField(
        desc="Character bios and relationship context to keep names and motivations specific.",
    )
    rules: str = dspy.InputField(
        desc="Rules of the world that govern what can happen in the chapter.",
    )
    locations: str = dspy.InputField(
        desc="Relevant location details to ground setting and scene description.",
    )
    chapter_plan: str = dspy.InputField(desc="The full plan for context.")
    current_chapter_description: str = dspy.InputField(
        desc="The specific event to write now.",
    )
    previous_chapters_summary: str = dspy.InputField(
        desc="Brief summary of what happened so far.",
    )
    enhancers_guide: str = dspy.InputField(
        desc=(
            "A guide evaluating which story enhancers (e.g., Tension, Mystery, "
            "Twists) are needed for specific scenes or chapters and how to apply "
            "them."
        ),
    )
    random_detail: str = dspy.InputField(
        desc=(
            "An optional creative flourish to weave naturally into the chapter. "
            "Empty string means no special detail is required."
        ),
    )
    title: str = dspy.OutputField(desc="The title of the chapter.")
    chapter_text: str = dspy.OutputField(
        desc="A long, immersive chapter with dialogue and description.",
    )


class GenerateChapterInpaintingSignature(dspy.Signature):
    """Expands an existing chapter with richer detail while preserving plot facts."""

    world_bible: str = dspy.InputField(
        desc="The comprehensive World Bible describing setting, rules, and characters.",
    )
    chapter_plan: str = dspy.InputField(
        desc="The full chapter plan for continuity and pacing context.",
    )
    chapter_header: str = dspy.InputField(
        desc="The chapter markdown header (e.g., '### Chapter 4: ...').",
    )
    chapter_text: str = dspy.InputField(
        desc="The already-written chapter text to be expanded.",
    )
    expansion_ratio: float = dspy.InputField(
        desc="Target expansion ratio. Add detail to approach this length multiplier without bloat.",
    )
    expanded_chapter_text: str = dspy.OutputField(
        desc=(
            "A more detailed chapter that preserves all major events, outcomes, and continuity. "
            "Do not introduce major new plot points."
        ),
    )


class ChapterInpaintingGenerator(dspy.Module):
    """Expand existing chapter drafts with richer narrative detail."""

    def __init__(self):
        super().__init__()
        self.expand_chapter = dspy.ChainOfThought(GenerateChapterInpaintingSignature)

    @observe()
    def forward(
        self,
        story: str,
        world_bible: str,
        chapter_plan: str,
        expansion_ratio: float = 1.35,
    ):
        """Expand each chapter while preserving chapter order and story facts."""
        if expansion_ratio <= 1.0:
            raise ValueError(f"expansion_ratio must be > 1.0, got {expansion_ratio}")

        chapters = _split_story_into_chapters(story)
        if not chapters:
            logger.warning("Chapter inpainting skipped: no chapter headings detected.")
            return dspy.Prediction(
                story=story,
                expanded_chapters=0,
                total_chapters=0,
            )

        expanded_chapters: list[tuple[str, str]] = []
        expanded_count = 0
        for chapter_header, chapter_text in chapters:
            try:
                result = self.expand_chapter(
                    world_bible=world_bible,
                    chapter_plan=chapter_plan,
                    chapter_header=chapter_header,
                    chapter_text=chapter_text,
                    expansion_ratio=expansion_ratio,
                )
                expanded_chapter_text = result.expanded_chapter_text.strip()
                if expanded_chapter_text:
                    expanded_count += 1
                    expanded_chapters.append((chapter_header, expanded_chapter_text))
                else:
                    expanded_chapters.append((chapter_header, chapter_text))
            except RECOVERABLE_MODEL_EXCEPTIONS as exc:
                logger.warning(
                    "Chapter inpainting failed for %s: %s",
                    chapter_header,
                    exc,
                )
                expanded_chapters.append((chapter_header, chapter_text))

        return dspy.Prediction(
            story=_compose_story_from_chapters(expanded_chapters),
            expanded_chapters=expanded_count,
            total_chapters=len(chapters),
        )

class StoryGenerator(dspy.Module):
    """Generate chapter plans, enhancer guidance, and full story chapters."""

    def __init__(self, random_detail_probability: float = RANDOM_DETAIL_PROBABILITY):
        if not 0.0 <= random_detail_probability <= 1.0:
            raise ValueError(
                f"random_detail_probability must be in [0.0, 1.0], got {random_detail_probability}"
            )
        super().__init__()
        self.random_detail_probability = random_detail_probability
        self.generate_chapter_plan = dspy.ChainOfThought(GenerateChapterPlanSignature)
        self.generate_enhancers = dspy.ChainOfThought(GenerateEnhancersSignature)
        self.generate_random_detail = dspy.Predict(GenerateRandomDetailSignature)
        self.write_chapter = dspy.ChainOfThought(GenerateSingleChapterSignature)

    def _maybe_generate_random_detail(self, world_bible: str, chapter_desc: str) -> str:
        """Roll the dice and, if triggered, generate a contextual creative flourish."""
        roll = random.random()
        triggered = roll < self.random_detail_probability
        logger.debug(
            "Random detail roll=%.4f threshold=%.4f triggered=%s",
            roll,
            self.random_detail_probability,
            triggered,
        )
        if not triggered:
            logger.debug("Random detail skipped for chapter.")
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
        except RECOVERABLE_MODEL_EXCEPTIONS as exc:
            logger.warning("Failed to generate random detail: %s", exc)
            return ""

    def _generate_chapter_plan_entries(
        self,
        core_premise: str,
        spine_template: str,
        world_bible: WorldBible,
    ) -> list[str]:
        chapter_entries: list[str] = []
        for act in _ACT_SEQUENCE:
            previous_chapters_text = "\n".join(chapter_entries)
            logger.debug(
                "Generating chapter plan for %s (continuing from %d prior chapters)...",
                act,
                len(chapter_entries),
            )
            chapter_plan_result = self.generate_chapter_plan(
                core_premise=core_premise,
                spine_template=spine_template,
                characters=world_bible.characters,
                plot_timeline=world_bible.plot_timeline,
                rules=world_bible.rules,
                previous_chapters=previous_chapters_text,
                act=act,
            )
            chapter_entries.extend(chapter_plan_result.chapter_plan)
            logger.debug("Added chapters: %s", chapter_plan_result.chapter_plan)

        return _normalize_chapter_plan_entries(chapter_entries)

    def _generate_enhancers_guide(
        self,
        world_bible: WorldBible,
        chapter_plan_text: str,
    ) -> str:
        enhancers_result = self.generate_enhancers(
            world_bible=world_bible.full_text,
            chapter_plan=chapter_plan_text,
        )
        return enhancers_result.enhancers_guide

    def _write_story_chapters(
        self,
        world_bible: WorldBible,
        chapter_plan_text: str,
        chapters_to_write: list[str],
        enhancers_guide: str,
    ) -> str:
        full_story = ""
        previous_chapters_summary = ""

        for index, chapter_desc in enumerate(chapters_to_write, start=1):
            try:
                random_detail = self._maybe_generate_random_detail(
                    world_bible.full_text,
                    chapter_desc,
                )
                if random_detail:
                    logger.info(
                        "Chapter %d: injecting probabilistic detail: %.300s",
                        index,
                        random_detail,
                    )

                result = self.write_chapter(
                    characters=world_bible.characters,
                    rules=world_bible.rules,
                    locations=world_bible.locations,
                    chapter_plan=chapter_plan_text,
                    current_chapter_description=chapter_desc,
                    previous_chapters_summary=previous_chapters_summary,
                    enhancers_guide=enhancers_guide,
                    random_detail=random_detail,
                )

                logger.debug("Chapter %d written: %.300s", index, result.chapter_text)
                clean_title = _clean_chapter_title(result.title)
                if not clean_title:
                    clean_title = _clean_chapter_title(chapter_desc)

                full_story += f"\n\n### Chapter {index}: {clean_title}\n\n{result.chapter_text}"
                previous_chapters_summary += f"Chapter {index}: {chapter_desc}\n"
            except RECOVERABLE_MODEL_EXCEPTIONS as exc:
                logger.error("Error writing chapter %d: %s", index, exc, exc_info=True)
                break

        return full_story

    @observe()
    def forward(
        self,
        core_premise: str,
        spine_template: str,
        world_bible: WorldBible,
    ):
        """Run the end-to-end chapter planning and drafting workflow."""
        logger.debug("StoryGenerator received spine template of %d chars", len(spine_template))
        chapters_to_write = self._generate_chapter_plan_entries(
            core_premise=core_premise,
            spine_template=spine_template,
            world_bible=world_bible,
        )
        chapter_plan_text = "\n".join(chapters_to_write)

        enhancers_guide = self._generate_enhancers_guide(
            world_bible=world_bible,
            chapter_plan_text=chapter_plan_text,
        )
        full_story = self._write_story_chapters(
            world_bible=world_bible,
            chapter_plan_text=chapter_plan_text,
            chapters_to_write=chapters_to_write,
            enhancers_guide=enhancers_guide,
        )

        if not full_story.strip():
            logger.warning(
                "Story generation produced no chapter content. "
                "chapters_to_write=%d, chapter_plan excerpt=%.500s",
                len(chapters_to_write),
                chapter_plan_text,
            )

        return dspy.Prediction(
            chapter_plan=chapter_plan_text,
            enhancers_guide=enhancers_guide,
            story=full_story.strip(),
        )
