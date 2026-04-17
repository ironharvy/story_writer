import dspy
import logging
import random
import re
from typing import Any, List
from pydantic import BaseModel, Field, model_validator
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

_chapter_heading_re = re.compile(r"^###\s+Chapter\s+\d+:.*$", re.MULTILINE)

_spine_label_pattern = re.compile(
    r"(once upon a time|every day|one day|because of that|until finally|ever since then)\s*[:,-]?",
    re.IGNORECASE,
)

_SPINE_BEAT_ORDER = [
    "once upon a time",
    "every day",
    "one day",
    "because of that",
    "because of that",
    "until finally",
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


def _deduplicate_chapter_plan_entries(chapters: list[str]) -> list[str]:
    deduplicated: list[str] = []
    seen_keys: set[str] = set()
    for chapter in chapters:
        chapter_text = _clean_chapter_title((chapter or "").strip())
        if not chapter_text:
            continue

        dedupe_key = re.sub(r"[^a-z0-9]+", " ", chapter_text.lower()).strip()
        if dedupe_key in seen_keys:
            logger.debug("Skipping duplicate chapter plan entry: %s", chapter_text)
            continue

        seen_keys.add(dedupe_key)
        deduplicated.append(chapter)

    return deduplicated


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


def _fallback_spine_beats(core_premise: str) -> list[str]:
    premise = " ".join((core_premise or "").split())
    if not premise:
        premise = "a protagonist is pushed into a difficult change"
    return [
        f"Once upon a time, {premise}.",
        "Every day, the protagonist tries to maintain normal life while pressure builds.",
        "One day, a disruptive event forces a point-of-no-return decision.",
        "Because of that, the protagonist takes action and faces escalating consequences.",
        "Because of that, the conflict deepens and demands a costly sacrifice.",
        "Until finally, the protagonist resolves the central conflict and emerges changed.",
    ]


def _normalize_spine_template_with_status(
    spine_template: str,
    core_premise: str,
) -> tuple[str, bool]:
    text = (spine_template or "").replace("```", " ").strip()
    fallback_text = "\n".join(_fallback_spine_beats(core_premise))
    if not text:
        return fallback_text, True

    matches = list(_spine_label_pattern.finditer(text))
    if not matches:
        return fallback_text, True

    extracted_beats: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip(" .:\n\t-—")
        if not body:
            continue
        extracted_beats.append((label, body))

    beat_values: dict[str, list[str]] = {label: [] for label in set(_SPINE_BEAT_ORDER)}
    for label, body in extracted_beats:
        if label in beat_values:
            beat_values[label].append(body)

    if len(extracted_beats) < 4:
        return fallback_text, True

    normalized_lines: list[str] = []
    because_index = 0
    fallback_lines = _fallback_spine_beats(core_premise)
    fallback_idx = 0
    for label in _SPINE_BEAT_ORDER:
        if label == "because of that":
            values = beat_values[label]
            if because_index < len(values):
                content = values[because_index]
                normalized_lines.append(f"Because of that, {content}.")
            else:
                normalized_lines.append(fallback_lines[3 + min(because_index, 1)])
            because_index += 1
        else:
            values = beat_values[label]
            if values:
                content = values[0]
                prefix = label[0].upper() + label[1:]
                normalized_lines.append(f"{prefix}, {content}.")
            else:
                normalized_lines.append(fallback_lines[fallback_idx])
        fallback_idx += 1

    return "\n".join(normalized_lines), False


def _normalize_spine_template(spine_template: str, core_premise: str) -> str:
    normalized, _ = _normalize_spine_template_with_status(spine_template, core_premise)
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
        result = self.generate(core_premise=core_premise)
        normalized_spine, used_fallback = _normalize_spine_template_with_status(
            spine_template=getattr(result, "spine_template", ""),
            core_premise=core_premise,
        )
        if used_fallback:
            logger.warning(
                "Spine template fallback applied due to empty or unstructured model output."
            )
        return dspy.Prediction(spine_template=normalized_spine)


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

class GenerateChapterPlanSignature(dspy.Signature):
    """Generates Level 2: Chapter Plan (Each arc broken into chapters)."""
    core_premise: str = dspy.InputField(desc="The Core Premise of the story.")
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    act: str = dspy.InputField(desc="The act of the story.")
    chapters_so_far: str = dspy.InputField(
        desc=(
            "A newline-separated list of chapters already planned. Do not repeat titles or "
            "major beats from this list. Continue escalating the story from prior chapters."
        ),
    )
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
            except Exception as exc:
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
    def __init__(self, random_detail_probability: float = RANDOM_DETAIL_PROBABILITY):
        if not (0.0 <= random_detail_probability <= 1.0):
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
        except Exception as e:
            logger.warning("Failed to generate random detail: %s", e)
            return ""

    @observe()
    def forward(self, core_premise: str, spine_template: str, world_bible: str):
        chapters_to_write: list[str] = []
        for act in ["Act 1 - Setup", "Act 2 - Confrontation", "Act 3 - Resolution"]:
            logger.debug("Generating chapter plan for %s...", act)
            chapters_so_far = "\n".join(
                _normalize_chapter_plan_entries(chapters_to_write)
            )
            chapter_plan_result = self.generate_chapter_plan(
                core_premise=core_premise,
                world_bible=world_bible,
                act=act,
                chapters_so_far=chapters_so_far,
            )
            chapters_to_write.extend(chapter_plan_result.chapter_plan)
            logger.debug("Added chapters: %s", chapter_plan_result.chapter_plan)

        chapters_to_write = _deduplicate_chapter_plan_entries(chapters_to_write)
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
                    logger.info("Chapter %d: injecting probabilistic detail: %.300s", i + 1, random_detail)

                result = self.write_chapter(
                    world_bible=world_bible,
                    chapter_plan=chapter_plan_text,
                    current_chapter_description=chapter_desc,
                    previous_chapters_summary=previous_chapters_summary,
                    enhancers_guide=enhancers_result.enhancers_guide,
                    random_detail=random_detail,
                )

                logger.debug("Chapter %d written: %.300s", i + 1, result.chapter_text)
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
            chapter_plan=chapter_plan_text,
            enhancers_guide=enhancers_result.enhancers_guide,
            story=full_story.strip()
        )
