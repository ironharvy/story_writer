import dspy
import os
import argparse
import logging
import coloredlogs
import pytest
from unittest.mock import patch, MagicMock
from story_modules import (
    QuestionGenerator,
    CorePremiseGenerator,
    SpineTemplateGenerator,
    StoryGenerator,
    CharacterVisualDescriber,
    SceneImagePromptGenerator,
    extract_chapter_descriptions,
)
from world_bible_modules import WorldBibleGenerator

# A mock LM to avoid needing an API key for automated testing
class MockLM(dspy.LM):
    def __init__(self):
        super().__init__(model="mock")

    def __call__(self, prompt=None, messages=None, **kwargs):
        content = prompt if prompt else str(messages)

        if "[[ ## character_visuals ## ]]" in content or ('"character_visuals"' in content and "reference_mix" in content):
            return ['```json\n{"character_visuals": [{"name": "Mock Hero", "reference_mix": "a mix of Guts from Berserk and Zuko from Avatar", "distinguishing_features": "short black hair, amber eyes, burn scar on left cheek, dark leather armor", "full_prompt": "anime portrait, a mix of Guts from Berserk and Zuko from Avatar, short black hair, amber eyes, burn scar on left cheek, dark leather armor"}]}\n```']
        if "[[ ## image_prompt ## ]]" in content or ('"image_prompt"' in content and "anime scene" in content):
            return ['```json\n{"image_prompt": "anime scene, a warrior with short black hair and amber eyes standing in a dark cathedral, dramatic lighting"}\n```']
        if "[[ ## chapter_text ## ]]" in content or ('"chapter_text"' in content and "immersive chapter" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "title": "Mock Title", "chapter_text": "Mock chapter text"}\n```']
        if "[[ ## enhancers_guide ## ]]" in content or ('"enhancers_guide"' in content and "evaluating which story enhancers" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "enhancers_guide": "Mock enhancers guide"}\n```']
        if "[[ ## chapter_plan ## ]]" in content or ('"chapter_plan"' in content and "Each arc broken into chapters" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "chapter_plan": "- **Chapter 1**: Mock chapter one"}\n```']
        if "[[ ## arc_outline ## ]]" in content or ('"arc_outline"' in content and "5-10 major events" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "arc_outline": "Mock arc outline"}\n```']
        if "[[ ## world_bible ## ]]" in content or ('"world_bible"' in content and "setting, lore, and characters" in content):
            return ['```json\n{"world_bible": "Mock world bible"}\n```']
        if "[[ ## plot_timeline ## ]]" in content or ('"plot_timeline"' in content and "A plot timeline." in content):
            return ['```json\n{"reasoning": "Mock reasoning", "plot_timeline": "Mock timeline"}\n```']
        if "[[ ## locations ## ]]" in content or ('"locations"' in content and "Places and locations" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "locations": "Mock locations"}\n```']
        if "[[ ## characters ## ]]" in content or ('"characters"' in content and "Character descriptions" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "characters": "Mock characters"}\n```']
        if "[[ ## world_rules ## ]]" in content or ('"world_rules"' in content and "Rules of the world" in content):
            return ['```json\n{"reasoning": "Mock reasoning", "world_rules": "Mock rules"}\n```']
        if "[[ ## spine_template ## ]]" in content or ('"spine_template"' in content and "Once upon a time" in content):
            return ['```json\n{"spine_template": "Mock spine"}\n```']
        if "[[ ## core_premise ## ]]" in content or ('"core_premise"' in content and "summarizing the foundation" in content):
            return ['```json\n{"core_premise": "Mock premise"}\n```']
        if "questions_with_answers" in content:
            return ['```json\n{"questions_with_answers": [{"question": "Mock?", "proposed_answer": "Yes"}]}\n```']

        # Fallback
        if "core_premise" in content and "spine_template" not in content:
            return ['```json\n{"core_premise": "Mock premise"}\n```']
        elif "spine_template" in content and "world_bible" not in content:
            return ['```json\n{"spine_template": "Mock spine"}\n```']
        elif "world_bible" in content and "arc_outline" not in content:
            return ['```json\n{"world_bible": "Mock world bible"}\n```']
        elif "arc_outline" in content and "chapter_plan" not in content:
            return ['```json\n{"reasoning": "Mock reasoning", "arc_outline": "Mock arc outline"}\n```']
        elif "chapter_plan" in content and "enhancers_guide" not in content and "story" not in content:
            return ['```json\n{"reasoning": "Mock reasoning", "chapter_plan": "- **Chapter 1**: Mock chapter one"}\n```']
        elif "enhancers_guide" in content and "story" not in content and "chapter_text" not in content:
            return ['```json\n{"reasoning": "Mock reasoning", "enhancers_guide": "Mock enhancers guide"}\n```']
        elif "chapter_text" in content:
            return ['```json\n{"reasoning": "Mock reasoning", "title": "Mock Title", "chapter_text": "Mock chapter text"}\n```']
        elif "story" in content:
            return ['```json\n{"story": "Mock final story"}\n```']
        return ["Mock response"]

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

IDEA = (
    "An unnamed child is raised by the Church as the ultimate weapon against demons. "
    "As child grows he learns that the church itself is corrupt and breeds demons for "
    "controlled chaos. The church recieves funding for protection and as such decides "
    "who should recieve help. The child eventually becomes overpowered and turns back "
    "on the Church"
)


@pytest.fixture(autouse=True)
def _configure_mock_lm():
    """Configure DSPy with MockLM for every test automatically."""
    lm = MockLM()
    dspy.configure(lm=lm)
    yield


def test_extract_chapter_descriptions():
    chapter_plan = """**Act 1: Beginning**
- **Chapter 1**: The hero is trained by the Church.
- **Chapter 2**: The hero visits the Breeding Complexes.

**Act 2: Revelation**
- **Chapter 3**: Forbidden texts reveal the truth.
"""
    assert extract_chapter_descriptions(chapter_plan) == [
        "The hero is trained by the Church.",
        "The hero visits the Breeding Complexes.",
        "Forbidden texts reveal the truth.",
    ]


def test_question_generation():
    q_gen = QuestionGenerator()
    q_result = q_gen(idea=IDEA)
    assert len(q_result.questions_with_answers) >= 1
    assert q_result.questions_with_answers[0].question


def test_core_premise():
    cp_gen = CorePremiseGenerator()
    cp_result = cp_gen(idea=IDEA, qa_pairs="Q: Mock?\nA: Yes")
    assert cp_result.core_premise


def test_spine_template():
    st_gen = SpineTemplateGenerator()
    st_result = st_gen(core_premise="Mock premise")
    assert st_result.spine_template


def test_story_generation():
    story_gen = StoryGenerator()
    result = story_gen(
        core_premise="Mock premise",
        spine_template="Mock spine",
        world_bible="Mock world bible",
    )
    assert result.arc_outline
    assert result.chapter_plan
    assert result.enhancers_guide
    assert result.story


def test_character_visual_describer():
    cv_describer = CharacterVisualDescriber()
    cv_result = cv_describer(world_bible="Mock world bible")
    assert len(cv_result.character_visuals) >= 1
    cv = cv_result.character_visuals[0]
    assert cv.name
    assert cv.reference_mix
    assert cv.full_prompt


def test_scene_image_prompt_generator():
    scene_gen = SceneImagePromptGenerator()
    result = scene_gen(
        chapter_text="Mock chapter text",
        character_visuals_summary="- Mock Hero: a mix of Guts and Zuko",
    )
    assert result.image_prompt


def test_pipeline(
    model_name="mock",
    api_base="http://localhost:11434",
    api_key=None,
    enable_images=False,
    output_dir=".tmp",
    max_tokens=2048
):
    kwargs = {"max_tokens": max_tokens}
    if api_base:
        kwargs["api_base"] = api_base

    if api_key is not None:
        kwargs["api_key"] = api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            kwargs["api_key"] = env_key
    elif "ollama" in model_name.lower():
        pass

    logger.info(f"Testing pipeline with model: {model_name}...")

    if model_name == "mock" or model_name == "test_mock":
        lm = MockLM()
        dspy.configure(lm=lm)
    else:
        if "openai" in model_name.lower() and not kwargs.get("api_key"):
            logger.warning("OPENAI_API_KEY not found. Skipping full integration test to avoid errors.")
            return

        lm = dspy.LM(model_name, **kwargs)
        dspy.configure(lm=lm)

    idea = IDEA

    # 1. Questions
    q_gen = QuestionGenerator()
    q_result = q_gen(idea=idea)
    logger.info(f"Generated {len(q_result.questions_with_answers)} questions.")

    # Fake answers
    qa_text = ""
    for q in q_result.questions_with_answers:
        qa_text += f"Q: {q.question}\nA: {q.proposed_answer}\n\n"

    # 2. Core Premise
    cp_gen = CorePremiseGenerator()
    cp_result = cp_gen(idea=idea, qa_pairs=qa_text)
    logger.info("Core Premise generated.")

    # 3. Spine Template
    st_gen = SpineTemplateGenerator()
    st_result = st_gen(core_premise=cp_result.core_premise)
    logger.info("Spine Template generated.")

    # 4. World Bible
    wb_gen = WorldBibleGenerator()
    wb_result = wb_gen(core_premise=cp_result.core_premise, spine_template=st_result.spine_template)
    logger.info("World Bible generated.")

    cv_result = None
    character_visuals_summary = ""

    if enable_images:
        # 5. Character Visual Descriptions
        cv_describer = CharacterVisualDescriber()
        cv_result = cv_describer(world_bible=wb_result.world_bible)
        print(f"Generated visuals for {len(cv_result.character_visuals)} characters.")
        for cv in cv_result.character_visuals:
            print(f"  - {cv.name}: {cv.reference_mix}")

        character_visuals_summary = "\n".join(
            f"- {cv.name}: {cv.reference_mix}. {cv.distinguishing_features}"
            for cv in cv_result.character_visuals
        )

        # 6. Mock image generation (no real API calls)
        from image_gen import ImageGenerator

        with patch.object(ImageGenerator, "__init__", lambda self, **kw: None):
            img_gen = ImageGenerator()
            img_gen.api_token = "mock_token"
            img_gen.output_dir = MagicMock()

            with patch.object(img_gen, "generate_character_portrait", return_value="images/mock_portrait.png"):
                for cv in cv_result.character_visuals:
                    path = img_gen.generate_character_portrait(prompt=cv.full_prompt, character_name=cv.name)
                    print(f"  Mock portrait for {cv.name}: {path}")

    # 7. Story
    story_gen = StoryGenerator()
    story_result = story_gen(core_premise=cp_result.core_premise, spine_template=st_result.spine_template, world_bible=wb_result.world_bible)
    logger.info("Story generated.")

    if enable_images:
        # 8. Scene image prompts
        scene_prompt_gen = SceneImagePromptGenerator()
        prompt_result = scene_prompt_gen(
            chapter_text="Mock chapter text for testing",
            character_visuals_summary=character_visuals_summary,
        )
        logger.info(f"Scene image prompt: {prompt_result.image_prompt[:80]}...")

    logger.info("Test passed successfully!")

    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "story_output.md")
    logger.info(f"Saving story output to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("# Story Output\n\n")
        f.write("## Core Premise\n")
        f.write(f"{cp_result.core_premise}\n\n")
        f.write("## Spine Template\n")
        f.write(f"{st_result.spine_template}\n\n")
        f.write("## World Bible\n")
        f.write(f"{wb_result.world_bible}\n\n")

        if cv_result and cv_result.character_visuals:
            f.write("## Character Visuals\n\n")
            for cv in cv_result.character_visuals:
                f.write(f"### {cv.name}\n")
                f.write(f"**Reference:** {cv.reference_mix}\n\n")
                f.write(f"**Features:** {cv.distinguishing_features}\n\n")

        f.write("## Arc Outline\n")
        f.write(f"{story_result.arc_outline}\n\n")
        f.write("## Chapter Plan\n")
        f.write(f"{story_result.chapter_plan}\n\n")
        f.write("## Enhancers Guide\n")
        f.write(f"{story_result.enhancers_guide}\n\n")
        f.write("## Final Story\n")
        f.write(f"{story_result.story}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AI DSPy Story Writer")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "mock"), help="The language model to use (e.g., openai/gpt-4o-mini, ollama_chat/llama3). Defaults to MODEL env var or mock.")
    parser.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL", "http://localhost:11434"), help="The custom API base URL (e.g., http://localhost:11434 for Ollama). Defaults to LLM_URL env var or http://localhost:11434.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY"), help="The API key for the model. Defaults to API_KEY env var.")
    parser.add_argument("--enable-images", action="store_true", default=False, help="Enable the mocked image-generation path during test runs.")
    parser.add_argument("--output-dir", type=str, default=".tmp", help="Directory to save output files. Defaults to .tmp.")
    parser.add_argument("--max_tokens", type=int, default=1024, help="max number of tokens to pass to an LLM call")

    args = parser.parse_args()

    test_pipeline(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        enable_images=args.enable_images,
        output_dir=args.output_dir,
	max_tokens=args.max_tokens
    )
