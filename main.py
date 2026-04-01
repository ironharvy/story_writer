import dspy
from rich.console import Console
from rich.prompt import Prompt, Confirm
from story_modules import (
    QuestionGenerator,
    CorePremiseGenerator,
    SpineTemplateGenerator,
    StoryGenerator,
    CharacterVisualDescriber,
    SceneImagePromptGenerator,
)
from world_bible_modules import (
    WorldBibleGenerator,
    WorldBibleQuestionGenerator,
)
import os
import argparse
import logging
import coloredlogs
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

console = Console()

def configure_dspy(model_name: str, api_base: str = None, api_key: str = None, max_tokens: int = 2000):
    kwargs = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key is not None:
        kwargs["api_key"] = api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            logger.warning("OPENAI_API_KEY not found in environment variables. Assuming mock or alternative setup.")
        else:
            kwargs["api_key"] = env_key
    elif "ollama" in model_name.lower():
        pass
        #kwargs["api_key"] = "" # Ollama typically doesn't need an API key

    logger.info(f"Configuring DSPy to use model '{model_name}'...")
    lm = dspy.LM(model_name, max_tokens=max_tokens, **kwargs)

    dspy.configure(lm=lm)

def get_answers_for_questions(questions_with_answers) -> str:
    qa_pairs = []
    for i, qa in enumerate(questions_with_answers):
        console.print(f"\n[bold cyan]Question {i+1}:[/bold cyan] {qa.question}")
        console.print(f"[bold green]Proposed Answer:[/bold green] {qa.proposed_answer}")

        accept = Confirm.ask("Accept this proposed answer?")
        if accept:
            user_answer = qa.proposed_answer
        else:
            user_answer = Prompt.ask("Enter your answer")

        qa_pairs.append(f"Q: {qa.question}\nA: {user_answer}")

    return "\n\n".join(qa_pairs)

def main():
    parser = argparse.ArgumentParser(description="AI DSPy Story Writer")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "openai/gpt-4o-mini"), help="The language model to use (e.g., openai/gpt-4o-mini, ollama_chat/llama3). Defaults to MODEL env var.")
    parser.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL"), help="The custom API base URL (e.g., http://localhost:11434 for Ollama). Defaults to LLM_URL env var.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY"), help="The API key for the model. Defaults to API_KEY env var.")
    parser.add_argument("--max-tokens", type=int, default=8000, help="The maximum number of tokens to use for the model. Defaults to 8000.")
    parser.add_argument("--enable-images", action="store_true", default=False, help="Enable image generation (requires Replicate API token).")
    parser.add_argument("--replicate-api-token", type=str, default=os.environ.get("REPLICATE_API_TOKEN"), help="Replicate API token. Defaults to REPLICATE_API_TOKEN env var.")

    args = parser.parse_args()

    configure_dspy(model_name=args.model, api_base=args.llm_url, api_key=args.api_key, max_tokens=args.max_tokens)

    console.print("[bold magenta]Welcome to the AI DSPy Story Writer![/bold magenta]")

    # 1. Prompt for initial idea
    idea = Prompt.ask("\n[bold yellow]What is your initial story idea/prompt?[/bold yellow]")

    # Initialize generators
    q_gen = QuestionGenerator()
    cp_gen = CorePremiseGenerator()
    st_gen = SpineTemplateGenerator()
    wb_gen = WorldBibleGenerator()
    story_gen = StoryGenerator()

    core_premise = ""
    while True:
        # 2. Ask questions
        console.print("\n[italic]Generating questions to interrogate your idea...[/italic]")
        q_result = q_gen(idea=idea)

        qa_text = get_answers_for_questions(q_result.questions_with_answers)

        # 3. Generate Core Premise
        console.print("\n[italic]Generating Core Premise...[/italic]")
        cp_result = cp_gen(idea=idea, qa_pairs=qa_text)
        core_premise = cp_result.core_premise

        console.print("\n[bold magenta]--- Core Premise ---[/bold magenta]")
        console.print(core_premise)
        console.print("[bold magenta]--------------------[/bold magenta]")

        refine = Confirm.ask("Do you want to refine this premise? (Choosing 'Yes' will let you provide more details and regenerate, 'No' proceeds)", default=False)
        if refine:
            refinement_details = Prompt.ask("Provide more details or changes")
            idea = f"Original idea: {idea}\nRefinements: {refinement_details}\nCurrent Core Premise: {core_premise}"
        else:
            break

    # 4. Generate Spine Template
    console.print("\n[italic]Generating Spine Template...[/italic]")
    st_result = st_gen(core_premise=core_premise)
    spine_template = st_result.spine_template

    console.print("\n[bold blue]--- Spine Template ---[/bold blue]")
    console.print(spine_template)
    console.print("[bold blue]--------------------[/bold blue]")

    Confirm.ask("Press Enter to continue to World Bible generation...", default=True, show_default=False)

    # 5. Ask World Bible Questions
    console.print("\n[italic]Generating follow-up questions to help flesh out the World Bible...[/italic]")
    wb_q_gen = WorldBibleQuestionGenerator()
    wb_q_result = wb_q_gen(core_premise=core_premise, spine_template=spine_template)

    wb_qa_text = get_answers_for_questions(wb_q_result.questions_with_answers)

    # 6. Generate World Bible
    console.print("\n[italic]Generating World Bible...[/italic]")
    wb_result = wb_gen(core_premise=core_premise, spine_template=spine_template, user_additions=wb_qa_text)
    world_bible = wb_result.world_bible

    console.print("\n[bold green]--- World Bible ---[/bold green]")
    console.print(world_bible)
    console.print("[bold green]-------------------[/bold green]")

    # 7. Character visuals & portrait generation (if images enabled)
    character_visuals = []
    character_portrait_paths = {}
    character_visuals_summary = ""
    image_gen = None

    if args.enable_images:
        if not args.replicate_api_token:
            console.print("[bold red]Error: --enable-images requires a Replicate API token. "
                          "Set REPLICATE_API_TOKEN env var or pass --replicate-api-token.[/bold red]")
            return

        try:
            from image_gen import ImageGenerator
        except ImportError as exc:
            console.print(
                "[bold red]Error: image generation dependencies are not installed. "
                "Install the optional image requirements and try again.[/bold red]"
            )
            logger.exception("Unable to import image generation dependencies.")
            raise SystemExit(1) from exc

        image_gen = ImageGenerator(api_token=args.replicate_api_token)

        console.print("\n[italic]Generating character visual descriptions...[/italic]")
        cv_describer = CharacterVisualDescriber()
        cv_result = cv_describer(world_bible=world_bible)
        character_visuals = cv_result.character_visuals

        lines = []
        for cv in character_visuals:
            console.print(f"\n[bold cyan]{cv.name}:[/bold cyan] {cv.reference_mix}")
            console.print(f"  [dim]{cv.distinguishing_features}[/dim]")
            lines.append(
                f"- {cv.name}: {cv.reference_mix}. {cv.distinguishing_features}"
            )
        character_visuals_summary = "\n".join(lines)

        console.print("\n[italic]Generating character portraits...[/italic]")
        for cv in character_visuals:
            try:
                path = image_gen.generate_character_portrait(
                    prompt=cv.full_prompt, character_name=cv.name
                )
                character_portrait_paths[cv.name] = path
                console.print(f"  [green]Saved portrait:[/green] {path}")
            except Exception as e:
                console.print(f"  [red]Failed to generate portrait for {cv.name}: {e}[/red]")

    Confirm.ask("Press Enter to continue to Story generation...", default=True, show_default=False)

    # 8. Generate Story
    console.print("\n[italic]Generating Story (Arc Outline, Chapter Plan, Final Story)...[/italic]")
    story_result = story_gen(core_premise=core_premise, spine_template=spine_template, world_bible=world_bible)

    console.print("\n[bold red]--- Level 1: Arc Outline ---[/bold red]")
    console.print(story_result.arc_outline)

    console.print("\n[bold red]--- Level 2: Chapter Plan ---[/bold red]")
    console.print(story_result.chapter_plan)

    console.print("\n[bold red]--- Enhancers Guide ---[/bold red]")
    console.print(story_result.enhancers_guide)

    console.print("\n[bold red]--- Final Story ---[/bold red]")
    console.print(story_result.story)

    # 9. Generate scene illustrations for each chapter (if images enabled)
    scene_image_paths = {}
    if args.enable_images and image_gen:
        console.print("\n[italic]Generating scene illustrations for each chapter...[/italic]")
        scene_prompt_gen = SceneImagePromptGenerator()

        chapters = story_result.story.split("### Chapter ")
        chapters = [c for c in chapters if c.strip()]

        primary_reference_path = next(iter(character_portrait_paths.values()), None)

        for i, chapter_text in enumerate(chapters, start=1):
            try:
                prompt_result = scene_prompt_gen(
                    chapter_text=chapter_text,
                    character_visuals_summary=character_visuals_summary,
                )
                path = image_gen.generate_scene_illustration(
                    prompt=prompt_result.image_prompt,
                    reference_image_path=primary_reference_path,
                    chapter_index=i,
                )
                scene_image_paths[i] = path
                console.print(f"  [green]Chapter {i} scene:[/green] {path}")
            except Exception as e:
                console.print(f"  [red]Failed to generate scene for chapter {i}: {e}[/red]")

    # 10. Save output to markdown
    output_dir = ".tmp"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "story_output.md")
    logger.info(f"Saving story output to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("# Story Output\n\n")
        f.write("## Core Premise\n")
        f.write(f"{core_premise}\n\n")
        f.write("## Spine Template\n")
        f.write(f"{spine_template}\n\n")
        f.write("## World Bible\n")
        f.write(f"{world_bible}\n\n")

        if character_visuals:
            f.write("## Character Visuals\n\n")
            for cv in character_visuals:
                f.write(f"### {cv.name}\n")
                f.write(f"**Reference:** {cv.reference_mix}\n\n")
                f.write(f"**Features:** {cv.distinguishing_features}\n\n")
                portrait = character_portrait_paths.get(cv.name)
                if portrait:
                    f.write(f"![{cv.name} portrait]({portrait})\n\n")

        f.write("## Arc Outline\n")
        f.write(f"{story_result.arc_outline}\n\n")
        f.write("## Chapter Plan\n")
        f.write(f"{story_result.chapter_plan}\n\n")
        f.write("## Enhancers Guide\n")
        f.write(f"{story_result.enhancers_guide}\n\n")
        f.write("## Final Story\n")

        if scene_image_paths:
            chapters = story_result.story.split("### Chapter ")
            chapters = [c for c in chapters if c.strip()]
            for i, chapter_text in enumerate(chapters, start=1):
                f.write(f"\n\n### Chapter {chapter_text}")
                scene = scene_image_paths.get(i)
                if scene:
                    f.write(f"\n\n![Chapter {i} scene]({scene})\n")
        else:
            f.write(f"{story_result.story}\n")

    console.print(f"\n[bold magenta]Story generation complete! Results saved to {output_filename}[/bold magenta]")

if __name__ == "__main__":
    main()
