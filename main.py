import dspy
from rich.console import Console
from rich.prompt import Prompt, Confirm
from story_modules import (
    QuestionGenerator,
    CorePremiseGenerator,
    SpineTemplateGenerator,
    StoryGenerator
)
from world_bible_modules import (
    WorldBibleGenerator,
    WorldBibleQuestionGenerator
)
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

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
            console.print("[yellow]Warning: OPENAI_API_KEY not found in environment variables. Assuming mock or alternative setup.[/yellow]")
        else:
            kwargs["api_key"] = env_key
    elif "ollama" in model_name.lower():
        pass
        #kwargs["api_key"] = "" # Ollama typically doesn't need an API key

    console.print(f"[italic]Configuring DSPy to use model '{model_name}'...[/italic]")
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

    Confirm.ask("Press Enter to continue to Story generation...", default=True, show_default=False)

    # 6. Generate Story
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

    # Save the output to a markdown file
    output_filename = "story_output.md"
    console.print(f"\n[italic]Saving story output to {output_filename}...[/italic]")
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("# Story Output\n\n")
        f.write("## Core Premise\n")
        f.write(f"{core_premise}\n\n")
        f.write("## Spine Template\n")
        f.write(f"{spine_template}\n\n")
        f.write("## World Bible\n")
        f.write(f"{world_bible}\n\n")
        f.write("## Arc Outline\n")
        f.write(f"{story_result.arc_outline}\n\n")
        f.write("## Chapter Plan\n")
        f.write(f"{story_result.chapter_plan}\n\n")
        f.write("## Enhancers Guide\n")
        f.write(f"{story_result.enhancers_guide}\n\n")
        f.write("## Final Story\n")
        f.write(f"{story_result.story}\n")

    console.print(f"\n[bold magenta]Story generation complete! Results saved to {output_filename}[/bold magenta]")

if __name__ == "__main__":
    main()
