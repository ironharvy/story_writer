import dspy
from rich.console import Console
from rich.prompt import Prompt, Confirm
from story_modules import (
    QuestionGenerator,
    CorePremiseGenerator,
    SpineTemplateGenerator,
    WorldBibleGenerator,
    WorldBibleQuestionGenerator,
    PlotGenerator
)
import os
from dotenv import load_dotenv

load_dotenv()

console = Console()

def configure_dspy():
    # Attempt to load OpenAI key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[yellow]Warning: OPENAI_API_KEY not found in environment variables. Assuming mock or alternative setup.[/yellow]")

    # Configure DSPy to use a language model (e.g., GPT-3.5 or GPT-4)
    # For testing, we might want to allow this to be overridden
    lm = dspy.LM('openai/gpt-4o-mini', max_tokens=2000)
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
    configure_dspy()
    console.print("[bold magenta]Welcome to the AI DSPy Story Writer![/bold magenta]")

    # 1. Prompt for initial idea
    idea = Prompt.ask("\n[bold yellow]What is your initial story idea/prompt?[/bold yellow]")

    # Initialize generators
    q_gen = QuestionGenerator()
    cp_gen = CorePremiseGenerator()
    st_gen = SpineTemplateGenerator()
    wb_gen = WorldBibleGenerator()
    plot_gen = PlotGenerator()

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

    Confirm.ask("Press Enter to continue to Plot generation...", default=True, show_default=False)

    # 6. Generate Plot
    console.print("\n[italic]Generating Plot (Arc Outline, Chapter Plan, Scenes)...[/italic]")
    plot_result = plot_gen(core_premise=core_premise, spine_template=spine_template, world_bible=world_bible)

    console.print("\n[bold red]--- Level 1: Arc Outline ---[/bold red]")
    console.print(plot_result.arc_outline)

    console.print("\n[bold red]--- Level 2: Chapter Plan ---[/bold red]")
    console.print(plot_result.chapter_plan)

    console.print("\n[bold red]--- Level 3: Scenes ---[/bold red]")
    console.print(plot_result.scenes)

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
        f.write("## Plot: Level 1 - Arc Outline\n")
        f.write(f"{plot_result.arc_outline}\n\n")
        f.write("## Plot: Level 2 - Chapter Plan\n")
        f.write(f"{plot_result.chapter_plan}\n\n")
        f.write("## Plot: Level 3 - Scenes\n")
        f.write(f"{plot_result.scenes}\n")

    console.print(f"\n[bold magenta]Story generation complete! Results saved to {output_filename}[/bold magenta]")

if __name__ == "__main__":
    main()
