"""Rich-based interactive UI layer for the Story Writer CLI.

All console.print / Prompt.ask / Confirm.ask calls live here.
Pipeline logic must not import Rich directly.
"""

from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.prompt import Confirm, Prompt

console = Console()


def print_welcome() -> None:
    """Print the CLI welcome banner."""
    console.print("[bold magenta]Welcome to the AI DSPy Story Writer![/bold magenta]")


def ask_idea() -> str:
    """Prompt the user for their initial story idea."""
    return Prompt.ask(
        "\n[bold yellow]What is your initial story idea/prompt?[/bold yellow]"
    )


def print_status(message: str) -> None:
    """Print an italic status/progress message."""
    console.print(f"\n[italic]{message}[/italic]")


def print_section(title: str, content: str, color: str = "magenta") -> None:
    """Print a titled section with colored delimiters."""
    console.print(f"\n[bold {color}]--- {title} ---[/bold {color}]")
    console.print(content)
    console.print(f"[bold {color}]{'-' * (len(title) + 8)}[/bold {color}]")


def ask_refine(subject: str) -> tuple[bool, str]:
    """Ask whether to refine a generated artifact and collect details.

    Returns:
        (should_refine, refinement_details). refinement_details is empty
        when should_refine is False.
    """
    should_refine = Confirm.ask(
        (
            f"Do you want to refine this {subject}? (Choosing 'Yes' will let you "
            "provide more details and regenerate, 'No' proceeds)"
        ),
        default=False,
    )
    if not should_refine:
        return False, ""
    refinement_details = Prompt.ask("Provide more details or changes")
    return True, refinement_details


def ask_continue() -> None:
    """Pause and wait for the user to press Enter."""
    Confirm.ask(
        "Press Enter to continue to Story generation...",
        default=True,
        show_default=False,
    )


def get_answers_for_questions(questions_with_answers: Sequence[Any]) -> str:
    """Collect accepted or user-edited answers for generated questions."""
    qa_pairs = []
    for i, qa in enumerate(questions_with_answers):
        console.print(f"\n[bold cyan]Question {i + 1}:[/bold cyan] {qa.question}")
        console.print(f"[bold green]Proposed Answer:[/bold green] {qa.proposed_answer}")

        accept = Confirm.ask("Accept this proposed answer?")
        if accept:
            user_answer = qa.proposed_answer
        else:
            user_answer = Prompt.ask("Enter your answer")

        qa_pairs.append(f"Q: {qa.question}\nA: {user_answer}")

    return "\n\n".join(qa_pairs)


def summarize_character_visuals(character_visuals: list[Any]) -> str:
    """Render character visuals to console and return a summary string."""
    lines = []
    for visual in character_visuals:
        console.print(f"\n[bold cyan]{visual.name}:[/bold cyan] {visual.reference_mix}")
        console.print(f"  [dim]{visual.distinguishing_features}[/dim]")
        lines.append(
            f"- {visual.name}: {visual.reference_mix}. {visual.distinguishing_features}"
        )
    return "\n".join(lines)


def print_world_bible_sections(world_bible: Any) -> None:
    """Display each world bible section with its own heading."""
    sections = [
        ("Rules of the World", world_bible.rules, "yellow"),
        ("Characters", world_bible.characters, "cyan"),
        ("Locations", world_bible.locations, "green"),
        ("Plot Timeline", world_bible.plot_timeline, "blue"),
    ]
    for title, content, color in sections:
        print_section(title, content, color=color)


def print_chapter_plan_entries(entries: list[str]) -> None:
    """Display numbered chapter plan entries."""
    console.print("\n[bold red]--- Chapter Plan ---[/bold red]")
    for entry in entries:
        console.print(f"  {entry}")
    console.print(f"[bold red]{'-' * 22}[/bold red]")


def ask_enhance_items(subject: str) -> bool:
    """Ask whether the user wants to enhance individual items in a section."""
    return Confirm.ask(
        f"Would you like to enhance individual {subject}?",
        default=False,
    )


def print_item_for_review(index: int, content: str, subject: str) -> None:
    """Display a single item (character or location) for user review."""
    console.print(f"\n[bold cyan]{subject.title()} {index}:[/bold cyan]")
    console.print(content)


def print_story_sections(story_result: Any) -> None:
    """Print generated story planning artifacts and final story."""
    console.print("\n[bold red]--- Chapter Plan ---[/bold red]")
    console.print(story_result.chapter_plan)
    console.print("\n[bold red]--- Enhancers Guide ---[/bold red]")
    console.print(story_result.enhancers_guide)
    console.print("\n[bold red]--- Final Story ---[/bold red]")
    console.print(story_result.story)


def print_inpainting_result(story: str, expanded: int, total: int) -> None:
    """Print chapter inpainting completion status and result."""
    console.print(
        f"[italic]Chapter inpainting complete ({expanded}/{total} "
        "chapters expanded).[/italic]"
    )
    console.print("\n[bold red]--- Inpainted Story ---[/bold red]")
    console.print(story)


def print_portrait_saved(path: str) -> None:
    """Print a saved portrait path."""
    console.print(f"  [green]Saved portrait:[/green] {path}")


def print_portrait_failed(name: str, exc: Exception) -> None:
    """Print a portrait generation failure."""
    console.print(f"  [red]Failed to generate portrait for {name}: {exc}[/red]")


def print_missing_replicate_token() -> None:
    """Print error about missing Replicate API token."""
    console.print(
        "[bold red]Error: --enable-images requires a Replicate API token. "
        "Set REPLICATE_API_TOKEN env var or pass --replicate-api-token.[/bold red]"
    )


def print_scene_saved(chapter_index: int, path: str) -> None:
    """Print a saved scene illustration path."""
    console.print(f"  [green]Chapter {chapter_index} scene:[/green] {path}")


def print_scene_failed(chapter_index: int, exc: Exception) -> None:
    """Print a scene generation failure."""
    console.print(f"  [red]Failed to generate scene for chapter {chapter_index}: {exc}[/red]")


def print_similarity_report(report: str) -> None:
    """Print the similar-sentence detection report."""
    console.print("\n[bold yellow]--- Similar Sentence Check ---[/bold yellow]")
    console.print(report)


def print_completion(output_filename: str) -> None:
    """Print final completion message for CLI run."""
    console.print(
        "\n[bold magenta]Story generation complete! "
        f"Results saved to {output_filename}[/bold magenta]"
    )
