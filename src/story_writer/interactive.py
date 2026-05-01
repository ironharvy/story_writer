from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.prompt import Confirm, Prompt

from story_writer.models.story import Clarification

console = Console()


def answer_questions(
    questions: list[Clarification],
    *,
    non_interactive: bool,
) -> list[Clarification]:
    answered: list[Clarification] = []
    for item in questions:
        if non_interactive:
            answered.append(item.model_copy(update={"answer": item.suggested_answer}))
            continue
        console.print(f"\n[bold]{item.question}[/bold]")
        answer = Prompt.ask("Answer", default=item.suggested_answer)
        answered.append(item.model_copy(update={"answer": answer}))
    return answered


def review_artifact(
    title: str,
    rendered: str,
    regenerate: Callable[[str], object],
    *,
    non_interactive: bool,
) -> object | None:
    if non_interactive:
        return None
    console.rule(title)
    console.print(rendered)
    while True:
        action = Prompt.ask(
            "Accept, edit feedback, or regenerate?",
            choices=["accept", "edit", "regenerate"],
            default="accept",
        )
        if action == "accept":
            return None
        feedback = ""
        if action == "edit":
            feedback = Prompt.ask("Feedback for regeneration")
        if action == "regenerate" or feedback:
            candidate = regenerate(feedback)
            console.rule(f"{title} regenerated")
            console.print(str(candidate))
            if Confirm.ask("Accept this version?", default=True):
                return candidate
