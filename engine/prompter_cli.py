"""Terminal/``rich``-based :class:`Prompter` implementation.

This wraps the exact interaction pattern used by ``main.py`` today so the CLI
behavior is unchanged after the refactor.
"""

from __future__ import annotations

from typing import Sequence

from rich.console import Console
from rich.prompt import Confirm, Prompt

from engine.prompter import Prompter
from engine.types import QAPair


_LEVEL_STYLES = {
    "info": "italic",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "header": "bold magenta",
}


class CLIPrompter(Prompter):
    """Interactive prompter backed by a ``rich.Console``.

    A single console instance is used for all output so the CLI looks
    consistent with prior behavior.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def ask_idea(self) -> str:
        return Prompt.ask(
            "\n[bold yellow]What is your initial story idea/prompt?[/bold yellow]"
        )

    def answer_questions(self, qa: Sequence[QAPair]) -> list[QAPair]:
        answered: list[QAPair] = []
        for i, pair in enumerate(qa):
            self.console.print(
                f"\n[bold cyan]Question {i + 1}:[/bold cyan] {pair.question}"
            )
            self.console.print(
                f"[bold green]Proposed Answer:[/bold green] {pair.proposed_answer}"
            )

            accept = Confirm.ask("Accept this proposed answer?")
            if accept:
                # ``None`` means "use the proposed answer"; we keep it explicit
                # via ``effective_answer`` on downstream reads.
                user_answer = None
            else:
                user_answer = Prompt.ask("Enter your answer")

            answered.append(
                QAPair(
                    question=pair.question,
                    proposed_answer=pair.proposed_answer,
                    user_answer=user_answer,
                )
            )
        return answered

    def confirm_premise(self, premise: str) -> tuple[bool, str]:
        self.console.print("\n[bold magenta]--- Core Premise ---[/bold magenta]")
        self.console.print(premise)
        self.console.print("[bold magenta]--------------------[/bold magenta]")

        refine = Confirm.ask(
            "Do you want to refine this premise? "
            "(Choosing 'Yes' will let you provide more details and regenerate, "
            "'No' proceeds)",
            default=False,
        )
        if not refine:
            return True, ""

        details = Prompt.ask("Provide more details or changes")
        return False, details

    def wait_continue(self, label: str) -> None:
        # Matches the legacy ``Confirm.ask("Press Enter to continue ...", default=True)``
        # calls in ``main.py``.
        Confirm.ask(label, default=True, show_default=False)

    def notify(self, level: str, message: str) -> None:
        style = _LEVEL_STYLES.get(level.lower(), "italic")
        self.console.print(f"[{style}]{message}[/{style}]")
