"""Human-input abstraction for the story generation pipeline.

The pipeline must work both interactively (a person at a TTY answering
questions) and headlessly (a web worker replaying pre-collected answers from a
database). Both modes implement the same :class:`Prompter` protocol; pipeline
code never branches on transport.

Concrete implementations live alongside this module:
- ``engine.prompter_cli``      — rich/terminal prompts (replacing the
  ``rich.Prompt`` / ``Confirm`` calls in ``main.py``).
- ``engine.prompter_scripted`` — deterministic, in-memory answers for workers
  and tests.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from engine.types import QAPair


@runtime_checkable
class Prompter(Protocol):
    """Contract for supplying human input to pipeline stages.

    Pipeline code calls methods on a ``Prompter`` wherever it needs a decision
    from the user. Implementations decide whether that means blocking on a TTY,
    looking up a stored answer, or returning a canned value in a test.
    """

    def ask_idea(self) -> str:
        """Return the user's initial story idea/prompt."""
        ...

    def answer_questions(self, qa: Sequence[QAPair]) -> list[QAPair]:
        """Given a list of questions with proposed answers, return the list
        with ``user_answer`` populated on each entry.

        Implementations may mutate the inputs or return new instances — callers
        should use the returned list.
        """
        ...

    def confirm_premise(self, premise: str) -> tuple[bool, str]:
        """Show the user the generated core premise and ask whether to accept.

        Returns:
            ``(accept, refinement_details)``. When ``accept`` is ``True`` the
            ``refinement_details`` value is ignored. When ``False``, the
            details are folded back into the idea and the premise is
            regenerated.
        """
        ...

    def wait_continue(self, label: str) -> None:
        """Optional checkpoint between stages.

        CLI implementations typically block on "Press Enter to continue";
        scripted implementations no-op.
        """
        ...

    def notify(self, level: str, message: str) -> None:
        """Surface a progress / status message.

        ``level`` is a free-form hint (e.g. ``"info"``, ``"warning"``,
        ``"error"``, ``"success"``). Implementations map it to whatever is
        appropriate — rich color for CLI, an SSE event for web workers, a log
        line for tests.
        """
        ...
