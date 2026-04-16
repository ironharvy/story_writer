"""Deterministic :class:`Prompter` backed by pre-collected answers.

This is the implementation used by web workers (where the user filled out a
form up-front and the pipeline replays their answers) and by tests.

The scripted prompter never blocks and never reads from stdin.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Sequence

from engine.prompter import Prompter
from engine.types import QAPair


logger = logging.getLogger(__name__)


@dataclass
class PremiseDecision:
    """One user decision on a generated core premise.

    If ``accept`` is ``True`` the pipeline moves on. Otherwise
    ``refinement_details`` is folded back into the idea and the premise is
    regenerated.
    """

    accept: bool
    refinement_details: str = ""


@dataclass
class ScriptedPrompter(Prompter):
    """Replay a pre-collected script of answers.

    The pipeline may call :meth:`answer_questions` an arbitrary number of
    times (one per ideation round triggered by :meth:`confirm_premise`, plus
    once for the world-bible questions). We model this as an ordered queue
    of "answer batches" — each call pops the next batch.

    Attributes:
        idea: the initial story idea returned by :meth:`ask_idea`.
        answer_batches: one list per expected ``answer_questions`` call, in
            call order. Each inner list must be long enough for the question
            batch the pipeline hands in; entries are either the full
            replacement ``user_answer`` string, or ``None`` to accept the
            proposed answer.
        premise_decisions: iterable of :class:`PremiseDecision` values. The
            final decision must be ``accept=True`` or the script will raise.
        notify_sink: optional callback receiving ``(level, message)`` — lets
            the web worker forward messages as SSE events. Defaults to a
            logger.
    """

    idea: str = ""
    answer_batches: Sequence[Sequence[Optional[str]]] = field(default_factory=list)
    premise_decisions: Sequence[PremiseDecision] = field(default_factory=list)
    notify_sink: Optional[Callable[[str, str], None]] = None

    # Lazy iterators keep the dataclass trivially constructible.
    _answer_iter: Optional[Iterator[Sequence[Optional[str]]]] = field(
        default=None, init=False, repr=False
    )
    _premise_iter: Optional[Iterator[PremiseDecision]] = field(
        default=None, init=False, repr=False
    )

    def ask_idea(self) -> str:
        if not self.idea:
            raise RuntimeError(
                "ScriptedPrompter.ask_idea called but no idea was provided."
            )
        return self.idea

    def answer_questions(self, qa: Sequence[QAPair]) -> list[QAPair]:
        if self._answer_iter is None:
            self._answer_iter = iter(self.answer_batches)

        try:
            answers = list(next(self._answer_iter))
        except StopIteration as exc:
            raise RuntimeError(
                "ScriptedPrompter.answer_questions called more times than "
                "the number of answer_batches provided."
            ) from exc

        if len(answers) < len(qa):
            raise RuntimeError(
                f"ScriptedPrompter batch has {len(answers)} answers but was "
                f"asked for {len(qa)}."
            )

        return [
            QAPair(
                question=pair.question,
                proposed_answer=pair.proposed_answer,
                user_answer=user_answer,
            )
            for pair, user_answer in zip(qa, answers)
        ]

    def confirm_premise(self, premise: str) -> tuple[bool, str]:
        if self._premise_iter is None:
            self._premise_iter = iter(self.premise_decisions)
        try:
            decision = next(self._premise_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "ScriptedPrompter exhausted premise_decisions; the final "
                "entry must be accept=True."
            ) from exc
        return decision.accept, decision.refinement_details

    def wait_continue(self, label: str) -> None:
        # No-op: scripted runs never block.
        logger.debug("ScriptedPrompter.wait_continue skipped: %s", label)

    def notify(self, level: str, message: str) -> None:
        if self.notify_sink is not None:
            self.notify_sink(level, message)
            return
        # Default: downgrade to a log record so the worker stays quiet.
        log_method = getattr(logger, level.lower(), None)
        if callable(log_method):
            log_method(message)
        else:
            logger.info("[%s] %s", level, message)
