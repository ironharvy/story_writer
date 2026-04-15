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

    Attributes:
        idea: the initial story idea supplied up-front.
        ideation_answers: answers for :meth:`answer_questions` during the
            ideation step, in call order. Each entry is the full replacement
            ``user_answer`` string, or ``None`` to accept the proposed answer.
        world_bible_answers: same shape, used for the world-bible question
            batch.
        premise_decisions: iterable of :class:`PremiseDecision` values. The
            pipeline may loop over premise refinement; the final decision
            must be ``accept=True`` or the script will raise.
        notify_sink: optional callback receiving ``(level, message)`` — lets
            the web worker forward messages as SSE events. Defaults to a
            logger.
    """

    idea: str = ""
    ideation_answers: Sequence[Optional[str]] = field(default_factory=list)
    world_bible_answers: Sequence[Optional[str]] = field(default_factory=list)
    premise_decisions: Sequence[PremiseDecision] = field(default_factory=list)
    notify_sink: Optional[Callable[[str, str], None]] = None

    # Private iterators are populated lazily so the dataclass remains
    # trivially constructible/serializable.
    _ideation_iter: Optional[Iterator[Optional[str]]] = field(
        default=None, init=False, repr=False
    )
    _wb_iter: Optional[Iterator[Optional[str]]] = field(
        default=None, init=False, repr=False
    )
    _premise_iter: Optional[Iterator[PremiseDecision]] = field(
        default=None, init=False, repr=False
    )
    # Toggles between the two batches. The pipeline calls ``answer_questions``
    # exactly twice (ideation, then world bible), so we route by call order.
    _questions_call_count: int = field(default=0, init=False, repr=False)

    def ask_idea(self) -> str:
        if not self.idea:
            raise RuntimeError(
                "ScriptedPrompter.ask_idea called but no idea was provided."
            )
        return self.idea

    def answer_questions(self, qa: Sequence[QAPair]) -> list[QAPair]:
        # First call = ideation questions; second = world bible questions.
        if self._questions_call_count == 0:
            answers = list(self.ideation_answers)
            label = "ideation"
        elif self._questions_call_count == 1:
            answers = list(self.world_bible_answers)
            label = "world_bible"
        else:
            raise RuntimeError(
                "ScriptedPrompter.answer_questions called more than twice; "
                "the pipeline contract only includes ideation and world-bible "
                "question batches."
            )
        self._questions_call_count += 1

        if len(answers) < len(qa):
            raise RuntimeError(
                f"ScriptedPrompter has {len(answers)} {label} answers but was "
                f"asked for {len(qa)}."
            )

        answered: list[QAPair] = []
        for pair, user_answer in zip(qa, answers):
            answered.append(
                QAPair(
                    question=pair.question,
                    proposed_answer=pair.proposed_answer,
                    user_answer=user_answer,
                )
            )
        return answered

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
