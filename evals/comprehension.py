"""Reading-comprehension self-review gate.

The idea: a chapter is only doing its job if a reader who has *only this
chapter* in front of them can answer the questions the chapter is supposed to
answer — who got introduced, what they wanted, where it happened, and (for a
hard-magic premise) what the magic cost. So we:

1. Derive those questions from the chapter's plan beats + the world bible
   (``generate_questions``), tagging each as essential or supporting.
2. Hand a fresh reader *only the prose* and ask it to answer each question
   from the text alone, marking anything the prose doesn't actually support as
   ``missing`` (``answer_and_grade``).
3. Turn essential gaps into blocking findings and supporting gaps into
   warnings, with the unanswered questions as the revision brief.

This is a genuine closed-book test: the answerer is told to ignore outside
knowledge and the plan, so "the prose never says" is a real signal that the
chapter under-delivered, not a model hallucinating coverage.
"""
from __future__ import annotations

import logging

import dspy
from pydantic import BaseModel, Field

from core.types import WorldBible
from evals.types import EvalFinding, GateReport

logger = logging.getLogger(__name__)


class ComprehensionQuestion(BaseModel):
    question: str = Field(description="A concrete question this chapter should let a reader answer")
    importance: str = Field(description="essential or supporting")
    why: str = Field(description="What plot/craft need this question tests")


class GenerateComprehensionQuestions(dspy.Signature):
    """Given a chapter's plan beats and the world bible, write the 4-7
    questions a reader should be able to answer after reading ONLY this
    chapter's prose.

    Cover the concrete work this chapter must do: characters introduced (and
    named, not just referred to as 'the man'), what the viewpoint character
    wants and what blocks them, where the scene physically takes place, and —
    when magic or the world's defining mechanic is used — what it cost. Mark a
    question 'essential' when the chapter's beats clearly require it and
    'supporting' when it merely strengthens the scene. Do not ask about events
    scheduled for other chapters.

    Ask about SUBSTANCE the prose must convey, never about exact proper nouns
    from the bible. A location vividly rendered counts even if the chapter
    never repeats its formal bible name; ask "where does the scene take place
    and is it portrayed consistently with the bible?", not "is the location
    called <BibleName>?". Likewise ask whether a character is clearly
    introduced, not whether a specific surname appears.
    """

    chapter_title: str = dspy.InputField()
    chapter_beats: str = dspy.InputField()
    rules_of_the_world: list[str] = dspy.InputField()
    story_so_far: str = dspy.InputField(desc="Summary of prior chapters, for context only")
    questions: list[ComprehensionQuestion] = dspy.OutputField()


class QuestionVerdict(BaseModel):
    question: str
    answer: str = Field(description="Answer drawn ONLY from the chapter prose, or 'NOT IN TEXT'")
    verdict: str = Field(description="answered, partial, or missing")
    note: str = Field(description="If partial/missing, what the prose needs to add")


class AnswerFromProseAndGrade(dspy.Signature):
    """Answer each question using ONLY the chapter prose provided. You have no
    other knowledge of this story — do not infer from outside the text.

    For each question: quote/paraphrase what the prose actually establishes. If
    the prose does not clearly support an answer, set answer to 'NOT IN TEXT'
    and verdict to 'missing'. Use 'partial' when the prose gestures at it but
    leaves it vague (e.g. a character acts but is never named; magic is cast
    but no cost is shown). Use 'answered' only when the text plainly delivers
    it. Be strict: a reader's reasonable inference is not the same as the text
    stating it.

    But judge substance over labels: if a question asks "where" and the prose
    paints the place vividly, that is 'answered' even if the formal place-name
    is absent; if it asks who a character is and the prose characterizes them
    clearly, that is 'answered' even without a surname. Do not mark 'missing'
    merely because an exact proper noun from the bible is not repeated.
    """

    chapter_prose: str = dspy.InputField()
    questions: list[str] = dspy.InputField()
    verdicts: list[QuestionVerdict] = dspy.OutputField()


def generate_questions(
    chapter_title: str,
    chapter_beats: str,
    world_bible: WorldBible,
    story_so_far: str,
) -> list[ComprehensionQuestion]:
    """Isolated model call (question generation). Tests monkeypatch this."""
    gen = dspy.ChainOfThought(GenerateComprehensionQuestions)
    result = gen(
        chapter_title=chapter_title,
        chapter_beats=chapter_beats,
        rules_of_the_world=world_bible.rules_of_the_world,
        story_so_far=story_so_far or "(this is the first chapter)",
    )
    return list(result.questions or [])


def answer_and_grade(prose: str, questions: list[str]) -> list[QuestionVerdict]:
    """Isolated model call (closed-book answering + grading). Tests monkeypatch."""
    grader = dspy.ChainOfThought(AnswerFromProseAndGrade)
    result = grader(chapter_prose=prose, questions=questions)
    return list(result.verdicts or [])


def check(
    chapter_title: str,
    chapter_beats: str,
    world_bible: WorldBible,
    prose: str,
    story_so_far: str = "",
) -> GateReport:
    """Run the comprehension self-review over one chapter."""
    report = GateReport(gate="comprehension")
    try:
        questions = generate_questions(chapter_title, chapter_beats, world_bible, story_so_far)
        if not questions:
            return report
        importance = {q.question.strip(): q.importance.strip().lower() for q in questions}
        verdicts = answer_and_grade(prose, [q.question for q in questions])
    except Exception as exc:
        logger.warning("comprehension gate failed for %r: %s", chapter_title, exc)
        return report

    for v in verdicts:
        verdict = v.verdict.strip().lower()
        if verdict == "answered":
            continue
        essential = importance.get(v.question.strip(), "supporting") == "essential"
        if verdict == "missing":
            severity = "fail" if essential else "warn"
        else:  # partial
            severity = "warn"
        report.findings.append(
            EvalFinding(
                check="comprehension",
                severity=severity,
                message=f"{verdict.upper()}: {v.question} — {v.answer}",
                fix=v.note or f"Make the chapter clearly establish: {v.question}",
                locus=chapter_title,
            )
        )
    logger.info(
        "comprehension %r: %d/%d questions unmet (%d blocking)",
        chapter_title, len(report.findings), len(questions), len(report.blocking),
    )
    return report
