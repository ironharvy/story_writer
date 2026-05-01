from __future__ import annotations

import re
from collections import Counter

from story_writer.models.qa import RuleResult
from story_writer.models.story import Chapter, WorldBible

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

META_OPENERS = ("in this chapter", "this chapter explores", "this scene")
ASSISTANT_LEAKS = ("as an ai", "i cannot", "i can't", "i am unable")
AFFIRMATION_OPENERS = ("certainly", "sure", "here is", "here's")
COMMON_ALLOWED_NAMES = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "chapter",
    "he",
    "her",
    "his",
    "i",
    "if",
    "in",
    "into",
    "its",
    "it",
    "monday",
    "one",
    "she",
    "saturday",
    "sunday",
    "the",
    "they",
    "then",
    "thursday",
    "to",
    "tuesday",
    "when",
    "while",
    "wednesday",
    "friday",
    "choose",
    "some",
    "you",
}


def detect_chapter(
    chapter: Chapter,
    world_bible: WorldBible,
    prior_chapters: list[Chapter] | None = None,
) -> list[RuleResult]:
    prior_chapters = prior_chapters or []
    text = chapter.prose.strip()
    return [
        rule_empty(text),
        rule_meta_framing(text),
        rule_assistant_leak(text),
        rule_affirmation_opener(text),
        rule_unknown_proper_nouns(text, world_bible),
        rule_missing_world_bible_anchor(text, world_bible),
        rule_repeated_sentences(text),
        rule_cross_chapter_repetition(text, prior_chapters),
        rule_repeated_opening(text, prior_chapters),
    ]


def has_hard_failures(results: list[RuleResult]) -> bool:
    return any(not result.passed and result.severity == "hard" for result in results)


def rule_empty(text: str) -> RuleResult:
    passed = bool(text.strip())
    return RuleResult(
        rule_id="R1",
        passed=passed,
        severity="hard",
        message="Chapter prose is present." if passed else "Chapter prose is empty.",
    )


def rule_meta_framing(text: str) -> RuleResult:
    lowered = text.lstrip().lower()
    spans = [phrase for phrase in META_OPENERS if lowered.startswith(phrase)]
    return RuleResult(
        rule_id="R2",
        passed=not spans,
        severity="hard",
        message="No meta-framing opener found."
        if not spans
        else "Meta-framing opener found.",
        spans=spans,
    )


def rule_assistant_leak(text: str) -> RuleResult:
    lowered = text.lower()
    spans = [phrase for phrase in ASSISTANT_LEAKS if phrase in lowered]
    return RuleResult(
        rule_id="R3",
        passed=not spans,
        severity="hard",
        message="No assistant-language leak found."
        if not spans
        else "Assistant-language leak found.",
        spans=spans,
    )


def rule_affirmation_opener(text: str) -> RuleResult:
    lowered = text.lstrip().lower()
    spans = [phrase for phrase in AFFIRMATION_OPENERS if lowered.startswith(phrase)]
    return RuleResult(
        rule_id="R4",
        passed=not spans,
        severity="hard",
        message="No affirmation opener found."
        if not spans
        else "Affirmation opener found.",
        spans=spans,
    )


def rule_unknown_proper_nouns(text: str, world_bible: WorldBible) -> RuleResult:
    known_names = {name.lower() for name in world_bible.known_names()}
    known_parts = {
        part.lower()
        for name in world_bible.known_names()
        for part in name.split()
        if len(part) > 1
    }
    unknown: list[str] = []
    for match in PROPER_NOUN_RE.finditer(text):
        value = match.group(0)
        key = value.lower()
        if key in COMMON_ALLOWED_NAMES or key in known_names:
            continue
        if " " not in value and _is_sentence_start(text, match.start()):
            continue
        if all(part.lower() in known_parts for part in value.split()):
            continue
        if value not in unknown:
            unknown.append(value)
    return RuleResult(
        rule_id="R5",
        passed=not unknown,
        severity="hard",
        message="No unknown proper nouns found."
        if not unknown
        else "Unknown proper nouns found.",
        spans=unknown[:20],
    )


def rule_missing_world_bible_anchor(text: str, world_bible: WorldBible) -> RuleResult:
    lowered = text.lower()
    anchors = _world_bible_anchor_terms(world_bible)
    present = [anchor for anchor in anchors if anchor in lowered]
    passed = bool(present) or not anchors
    return RuleResult(
        rule_id="R6",
        passed=passed,
        severity="soft",
        message="World-bible anchor present."
        if passed
        else "No world-bible names appear in chapter.",
    )


def _world_bible_anchor_terms(world_bible: WorldBible) -> set[str]:
    terms: set[str] = set()
    for name in world_bible.known_names():
        lowered = name.lower()
        terms.add(lowered)
        terms.update(part.lower() for part in name.split() if len(part) > 2)
    return terms


def rule_repeated_sentences(text: str) -> RuleResult:
    sentences = _sentences(text)
    counts = Counter(sentence.lower() for sentence in sentences)
    repeated = [sentence for sentence, count in counts.items() if count > 1]
    return RuleResult(
        rule_id="R7",
        passed=not repeated,
        severity="soft",
        message="No repeated sentences found."
        if not repeated
        else "Repeated sentence found.",
        spans=repeated[:10],
    )


def rule_cross_chapter_repetition(
    text: str, prior_chapters: list[Chapter]
) -> RuleResult:
    current = {sentence.lower() for sentence in _sentences(text)}
    prior = {
        sentence.lower()
        for chapter in prior_chapters
        for sentence in _sentences(chapter.prose)
    }
    repeated = sorted(current & prior)
    return RuleResult(
        rule_id="R8",
        passed=not repeated,
        severity="soft",
        message="No cross-chapter repeated sentences found."
        if not repeated
        else "Cross-chapter repeated sentence found.",
        spans=repeated[:10],
    )


def rule_repeated_opening(text: str, prior_chapters: list[Chapter]) -> RuleResult:
    opening = _opening_signature(text)
    prior_openings = {
        _opening_signature(chapter.prose): chapter.number
        for chapter in prior_chapters
        if _opening_signature(chapter.prose)
    }
    repeated = opening in prior_openings
    return RuleResult(
        rule_id="R9",
        passed=not repeated,
        severity="soft",
        message="Chapter opening is distinct."
        if not repeated
        else f"Chapter opening repeats chapter {prior_openings[opening]}.",
        spans=[opening] if repeated and opening else [],
    )


def _sentences(text: str) -> list[str]:
    return [
        sentence.strip() for sentence in SENTENCE_RE.split(text) if sentence.strip()
    ]


def _opening_signature(text: str) -> str:
    sentences = _sentences(text)
    if not sentences:
        return ""
    words = re.findall(r"[a-z0-9']+", sentences[0].lower())
    return " ".join(words[:16])


def _is_sentence_start(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    return not prefix or prefix.endswith((".", "!", "?", "\n"))
