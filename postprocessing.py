"""Post-processing utilities for detecting similar/duplicate sentences in generated stories.

The core trick: split text into sentences, sort them alphabetically, and similar
sentences naturally cluster together — making repetition easy to spot.
"""

import re
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Sentence-ending punctuation pattern
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

# Markdown/list artifacts to strip before comparison.
# Each pattern targets a specific construct so that meaningful opening
# punctuation (quotes, parentheses, etc.) is never consumed.
_MARKDOWN_NOISE_RES = [
    re.compile(r'^\s*#{1,6}\s+'),   # headings:  ## Foo
    re.compile(r'^\s*[-*]\s+'),     # unordered list markers:  - Foo / * Foo
    re.compile(r'^\s*\d+\.\s+'),    # ordered list markers:  1. Foo
    re.compile(r'^\s*>\s*'),        # blockquotes:  > Foo
    re.compile(r'^\s*[_*]{1,3}(?=[A-Za-z"\'])'),  # leading emphasis: _Foo / **Foo
]


def _normalize(sentence: str) -> str:
    """Lowercase, strip markdown noise and extra whitespace for comparison.

    Meaningful leading punctuation (opening quotes, parentheses, etc.) is
    preserved so that similarity matching remains faithful to the prose.
    """
    s = sentence
    for pat in _MARKDOWN_NOISE_RES:
        s = pat.sub('', s)
    s = s.strip()
    return ' '.join(s.lower().split())


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering out short fragments and headings."""
    raw = _SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Skip very short fragments (headings, labels, etc.)
        if len(s.split()) < 4:
            continue
        sentences.append(s)
    return sentences


def find_similar_sentences(
    text: str,
    threshold: float = 0.65,
) -> list[tuple[str, str, float]]:
    """Find similar sentence pairs using alphabetical-sort clustering.

    Steps:
        1. Extract and normalize sentences.
        2. Sort alphabetically — similar sentences cluster together.
        3. Compare adjacent pairs with SequenceMatcher.
        4. Return pairs above the similarity threshold.

    Args:
        text: The story text to analyze.
        threshold: Minimum similarity ratio (0-1) to flag a pair. Default 0.65.

    Returns:
        List of (sentence_a, sentence_b, similarity_score) tuples, sorted by
        score descending.
    """
    sentences = extract_sentences(text)
    if len(sentences) < 2:
        return []

    # Build (normalized, original) pairs and sort by normalized form
    pairs = [(_normalize(s), s) for s in sentences]
    pairs.sort(key=lambda p: p[0])

    similar: list[tuple[str, str, float]] = []

    for i in range(len(pairs) - 1):
        norm_a, orig_a = pairs[i]
        norm_b, orig_b = pairs[i + 1]

        # Skip exact duplicates of normalization (same sentence repeated)
        if norm_a == norm_b:
            similar.append((orig_a, orig_b, 1.0))
            continue

        # Quick prefix check: skip if first 3 chars don't match
        # (alphabetical sort means similar sentences share prefixes)
        if norm_a[:3] != norm_b[:3]:
            continue

        ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
        if ratio >= threshold:
            similar.append((orig_a, orig_b, ratio))

    similar.sort(key=lambda x: x[2], reverse=True)
    return similar


def format_report(similar_pairs: list[tuple[str, str, float]]) -> str:
    """Format similar sentence pairs into a readable report."""
    if not similar_pairs:
        return "No similar sentences found."

    lines = [f"Found {len(similar_pairs)} similar sentence pair(s):\n"]
    for i, (a, b, score) in enumerate(similar_pairs, 1):
        lines.append(f"--- Pair {i} (similarity: {score:.0%}) ---")
        lines.append(f"  A: {a[:120]}{'...' if len(a) > 120 else ''}")
        lines.append(f"  B: {b[:120]}{'...' if len(b) > 120 else ''}")
        lines.append("")

    return '\n'.join(lines)
