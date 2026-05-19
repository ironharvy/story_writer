#!/usr/bin/env python3
"""Count per-word frequency in the Final Story section of a markdown file."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


FINAL_STORY_HEADER = "## Final Story"
HEADING_RE = re.compile(r"^##\s+")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "about",
    "after",
    "again",
    "all",
    "also",
    "always",
    "any",
    "around",
    "away",
    "back",
    "at",
    "be",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "done",
    "down",
    "each",
    "else",
    "even",
    "every",
    "few",
    "find",
    "first",
    "from",
    "get",
    "going",
    "got",
    "here",
    "how",
    "just",
    "know",
    "look",
    "looked",
    "made",
    "make",
    "many",
    "may",
    "might",
    "more",
    "most",
    "much",
    "must",
    "need",
    "never",
    "new",
    "next",
    "now",
    "off",
    "often",
    "old",
    "once",
    "one",
    "only",
    "other",
    "out",
    "over",
    "really",
    "right",
    "same",
    "see",
    "seen",
    "should",
    "since",
    "so",
    "some",
    "still",
    "such",
    "than",
    "then",
    "there",
    "these",
    "they're",
    "thing",
    "things",
    "through",
    "too",
    "under",
    "up",
    "very",
    "want",
    "wasn't",
    "way",
    "well",
    "went",
    "were",
    "we're",
    "what's",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "without",
    "won't",
    "would",
    "wouldn't",
    "yeah",
    "yes",
    "yet",
    "you'd",
    "you're",
    "you've",
    "yourself",
    "for",
    "from",
    "not",
    "what",
    "like",
    "said",
    "them",
    "had",
    "has",
    "have",
    "he",
    "we",
    "into",
    "who",
    "her",
    "his",
    "him",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "she",
    "our",
    "ours",
    "us",
    "you",
    "your",
}


def extract_final_story(text: str) -> str:
    """Extract the Final Story section from markdown output.

    If the section is not found, return the full text as a fallback.
    """
    lines = text.splitlines()

    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == FINAL_STORY_HEADER:
            start_index = index + 1
            break

    if start_index is None:
        return text

    end_index = len(lines)
    for index in range(start_index, len(lines)):
        if HEADING_RE.match(lines[index]):
            end_index = index
            break

    return "\n".join(lines[start_index:end_index]).strip()


def normalize_word(token: str) -> str:
    """Normalize a token for counting."""
    word = token.lower().strip("'")
    if word.endswith("'s") and len(word) > 2:
        word = word[:-2]
    return word


def count_words(
    file_path: Path,
    *,
    include_stopwords: bool = False,
    min_length: int = 4,
    min_count: int = 2,
) -> Counter[str]:
    """Return filtered per-word counts for the final story text in *file_path*."""
    text = file_path.read_text(encoding="utf-8")
    final_story = extract_final_story(text)
    words = [normalize_word(match.group(0)) for match in WORD_RE.finditer(final_story)]
    words = [word for word in words if word and not word.isdigit()]
    if not include_stopwords:
        words = [word for word in words if word not in DEFAULT_STOPWORDS]
    words = [word for word in words if len(word) >= min_length]
    counts = Counter(words)
    if min_count > 1:
        counts = Counter(
            {word: count for word, count in counts.items() if count >= min_count}
        )
    return counts


def drop_top_words(counts: Counter[str], drop_top: int) -> Counter[str]:
    """Return a copy of *counts* without the most frequent *drop_top* words."""
    if drop_top <= 0:
        return counts

    filtered = counts.copy()
    for word, _ in counts.most_common(drop_top):
        filtered.pop(word, None)
    return filtered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Count per-word frequency in the Final Story section of a markdown file.",
    )
    parser.add_argument("file", type=Path, help="Path to the file to count words in.")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Optional maximum number of words to print.",
    )
    parser.add_argument(
        "--include-stopwords",
        action="store_true",
        default=False,
        help="Include common stopwords like 'the' and 'to' in the output.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="Ignore words shorter than this length (default: 4).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Ignore words that appear fewer than this count (default: 2).",
    )
    parser.add_argument(
        "--drop-top",
        type=int,
        default=20,
        help="Drop this many highest-frequency words after filtering (default: 20).",
    )
    args = parser.parse_args(argv)

    path: Path = args.file
    if not path.is_file():
        print(f"Error: '{path}' is not a file or does not exist.", file=sys.stderr)
        sys.exit(1)

    counts = count_words(
        path,
        include_stopwords=args.include_stopwords,
        min_length=args.min_length,
        min_count=args.min_count,
    )
    counts = drop_top_words(counts, args.drop_top)
    items = counts.most_common(args.top)

    if not items:
        print("No words found.")
        return

    for word, count in items:
        print(f"{word}\t{count}")


if __name__ == "__main__":
    main()
