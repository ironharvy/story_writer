"""Post-processing utilities for generated stories.

Currently provides two families of checks:

1. Similar-sentence detection (``find_similar_sentences``) — sorts sentences
   alphabetically so near-duplicates cluster together.
2. Over-used vocabulary detection (``find_overused_words``) — catches cases
   like "parchment" appearing seven times by counting lemmas and ignoring a
   world-bible allowlist plus common English stopwords.
"""

import logging
import re
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Over-used vocabulary detection
# ---------------------------------------------------------------------------

# Word tokens: start with a letter, may contain letters, apostrophes, hyphens.
_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z'\-]*\b")

# Sentences used by the lemma-lookup helper.  Less strict than
# ``extract_sentences`` so short fragments still participate.
_SENTENCE_FIND_RE = re.compile(r"(?<=[.!?])\s+")

# Function words and very common narrative verbs/nouns we never want to flag.
# Deliberately broad — the point of this list is to suppress noise, not to be
# linguistically principled.
_STOPWORDS = frozenset({
    # articles / determiners / conjunctions / prepositions
    "the", "a", "an", "and", "or", "but", "if", "then", "as", "at",
    "by", "for", "from", "in", "into", "of", "on", "onto", "to",
    "with", "without", "within", "through", "about", "over", "under",
    "up", "down", "out", "off", "so", "not", "no", "yes", "than",
    "because", "while", "until", "since", "although", "though",
    "after", "before", "between", "across", "around", "toward", "towards",
    # pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "this", "that", "these", "those",
    "who", "whom", "whose", "which", "what", "when", "where", "why", "how",
    # to be / to have / to do / modals
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "done", "doing",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    # extremely common narrative verbs
    "say", "said", "says", "saying",
    "go", "going", "gone",
    "come", "came", "coming", "comes",
    "get", "got", "getting", "gets",
    "make", "made", "making", "makes",
    "take", "took", "taken", "taking", "takes",
    "see", "saw", "seen", "seeing", "sees",
    "know", "knew", "known", "knowing", "knows",
    "think", "thought", "thinking", "thinks",
    "look", "looked", "looking", "looks",
    "want", "wanted", "wanting", "wants",
    "give", "gave", "given", "giving", "gives",
    "tell", "told", "telling", "tells",
    "feel", "felt", "feeling", "feels",
    "find", "found", "finding", "finds",
    "turn", "turned", "turning", "turns",
    "let", "letting", "lets",
    "asked", "replied", "answered", "whisper", "whispered",
    # hedges / adverbs
    "there", "here", "now", "just", "only", "still", "again", "ever", "never",
    "always", "very", "too", "also", "even", "perhaps", "maybe",
    # quantifiers
    "some", "any", "all", "every", "each", "many", "much", "most", "more",
    "less", "few", "several", "enough", "another", "other", "such",
    "one", "two", "three", "first", "last", "next",
    # extremely common generic nouns / adjectives
    "day", "night", "morning", "evening", "time", "year", "years", "hour",
    "minute", "moment", "week", "month",
    "thing", "things", "something", "nothing", "everything", "anything",
    "someone", "everyone", "anyone", "nobody",
    "man", "woman", "girl", "boy", "people", "person", "child", "children",
    "way", "ways", "place", "part", "side", "end", "beginning",
    "good", "bad", "new", "old", "same", "little", "big", "small",
    "long", "short", "high", "low", "great", "whole",
})


def lemmatize(token: str) -> str:
    """Crude deterministic stemmer.

    Not a real lemmatizer — just enough to collapse trivial plural/tense
    variants so ``parchment`` and ``parchments`` count together.  We prefer
    under-stemming to over-stemming (false merges are worse than missed
    merges for this use case).  Every rule also requires that the resulting
    stem be at least 4 characters long, which keeps short words like
    ``ring``, ``spring``, ``red``, ``bed``, ``bus``, ``pass``, ``focus``
    from being chewed up.
    """
    w = token.lower()

    # stories -> story
    if len(w) >= 6 and w.endswith("ies"):
        stem = w[:-3] + "y"
        if len(stem) >= 4:
            return stem

    # classes -> class, kisses -> kiss
    if len(w) >= 6 and w.endswith("sses"):
        return w[:-2]

    # whispering -> whisper, running -> run (via CVC doubling)
    if len(w) >= 5 and w.endswith("ing"):
        stem = w[:-3]
        if len(stem) >= 4:
            if stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            return stem

    # whispered -> whisper, stopped -> stop
    if len(w) >= 4 and w.endswith("ed"):
        stem = w[:-2]
        if len(stem) >= 4:
            if stem[-1] == stem[-2] and stem[-1] not in "aeiou":
                stem = stem[:-1]
            return stem

    # parchments -> parchment, houses -> house.  Deliberately no "-es" rule:
    # it causes more false merges than it fixes (boxes/box, parses/parse
    # need different stems; we'd rather miss merges than make wrong ones).
    if (
        len(w) >= 5
        and w.endswith("s")
        and not w.endswith("ss")
        and not w.endswith("us")
    ):
        return w[:-1]

    return w


def extract_world_bible_allowlist(world_bible: str | None) -> set[str]:
    """Collect lemmas from the world bible that must never be flagged.

    Character names, invented places, magic-system vocabulary — anything the
    author has explicitly introduced into the world — gets a pass.  Even if
    it repeats often, that is typically intentional world-building, not
    lazy prose.
    """
    if not world_bible:
        return set()
    lemmas: set[str] = set()
    for match in _WORD_RE.finditer(world_bible):
        lemmas.add(lemmatize(match.group(0)))
    return lemmas


def find_overused_words(
    text: str,
    max_occurrences: int = 3,
    allowlist: set[str] | None = None,
    min_word_length: int = 5,
) -> list[dict]:
    """Find distinctive words that appear too many times.

    Words are collapsed via :func:`lemmatize` before counting, so ``parchment``
    and ``parchments`` are tallied together.  Stopwords, allowlisted terms,
    and tokens shorter than ``min_word_length`` are ignored.

    Args:
        text: The story text to scan.
        max_occurrences: Strict upper bound.  A lemma is flagged when its
            count exceeds this number.
        allowlist: Lemmas to exclude (typically produced by
            :func:`extract_world_bible_allowlist`).
        min_word_length: Minimum surface-form length before a token is
            considered.  Filters out most remaining function words.

    Returns:
        List of ``{"lemma": str, "count": int, "tokens": list[str]}`` dicts
        sorted by count descending.  ``tokens`` preserves the original-form
        occurrences in reading order.
    """
    allow = allowlist or set()
    counts: dict[str, list[str]] = defaultdict(list)

    for match in _WORD_RE.finditer(text):
        token = match.group(0)
        if len(token) < min_word_length:
            continue
        lower = token.lower()
        if lower in _STOPWORDS:
            continue
        lemma = lemmatize(token)
        if len(lemma) < 4:
            continue
        if lemma in _STOPWORDS or lemma in allow:
            continue
        counts[lemma].append(token)

    flagged = [
        {"lemma": lemma, "count": len(tokens), "tokens": list(tokens)}
        for lemma, tokens in counts.items()
        if len(tokens) > max_occurrences
    ]
    flagged.sort(key=lambda entry: (-entry["count"], entry["lemma"]))
    return flagged


def find_sentences_containing_lemma(text: str, lemma: str) -> list[str]:
    """Return every sentence in ``text`` that contains ``lemma``.

    Preserves document order so callers can choose which occurrence to keep
    (typically the first) and which to rewrite.
    """
    target = lemma.lower()
    results: list[str] = []
    for raw in _SENTENCE_FIND_RE.split(text or ""):
        sentence = raw.strip()
        if not sentence:
            continue
        for match in _WORD_RE.finditer(sentence):
            if lemmatize(match.group(0)) == target:
                results.append(sentence)
                break
    return results


def context_for_sentence(text: str, sentence: str, window: int = 2) -> str:
    """Return the ``window`` sentences before and after ``sentence``.

    Used to give the rewrite model enough surrounding prose to preserve
    tone, voice, and continuity when replacing a single sentence.
    """
    sentences = [s.strip() for s in _SENTENCE_FIND_RE.split(text or "") if s.strip()]
    try:
        idx = sentences.index(sentence.strip())
    except ValueError:
        return ""
    lo = max(0, idx - window)
    hi = min(len(sentences), idx + window + 1)
    before = sentences[lo:idx]
    after = sentences[idx + 1:hi]
    chunks = []
    if before:
        chunks.append(" ".join(before))
    if after:
        chunks.append(" ".join(after))
    return " [...] ".join(chunks)


def sentence_contains_lemma(sentence: str, lemma: str) -> bool:
    """True if any token in ``sentence`` lemmatizes to ``lemma``."""
    target = lemma.lower()
    return any(
        lemmatize(match.group(0)) == target
        for match in _WORD_RE.finditer(sentence or "")
    )


def format_overused_report(flagged: list[dict]) -> str:
    """Format the output of :func:`find_overused_words` into a readable report."""
    if not flagged:
        return "No overused words found."

    lines = [f"Found {len(flagged)} overused word(s):\n"]
    for entry in flagged:
        lemma = entry["lemma"]
        count = entry["count"]
        sample_tokens = ", ".join(sorted(set(entry["tokens"]))[:5])
        lines.append(f"  - {lemma!r} x{count} (forms: {sample_tokens})")
    return "\n".join(lines)
