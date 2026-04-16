from postprocessing import (
    _normalize,
    context_for_sentence,
    extract_sentences,
    extract_world_bible_allowlist,
    find_overused_words,
    find_sentences_containing_lemma,
    find_similar_sentences,
    format_overused_report,
    format_report,
    lemmatize,
    sentence_contains_lemma,
)


SAMPLE_STORY = (
    "The knight rode through the dark forest. Birds scattered from the treetops. "
    "The knight rode through the shadowy forest. A distant horn echoed across the hills. "
    "She drew her sword and advanced carefully. The air smelled of pine and damp earth. "
    "She drew her sword and moved forward carefully. Nothing stirred in the underbrush."
)


def test_normalize_preserves_leading_quote():
    assert _normalize('"Hello there," she said warmly') == '"hello there," she said warmly'


def test_normalize_preserves_leading_parenthesis():
    assert _normalize("(Whispering) she crept forward") == "(whispering) she crept forward"


def test_normalize_strips_markdown_heading():
    assert _normalize("## The Grand Adventure") == "the grand adventure"


def test_normalize_strips_unordered_list_marker():
    assert _normalize("- item one goes here") == "item one goes here"
    assert _normalize("* item two goes here") == "item two goes here"


def test_normalize_strips_ordered_list_marker():
    assert _normalize("1. first ordered item") == "first ordered item"


def test_normalize_strips_blockquote():
    assert _normalize("> quoted text here") == "quoted text here"


def test_extract_sentences_filters_short_fragments():
    text = "Hello. This is a longer proper sentence. OK. Another real sentence here."
    result = extract_sentences(text)
    assert len(result) == 2
    assert all(len(s.split()) >= 4 for s in result)


def test_extract_sentences_handles_multiple_punctuation():
    text = "What is happening here? She screamed into the void! Then silence fell over the land."
    result = extract_sentences(text)
    assert len(result) == 3


def test_find_similar_sentences_detects_near_duplicates():
    pairs = find_similar_sentences(SAMPLE_STORY, threshold=0.6)
    assert len(pairs) >= 2

    # The knight sentences should be flagged
    knight_found = any(
        "knight rode" in a.lower() and "knight rode" in b.lower()
        for a, b, _ in pairs
    )
    assert knight_found, f"Expected knight sentences to be paired, got: {pairs}"

    # The sword sentences should be flagged
    sword_found = any(
        "drew her sword" in a.lower() and "drew her sword" in b.lower()
        for a, b, _ in pairs
    )
    assert sword_found, f"Expected sword sentences to be paired, got: {pairs}"


def test_find_similar_sentences_exact_duplicates():
    text = "The cat sat on the mat. The dog barked loudly at strangers. The cat sat on the mat."
    pairs = find_similar_sentences(text, threshold=0.65)
    assert len(pairs) >= 1
    assert pairs[0][2] == 1.0  # exact duplicate


def test_find_similar_sentences_no_false_positives():
    text = (
        "The sun rose over the mountain. A fisherman cast his net into the river. "
        "Meanwhile the queen pondered her next strategic move. "
        "Children laughed and played in the village square."
    )
    pairs = find_similar_sentences(text, threshold=0.65)
    assert len(pairs) == 0


def test_find_similar_sentences_empty_text():
    assert find_similar_sentences("") == []
    assert find_similar_sentences("Short.") == []


def test_format_report_no_pairs():
    assert format_report([]) == "No similar sentences found."


def test_format_report_with_pairs():
    pairs = [("Sentence A is here.", "Sentence A was here.", 0.85)]
    report = format_report(pairs)
    assert "1 similar" in report
    assert "85%" in report
    assert "Sentence A" in report


# ---------------------------------------------------------------------------
# Overused-vocabulary detection
# ---------------------------------------------------------------------------


def test_lemmatize_collapses_plurals():
    assert lemmatize("parchment") == "parchment"
    assert lemmatize("parchments") == "parchment"
    assert lemmatize("Parchment") == "parchment"


def test_lemmatize_collapses_common_tense():
    assert lemmatize("whispered") == "whisper"
    assert lemmatize("whispering") == "whisper"


def test_lemmatize_preserves_short_words():
    # Don't over-stem: 'bus' should stay 'bus', not become 'bu'.
    assert lemmatize("bus") == "bus"
    assert lemmatize("pass") == "pass"
    assert lemmatize("focus") == "focus"


def test_find_overused_words_detects_repetition():
    text = (
        "His skin was parchment. Her voice was dry as parchment. "
        "The scroll was parchment too. He held another parchment up. "
        "It smelled of old parchment. The parchment crumbled."
    )
    flagged = find_overused_words(text, max_occurrences=3)
    lemmas = {entry["lemma"] for entry in flagged}
    assert "parchment" in lemmas
    parchment = next(e for e in flagged if e["lemma"] == "parchment")
    assert parchment["count"] == 6


def test_find_overused_words_respects_threshold():
    text = "Dragon. Dragon. Dragon. Dragon."
    assert find_overused_words(text, max_occurrences=3) != []
    assert find_overused_words(text, max_occurrences=10) == []


def test_find_overused_words_ignores_stopwords():
    text = " ".join(["the"] * 20)
    assert find_overused_words(text, max_occurrences=3) == []


def test_find_overused_words_ignores_short_tokens():
    # 'cat' is only 3 chars; should be filtered by min_word_length.
    text = "cat " * 20
    assert find_overused_words(text, max_occurrences=3) == []


def test_find_overused_words_honors_allowlist():
    text = (
        "Aragorn rode north. Aragorn drew his sword. Aragorn shouted. "
        "Aragorn charged. Aragorn fell."
    )
    allowlist = {"aragorn"}
    assert find_overused_words(text, max_occurrences=3, allowlist=allowlist) == []


def test_extract_world_bible_allowlist_collects_lemmas():
    wb = "The hero Aragorn travels to Minas Tirith carrying palantiri."
    allow = extract_world_bible_allowlist(wb)
    # Proper nouns and invented terms end up in the allowlist.
    assert "aragorn" in allow
    assert "tirith" in allow
    assert "palantiri" in allow or "palantir" in allow
    # "Minas" is lemmatized (stripped 's') — the lemma form is what gets
    # stored, and "Minas" in the story will be lemmatized the same way.
    assert lemmatize("Minas") in allow


def test_extract_world_bible_allowlist_handles_empty():
    assert extract_world_bible_allowlist("") == set()
    assert extract_world_bible_allowlist(None) == set()


def test_find_sentences_containing_lemma_preserves_order():
    text = (
        "A parchment lay on the desk. The hero studied the map. "
        "Another parchment fluttered to the floor. Wind stirred the curtains. "
        "He lifted the third parchment carefully."
    )
    sentences = find_sentences_containing_lemma(text, "parchment")
    assert len(sentences) == 3
    assert sentences[0].startswith("A parchment")
    assert "third parchment" in sentences[2]


def test_find_sentences_containing_lemma_matches_inflections():
    text = "She kept many parchments. One parchment fell. He studied parchment."
    sentences = find_sentences_containing_lemma(text, "parchment")
    assert len(sentences) == 3


def test_context_for_sentence_returns_neighbors():
    text = "Alpha alpha alpha. Beta beta beta. Gamma gamma gamma. Delta delta delta."
    ctx = context_for_sentence(text, "Gamma gamma gamma.", window=1)
    assert "Beta" in ctx
    assert "Delta" in ctx
    assert "Gamma" not in ctx  # target itself is excluded


def test_context_for_sentence_missing_sentence_returns_empty():
    assert context_for_sentence("Some text here.", "Not present.") == ""


def test_sentence_contains_lemma_matches_inflected_forms():
    assert sentence_contains_lemma("She held the parchments tightly.", "parchment")
    assert sentence_contains_lemma("He whispered softly.", "whisper")
    assert not sentence_contains_lemma("The scroll was clean.", "parchment")


def test_format_overused_report_empty():
    assert format_overused_report([]) == "No overused words found."


def test_format_overused_report_with_entries():
    report = format_overused_report([
        {"lemma": "parchment", "count": 7, "tokens": ["parchment"] * 7},
    ])
    assert "parchment" in report
    assert "x7" in report
