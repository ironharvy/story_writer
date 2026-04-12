from postprocessing import _normalize, extract_sentences, find_similar_sentences, format_report


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
