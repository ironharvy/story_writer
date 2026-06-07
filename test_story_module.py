from generators._dspy_module_impl import DraftedChapter, WriteStory


def test_drafted_chapter_holds_outline_entry_and_prose():
    chapter = DraftedChapter(
        chapter_title="The Letter",
        chapter_beats="Mara finds the letter; she decides to leave.",
        prose="Mara turned the envelope over twice before she broke the seal.",
    )
    assert chapter.chapter_title == "The Letter"
    assert "broke the seal" in chapter.prose


def test_write_story_module_builds_its_two_stages():
    module = WriteStory()
    assert hasattr(module, "build_outline")
    assert hasattr(module, "draft_chapter")
