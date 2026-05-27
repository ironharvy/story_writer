"""Variant C (dspy.Module) pipeline: foundation → WriteStory module.

Shares the foundation with variants A and B via
:func:`core.foundation.build_foundation`, then a non-interactive two-stage
``dspy.Module`` — outline → draft each chapter, with no rolling summary —
takes over. See :class:`story_module.WriteStory`.

Once the registry refactor (Phase 7) lands, the body of :func:`write` becomes
the entry point of ``generators/dspy_module.py``.
"""

import logging

from _compat import observe
from core.artifact import update_artifact
from core.foundation import build_foundation, sanity_check
from story_module import WriteStory

logger = logging.getLogger(__name__)


def _record_outline(output_file: str, outline) -> None:
    """Write the chapter outline to the artifact."""
    update_artifact(output_file, "Chapters Plan", "", level=3)
    for i, spec in enumerate(outline, 1):
        update_artifact(
            output_file,
            f"Chapter {i}: {spec.chapter_title}",
            spec.chapter_beats,
            level=4,
        )


def _record_chapters(output_file: str, chapters) -> None:
    """Write the drafted chapters to the artifact."""
    update_artifact(output_file, "Final Story", "", level=2)
    for i, chapter in enumerate(chapters, 1):
        update_artifact(
            output_file, f"Chapter {i}: {chapter.chapter_title}", chapter.prose, level=3
        )
        logger.info("Chapter %d written (%d chars)", i, len(chapter.prose))


@observe()
def write(idea: str, title: str, output_file: str, number_of_chapters: int = 7) -> None:
    """Run the module-based story pipeline end to end."""
    updated_idea, story_title, spine, world_bible = build_foundation(
        idea, title, output_file
    )

    if not sanity_check(updated_idea, story_title, spine, world_bible):
        logger.error("Idea, spine, and world bible are not consistent")
        return

    result = WriteStory()(
        story_idea=updated_idea,
        story_title=story_title,
        story_spine=spine,
        world_bible=world_bible,
        number_of_chapters=number_of_chapters,
    )

    _record_outline(output_file, result.outline)
    _record_chapters(output_file, result.chapters)
