"""Story pipeline variant that drafts chapters recursively.

Everything up to and including the world bible is the existing interactive
pipeline (reused verbatim via :func:`pipeline_ws._build_foundation`). From there
a non-interactive recursive ``dspy.Module`` — outline → recurse into sub-beats
→ draft prose at the leaves → concatenate — takes over. See
:class:`story_recursive.RecursiveWriteStory`.
"""

import logging

from _compat import observe
from artifact import update_artifact
from pipeline_ws import _build_foundation  # shared idea→premise→spine→world-bible steps
from story import sanity_check
from story_recursive import RecursiveWriteStory, StoryNode

logger = logging.getLogger(__name__)


def _scene_plan_lines(nodes: list[StoryNode], indent: int = 0) -> list[str]:
    lines: list[str] = []
    for node in nodes:
        lines.append(f"{'  ' * indent}- {node.title}")
        if node.children:
            lines.extend(_scene_plan_lines(node.children, indent + 1))
    return lines


def _record_outline(output_file: str, outline) -> None:
    """Write the top-level chapter outline to the artifact."""
    update_artifact(output_file, "Chapters Plan", "", level=3)
    for i, spec in enumerate(outline, 1):
        update_artifact(
            output_file,
            f"Chapter {i}: {spec.chapter_title}",
            spec.chapter_beats,
            level=4,
        )


def _record_chapters(output_file: str, roots: list[StoryNode]) -> None:
    """Write each chapter (its scene plan, then its assembled prose)."""
    update_artifact(output_file, "Final Story", "", level=2)
    for i, node in enumerate(roots, 1):
        if node.children:
            update_artifact(
                output_file,
                f"Chapter {i}: {node.title} — scene plan",
                "\n".join(_scene_plan_lines(node.children)),
                level=4,
            )
        prose = node.assembled_prose()
        update_artifact(output_file, f"Chapter {i}: {node.title}", prose, level=3)
        logger.info(
            "Chapter %d written (%d chars, %d leaves)",
            i, len(prose), node.leaf_count(),
        )


@observe()
def write(
    idea: str,
    title: str,
    output_file: str,
    number_of_chapters: int = 7,
    max_depth: int = 2,
) -> None:
    """Run the recursive story pipeline end to end."""
    updated_idea, story_title, spine, world_bible = _build_foundation(
        idea, title, output_file
    )

    if not sanity_check(updated_idea, story_title, spine, world_bible):
        logger.error("Idea, spine, and world bible are not consistent")
        return

    result = RecursiveWriteStory()(
        story_idea=updated_idea,
        story_title=story_title,
        story_spine=spine,
        world_bible=world_bible,
        number_of_chapters=number_of_chapters,
        max_depth=max_depth,
    )

    _record_outline(output_file, result.outline)
    _record_chapters(output_file, result.roots)
