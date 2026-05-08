"""Post-generation quality assurance checks for story output."""

import logging
from argparse import Namespace

from postprocessing import find_similar_sentences, format_report
import ui

logger = logging.getLogger(__name__)


def run_quality_checks(args: Namespace, final_story_text: str) -> None:
    """Run all post-generation QA checks on the final story text."""
    _run_similarity_check(
        check_enabled=args.check_similar,
        threshold=args.similar_threshold,
        final_story_text=final_story_text,
    )


def _run_similarity_check(
    check_enabled: bool,
    threshold: float,
    final_story_text: str,
) -> None:
    """Detect and report similar sentences in the generated story."""
    if not check_enabled:
        return

    similar_pairs = find_similar_sentences(
        final_story_text,
        threshold=threshold,
    )
    report = format_report(similar_pairs)
    ui.print_similarity_report(report)
    if similar_pairs:
        logger.warning(
            "Detected %d similar sentence pair(s) in generated story",
            len(similar_pairs),
        )
