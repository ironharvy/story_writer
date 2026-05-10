"""CLI entry point for the dspy.Module-based story pipeline variant.

Reuses argument parsing and runtime setup from :mod:`mymain`; only the
pipeline implementation differs (see :mod:`pipeline_module`).
"""

import logging

from dotenv import load_dotenv

from artifact import initialize_artifact, update_artifact
from mymain import (
    configure_logging,
    configure_runtime,
    format_generation_parameters,
    parse_args,
)
from pipeline_module import write

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    configure_logging(args)
    configure_runtime(args)

    initialize_artifact(args.output_file)
    update_artifact(
        args.output_file, "Generation Parameters", format_generation_parameters(args)
    )

    logger.info(
        "Starting module-based story writer with %d chapters", args.number_of_chapters
    )
    write(args.idea, args.title, args.output_file, args.number_of_chapters)
