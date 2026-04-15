"""DSPy runtime configuration and generator wiring.

Extracted from ``main.py`` without behavior changes so the CLI and future
headless workers share the same setup path.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import dspy

from dspy_optimization import try_load_optimized_module
from story_modules import (
    ChapterInpaintingGenerator,
    CorePremiseGenerator,
    QuestionGenerator,
    SpineTemplateGenerator,
    StoryGenerator,
)
from world_bible_modules import (
    WorldBibleGenerator,
    WorldBibleQuestionGenerator,
)


logger = logging.getLogger(__name__)


# Module-level guard so long-lived worker processes don't re-instrument DSPy
# for every job. Instrumentation hooks are global; re-attaching them leaks
# callbacks and can distort traces.
_LANGFUSE_INSTRUMENTED = False


def configure_dspy(
    model_name: str,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 2000,
    cache: bool = True,
    memory_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> None:
    """Configure the global DSPy language model and cache.

    This mirrors the pre-refactor behavior of ``main.configure_dspy`` exactly.
    DSPy's configuration is process-global (see ``dspy.configure``), so this
    should be called once per worker process, not per request.
    """
    kwargs: dict = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key is not None:
        kwargs["api_key"] = api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            logger.warning(
                "OPENAI_API_KEY not found in environment variables. "
                "Assuming mock or alternative setup."
            )
        else:
            kwargs["api_key"] = env_key
    elif "ollama" in model_name.lower():
        # Ollama typically doesn't need an API key; leave ``kwargs`` untouched.
        pass

    dspy.configure_cache(
        enable_disk_cache=cache,
        enable_memory_cache=memory_cache,
        disk_cache_dir=cache_dir,
    )

    logger.info(
        "Configuring DSPy model=%r max_tokens=%s cache=%s memory_cache=%s cache_dir=%r",
        model_name,
        max_tokens,
        cache,
        memory_cache,
        cache_dir,
    )
    lm = dspy.LM(model_name, max_tokens=max_tokens, cache=cache, **kwargs)

    global _LANGFUSE_INSTRUMENTED
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and not _LANGFUSE_INSTRUMENTED:
        try:
            from openinference.instrumentation.dspy import DSPyInstrumentor

            DSPyInstrumentor().instrument()
            _LANGFUSE_INSTRUMENTED = True
        except Exception as exc:
            logger.warning("Langfuse DSPy callback handler unavailable: %s", exc)

    dspy.configure(lm=lm)


def initialize_text_generators(
    *,
    use_optimized: bool = False,
    optimized_manifest: Optional[str] = None,
) -> dict:
    """Instantiate and (optionally) load optimized artifacts for the text
    generation modules used by the pipeline.

    Returns a dict keyed by module name, matching ``main.py``'s prior
    contract so the CLI refactor is a drop-in replacement.
    """
    generators = {
        "QuestionGenerator": QuestionGenerator(),
        "CorePremiseGenerator": CorePremiseGenerator(),
        "SpineTemplateGenerator": SpineTemplateGenerator(),
        "WorldBibleQuestionGenerator": WorldBibleQuestionGenerator(),
        "WorldBibleGenerator": WorldBibleGenerator(),
        "StoryGenerator": StoryGenerator(),
        "ChapterInpaintingGenerator": ChapterInpaintingGenerator(),
    }

    if use_optimized:
        logger.info(
            "Optimized text modules enabled (manifest=%r)",
            optimized_manifest,
        )
        for module_name, module in generators.items():
            try_load_optimized_module(
                module,
                module_name=module_name,
                manifest_path=optimized_manifest,
                logger=logger,
            )

    return generators
