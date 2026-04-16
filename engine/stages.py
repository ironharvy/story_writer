"""Pipeline stages.

Each stage is a pure function:

    def run_<stage>(state, generators, prompter, options) -> StoryState

Stages read what they need from ``state``, write their slice, and return the
(possibly mutated) state. They never call ``print`` directly; status goes
through :meth:`Prompter.notify`, and any blocking waits go through
:meth:`Prompter.wait_continue` (no-op under the scripted prompter).

Optional stages (character visuals, portraits, inpainting, scene images,
similarity check) are invoked by the pipeline runner when the corresponding
:class:`PipelineOptions` flag is set. They are written to be safe to call
without that flag, but the pipeline won't by default.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

from engine.prompter import Prompter
from engine.types import (
    CharacterVisualState,
    PipelineOptions,
    QAPair,
    StoryState,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _qa_pairs_from_dspy(dspy_qa: Iterable) -> list[QAPair]:
    """Adapt ``story_modules.QuestionWithAnswer`` instances (or anything with
    ``.question`` / ``.proposed_answer``) to our persistence-friendly
    :class:`QAPair`."""
    return [
        QAPair(question=item.question, proposed_answer=item.proposed_answer)
        for item in dspy_qa
    ]


def _format_qa_text(answered: Iterable[QAPair]) -> str:
    """Render answered QAPairs back into the ``Q: ...\\nA: ...`` string that
    the DSPy premise/world-bible modules expect."""
    return "\n\n".join(
        f"Q: {qa.question}\nA: {qa.effective_answer}" for qa in answered
    )


def _split_story_chapters(story_text: str) -> list[str]:
    """Split a story by ``### Chapter `` headings, preserving prior behavior
    from ``main.py`` (drop empty fragments)."""
    return [c for c in story_text.split("### Chapter ") if c.strip()]


# ---------------------------------------------------------------------------
# Text stages
# ---------------------------------------------------------------------------


def run_ideate(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Ask the user for the initial story idea."""
    if not state.idea:
        state.idea = prompter.ask_idea()
    return state


def run_premise(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Ideation-question loop + core-premise generation + refinement loop.

    Matches the legacy ``while True:`` flow in ``main.py``: questions →
    premise → accept/refine. When refining, the user's details are folded
    back into the idea string (stacking across iterations, same as before).
    """
    q_gen = generators["QuestionGenerator"]
    cp_gen = generators["CorePremiseGenerator"]

    idea = state.idea
    while True:
        prompter.notify("info", "Generating questions to interrogate your idea...")
        q_result = q_gen(idea=idea)

        proposed = _qa_pairs_from_dspy(q_result.questions_with_answers)
        answered = prompter.answer_questions(proposed)
        # Only the most recent round is kept — prior rounds fed the refined
        # idea string so their info is already captured in ``idea``.
        state.ideation_qa = list(answered)

        prompter.notify("info", "Generating Core Premise...")
        cp_result = cp_gen(idea=idea, qa_pairs=_format_qa_text(answered))
        state.core_premise = cp_result.core_premise

        accept, refinement_details = prompter.confirm_premise(state.core_premise)
        if accept:
            break

        idea = (
            f"Original idea: {idea}\n"
            f"Refinements: {refinement_details}\n"
            f"Current Core Premise: {state.core_premise}"
        )

    return state


def run_spine(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Generate the narrative spine template from the core premise."""
    prompter.notify("info", "Generating Spine Template...")
    st_gen = generators["SpineTemplateGenerator"]
    result = st_gen(core_premise=state.core_premise)
    state.spine_template = result.spine_template

    prompter.wait_continue("Press Enter to continue to World Bible generation...")
    return state


def run_world_bible_questions(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Generate and answer follow-up questions that flesh out the world bible."""
    prompter.notify(
        "info",
        "Generating follow-up questions to help flesh out the World Bible...",
    )
    wb_q_gen = generators["WorldBibleQuestionGenerator"]
    result = wb_q_gen(
        core_premise=state.core_premise, spine_template=state.spine_template
    )
    proposed = _qa_pairs_from_dspy(result.questions_with_answers)
    state.world_bible_qa = list(prompter.answer_questions(proposed))
    return state


def run_world_bible(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Generate the world bible from premise + spine + answered WB questions."""
    prompter.notify("info", "Generating World Bible...")
    wb_gen = generators["WorldBibleGenerator"]
    result = wb_gen(
        core_premise=state.core_premise,
        spine_template=state.spine_template,
        user_additions=_format_qa_text(state.world_bible_qa),
    )
    state.world_bible = result.world_bible
    return state


# ---------------------------------------------------------------------------
# Image pipeline stages (optional)
# ---------------------------------------------------------------------------


def run_character_visuals(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Describe the main characters visually (text-only, no Replicate call).

    Split from :func:`run_character_portraits` so the text half can run even
    without a Replicate token — useful for tests and for future UI preview.
    """
    # Late import: ``story_modules`` pulls in dspy; keeping the import here
    # matches the lazy pattern in the legacy CLI.
    from story_modules import CharacterVisualDescriber

    prompter.notify("info", "Generating character visual descriptions...")
    describer = CharacterVisualDescriber()
    result = describer(world_bible=state.world_bible)

    visuals: list[CharacterVisualState] = []
    for cv in result.character_visuals:
        visuals.append(
            CharacterVisualState(
                name=cv.name,
                reference_mix=cv.reference_mix,
                distinguishing_features=cv.distinguishing_features,
                full_prompt=cv.full_prompt,
            )
        )
        prompter.notify("info", f"{cv.name}: {cv.reference_mix}")
    state.character_visuals = visuals
    return state


def run_character_portraits(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Render a portrait per character via Replicate.

    Requires ``options.replicate_api_token``. If the token is missing, this
    stage raises — :meth:`PipelineOptions.validate` should catch that earlier.
    """
    if not options.replicate_api_token:
        raise RuntimeError(
            "run_character_portraits requires options.replicate_api_token"
        )
    if not state.character_visuals:
        prompter.notify(
            "warning",
            "No character visuals available; skipping portrait generation.",
        )
        return state

    # Late import so unit tests that don't touch images don't need Replicate.
    from image_gen import ImageGenerator

    image_gen = ImageGenerator(api_token=options.replicate_api_token)
    prompter.notify("info", "Generating character portraits...")
    for cv in state.character_visuals:
        try:
            path = image_gen.generate_character_portrait(
                prompt=cv.full_prompt, character_name=cv.name
            )
            state.character_portrait_paths[cv.name] = path
            prompter.notify("success", f"Saved portrait: {path}")
        except Exception as exc:
            prompter.notify(
                "error",
                f"Failed to generate portrait for {cv.name}: {exc}",
            )
    return state


# ---------------------------------------------------------------------------
# Story + post-processing stages
# ---------------------------------------------------------------------------


def run_story(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Generate the arc outline, chapter plan, enhancers guide, and story."""
    prompter.wait_continue("Press Enter to continue to Story generation...")
    prompter.notify(
        "info", "Generating Story (Arc Outline, Chapter Plan, Final Story)..."
    )

    story_gen = generators["StoryGenerator"]
    result = story_gen(
        core_premise=state.core_premise,
        spine_template=state.spine_template,
        world_bible=state.world_bible,
    )

    state.arc_outline = result.arc_outline
    state.chapter_plan = result.chapter_plan
    state.enhancers_guide = result.enhancers_guide
    state.story_text = result.story
    # Keep ``final_story_text`` in sync so downstream stages work even when
    # inpainting is disabled.
    state.final_story_text = result.story
    return state


def run_inpaint(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Run the chapter-expansion pass and replace the final story text."""
    if options.inpaint_ratio <= 1.0:
        raise ValueError(
            f"inpaint_ratio must be > 1.0, got {options.inpaint_ratio}"
        )

    prompter.notify("info", "Running chapter inpainting pass...")
    inpaint_gen = generators["ChapterInpaintingGenerator"]
    result = inpaint_gen(
        story=state.story_text,
        world_bible=state.world_bible,
        chapter_plan=state.chapter_plan,
        expansion_ratio=options.inpaint_ratio,
    )
    state.final_story_text = result.story
    prompter.notify(
        "info",
        f"Chapter inpainting complete "
        f"({result.expanded_chapters}/{result.total_chapters} chapters expanded).",
    )
    return state


def run_scene_images(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Render a scene illustration per chapter, referencing character portraits."""
    if not options.replicate_api_token:
        raise RuntimeError(
            "run_scene_images requires options.replicate_api_token"
        )

    from image_gen import ImageGenerator
    from story_modules import SceneImagePromptGenerator

    prompter.notify("info", "Generating scene illustrations for each chapter...")
    image_gen = ImageGenerator(api_token=options.replicate_api_token)
    scene_prompt_gen = SceneImagePromptGenerator()

    character_visuals_summary = "\n".join(
        f"- {cv.name}: {cv.reference_mix}. {cv.distinguishing_features}"
        for cv in state.character_visuals
    )
    reference_paths = list(state.character_portrait_paths.values())

    chapters = _split_story_chapters(state.final_story_text)
    for i, chapter_text in enumerate(chapters, start=1):
        try:
            prompt_result = scene_prompt_gen(
                chapter_text=chapter_text,
                character_visuals_summary=character_visuals_summary,
            )
            path = image_gen.generate_scene_illustration(
                prompt=prompt_result.image_prompt,
                reference_image_paths=reference_paths,
                chapter_index=i,
            )
            state.scene_image_paths[i] = path
            prompter.notify("success", f"Chapter {i} scene: {path}")
        except Exception as exc:
            prompter.notify(
                "error", f"Failed to generate scene for chapter {i}: {exc}"
            )
    return state


def run_similarity_check(
    state: StoryState,
    generators: Mapping[str, object],
    prompter: Prompter,
    options: PipelineOptions,
) -> StoryState:
    """Scan the final story for near-duplicate sentences."""
    # Imported lazily so the engine package doesn't hard-require numpy/etc.
    from postprocessing import find_similar_sentences, format_report

    prompter.notify("header", "--- Similar Sentence Check ---")
    similar_pairs = find_similar_sentences(
        state.final_story_text, threshold=options.similar_threshold
    )
    report = format_report(similar_pairs)
    state.similarity_report = report
    prompter.notify("info", report)
    if similar_pairs:
        logger.warning(
            "Detected %d similar sentence pair(s) in generated story",
            len(similar_pairs),
        )
    return state
