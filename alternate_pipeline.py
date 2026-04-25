"""Pure orchestration helpers for the alternate story pipeline."""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from alternate_models import (
    AlternateStoryArtifacts,
    ChapterDraft,
    ChapterEnhancement,
    ChapterPlanEntry,
    LocationNeeds,
    StorySpine,
)
from alternate_modules import (
    AlternateChapterDraftGenerator,
    AlternateChapterPlanGenerator,
    AlternateChapterSummaryGenerator,
    AlternateCharactersGenerator,
    AlternateLocationsGenerator,
    AlternateRandomDetailGenerator,
    AlternateSpineGenerator,
    AlternateTimelineGenerator,
    AlternateWorldQuestionGenerator,
    AlternateWorldRulesGenerator,
    ChapterEnhancementGenerator,
    LocationNeedsGenerator,
)
from story_modules import CorePremiseGenerator, QuestionGenerator, QuestionWithAnswer
from world_bible import WorldBible


FeedbackRunner = Callable[[Callable[..., object], dict[str, object], str], object]


@dataclass
class AlternateGenerators:
    """DSPy modules used by the alternate pipeline."""

    question: object
    core_premise: object
    spine: object
    world_questions: object
    world_rules: object
    characters: object
    location_needs: object
    locations: object
    timeline: object
    chapter_plan: object
    enhancement: object
    random_detail: object
    chapter_draft: object
    chapter_summary: object


@dataclass
class WorldSections:
    """Approved world-bible sections."""

    rules: str
    characters: str
    location_needs: LocationNeeds
    locations: str
    timeline: str

    @property
    def world_bible(self) -> WorldBible:
        """Return the shared world-bible model."""
        return WorldBible(
            rules=self.rules,
            characters=self.characters,
            locations=self.locations,
            plot_timeline=self.timeline,
        )


@dataclass
class ChapterArtifacts:
    """Approved per-chapter outputs."""

    enhancements: list[ChapterEnhancement]
    random_details: list[str]
    summaries: list[str]
    drafts: list[ChapterDraft]


def build_alternate_generators() -> AlternateGenerators:
    """Create all DSPy modules used by the alternate pipeline."""
    return AlternateGenerators(
        question=QuestionGenerator(),
        core_premise=CorePremiseGenerator(),
        spine=AlternateSpineGenerator(),
        world_questions=AlternateWorldQuestionGenerator(),
        world_rules=AlternateWorldRulesGenerator(),
        characters=AlternateCharactersGenerator(),
        location_needs=LocationNeedsGenerator(),
        locations=AlternateLocationsGenerator(),
        timeline=AlternateTimelineGenerator(),
        chapter_plan=AlternateChapterPlanGenerator(),
        enhancement=ChapterEnhancementGenerator(),
        random_detail=AlternateRandomDetailGenerator(),
        chapter_draft=AlternateChapterDraftGenerator(),
        chapter_summary=AlternateChapterSummaryGenerator(),
    )


def compose_qa_text(questions_with_answers: Sequence[QuestionWithAnswer]) -> str:
    """Render approved QA pairs as stable text for DSPy prompts."""
    pairs = [
        f"Q: {item.question}\nA: {item.proposed_answer}"
        for item in questions_with_answers
    ]
    return "\n\n".join(pairs)


def render_chapter_plan(chapter_plan: Sequence[ChapterPlanEntry]) -> str:
    """Render structured chapter plans as markdown."""
    return "\n\n".join(chapter.full_text for chapter in chapter_plan)


def generate_core_premise(
    generators: AlternateGenerators,
    idea: str,
    qa_text: str,
    run_with_feedback: FeedbackRunner,
) -> str:
    """Generate and approve the core premise."""
    result = run_with_feedback(
        generators.core_premise,
        {"idea": idea, "qa_pairs": qa_text},
        "core_premise",
    )
    return str(getattr(result, "core_premise"))


def generate_spine(
    generators: AlternateGenerators,
    core_premise: str,
    qa_text: str,
    run_with_feedback: FeedbackRunner,
) -> StorySpine:
    """Generate and approve the structured story spine."""
    result = run_with_feedback(
        generators.spine,
        {"core_premise": core_premise, "qa_pairs": qa_text},
        "spine",
    )
    return getattr(result, "spine")


def generate_world_qa(
    generators: AlternateGenerators,
    core_premise: str,
    qa_text: str,
) -> str:
    """Generate world-building questions and return proposed answers."""
    result = generators.world_questions(core_premise=core_premise, qa_pairs=qa_text)
    return compose_qa_text(result.questions_with_answers)


def generate_rules(
    generators: AlternateGenerators,
    core_premise: str,
    qa_text: str,
    world_qa: str,
    run_with_feedback: FeedbackRunner,
) -> str:
    """Generate and approve world rules."""
    result = run_with_feedback(
        generators.world_rules,
        {"core_premise": core_premise, "qa_pairs": qa_text, "world_qa": world_qa},
        "world_rules",
    )
    return str(getattr(result, "world_rules"))


def generate_characters(
    generators: AlternateGenerators,
    core_premise: str,
    qa_text: str,
    world_rules: str,
    run_with_feedback: FeedbackRunner,
) -> str:
    """Generate and approve characters."""
    result = run_with_feedback(
        generators.characters,
        {"core_premise": core_premise, "qa_pairs": qa_text, "world_rules": world_rules},
        "characters",
    )
    return str(getattr(result, "characters"))


def generate_location_needs(
    generators: AlternateGenerators,
    characters: str,
    run_with_feedback: FeedbackRunner,
) -> LocationNeeds:
    """Generate and approve compact location needs."""
    result = run_with_feedback(
        generators.location_needs,
        {"characters": characters},
        "location_needs",
    )
    return getattr(result, "location_needs")


def generate_locations(
    generators: AlternateGenerators,
    core_premise: str,
    world_rules: str,
    location_needs: LocationNeeds,
    run_with_feedback: FeedbackRunner,
) -> str:
    """Generate and approve locations."""
    result = run_with_feedback(
        generators.locations,
        {
            "core_premise": core_premise,
            "world_rules": world_rules,
            "location_needs": location_needs.full_text,
        },
        "locations",
    )
    return str(getattr(result, "locations"))


def generate_timeline(
    generators: AlternateGenerators,
    core_premise: str,
    spine: StorySpine,
    world_bible: WorldBible,
    run_with_feedback: FeedbackRunner,
) -> str:
    """Generate and approve timeline using premise, spine, and world bible."""
    result = run_with_feedback(
        generators.timeline,
        {
            "core_premise": core_premise,
            "spine": spine.full_text,
            "world_rules": world_bible.rules,
            "characters": world_bible.characters,
            "locations": world_bible.locations,
        },
        "plot_timeline",
    )
    return str(getattr(result, "plot_timeline"))


def generate_world_sections(
    generators: AlternateGenerators,
    core_premise: str,
    qa_text: str,
    spine: StorySpine,
    world_qa: str,
    run_with_feedback: FeedbackRunner,
) -> WorldSections:
    """Generate approved world-bible sections one by one."""
    rules = generate_rules(
        generators,
        core_premise,
        qa_text,
        world_qa,
        run_with_feedback,
    )
    characters = generate_characters(
        generators,
        core_premise,
        qa_text,
        rules,
        run_with_feedback,
    )
    needs = generate_location_needs(generators, characters, run_with_feedback)
    locations = generate_locations(
        generators,
        core_premise,
        rules,
        needs,
        run_with_feedback,
    )
    partial_bible = WorldBible(
        rules=rules,
        characters=characters,
        locations=locations,
        plot_timeline="",
    )
    timeline = generate_timeline(
        generators,
        core_premise,
        spine,
        partial_bible,
        run_with_feedback,
    )
    return WorldSections(rules, characters, needs, locations, timeline)


def generate_chapter_plan(
    generators: AlternateGenerators,
    core_premise: str,
    spine: StorySpine,
    world_bible: WorldBible,
    run_with_feedback: FeedbackRunner,
) -> list[ChapterPlanEntry]:
    """Generate and approve the structured chapter plan."""
    result = run_with_feedback(
        generators.chapter_plan,
        {
            "core_premise": core_premise,
            "spine": spine.full_text,
            "world_bible": world_bible.full_text,
        },
        "chapter_plan",
    )
    return list(getattr(result, "chapter_plan"))


def generate_chapter_artifacts(
    generators: AlternateGenerators,
    world_bible: WorldBible,
    chapter_plan: Sequence[ChapterPlanEntry],
    run_with_feedback: FeedbackRunner,
) -> ChapterArtifacts:
    """Generate approved enhancement, random detail, prose, and summary per chapter."""
    plan_text = render_chapter_plan(chapter_plan)
    enhancements: list[ChapterEnhancement] = []
    random_details: list[str] = []
    summaries: list[str] = []
    drafts: list[ChapterDraft] = []
    for chapter in chapter_plan:
        artifact = generate_one_chapter(
            generators, world_bible, plan_text, chapter, summaries, run_with_feedback
        )
        enhancements.append(artifact.enhancement)
        random_details.append(artifact.random_detail)
        drafts.append(artifact.draft)
        summaries.append(artifact.summary)
    return ChapterArtifacts(enhancements, random_details, summaries, drafts)


@dataclass
class SingleChapterArtifacts:
    """Artifacts for a single approved chapter."""

    enhancement: ChapterEnhancement
    random_detail: str
    draft: ChapterDraft
    summary: str


def generate_one_chapter(
    generators: AlternateGenerators,
    world_bible: WorldBible,
    plan_text: str,
    chapter: ChapterPlanEntry,
    previous_summaries: Sequence[str],
    run_with_feedback: FeedbackRunner,
) -> SingleChapterArtifacts:
    """Generate approved artifacts for one chapter."""
    current = chapter.full_text
    enhancement = generate_enhancement(generators, world_bible, plan_text, current)
    random_detail = generate_random_detail(generators, world_bible, current)
    draft = generate_draft(
        generators,
        world_bible,
        plan_text,
        current,
        previous_summaries,
        enhancement,
        random_detail,
        run_with_feedback,
    )
    summary = generate_summary(generators, current, draft.chapter_text)
    return SingleChapterArtifacts(enhancement, random_detail, draft, summary)


def generate_enhancement(
    generators: AlternateGenerators,
    world_bible: WorldBible,
    plan_text: str,
    current_chapter: str,
) -> ChapterEnhancement:
    """Generate chapter enhancement notes."""
    result = generators.enhancement(
        world_bible=world_bible.full_text,
        chapter_plan=plan_text,
        current_chapter=current_chapter,
    )
    return result.enhancement


def generate_random_detail(
    generators: AlternateGenerators,
    world_bible: WorldBible,
    current_chapter: str,
) -> str:
    """Generate a random detail for chapter prose."""
    result = generators.random_detail(
        world_bible=world_bible.full_text,
        current_chapter=current_chapter,
    )
    return str(result.random_detail)


def generate_draft(
    generators: AlternateGenerators,
    world_bible: WorldBible,
    plan_text: str,
    current_chapter: str,
    previous_summaries: Sequence[str],
    enhancement: ChapterEnhancement,
    random_detail: str,
    run_with_feedback: FeedbackRunner,
) -> ChapterDraft:
    """Generate and approve one chapter draft."""
    result = run_with_feedback(
        generators.chapter_draft,
        {
            "world_bible": world_bible.full_text,
            "chapter_plan": plan_text,
            "current_chapter": current_chapter,
            "previous_summaries": "\n".join(previous_summaries),
            "enhancement": enhancement.full_text,
            "random_detail": random_detail,
        },
        "chapter",
    )
    return getattr(result, "chapter")


def generate_summary(
    generators: AlternateGenerators,
    current_chapter: str,
    chapter_text: str,
) -> str:
    """Generate a factual summary of approved chapter prose."""
    result = generators.chapter_summary(
        current_chapter=current_chapter,
        chapter_text=chapter_text,
    )
    return str(result.chapter_summary)


def build_artifacts(
    idea: str,
    qa_text: str,
    core_premise: str,
    spine: StorySpine,
    world_sections: WorldSections,
    chapter_plan: list[ChapterPlanEntry],
    chapter_artifacts: ChapterArtifacts,
) -> AlternateStoryArtifacts:
    """Collect approved artifacts into the save model."""
    return AlternateStoryArtifacts(
        idea=idea,
        qa_text=qa_text,
        core_premise=core_premise,
        spine=spine,
        location_needs=world_sections.location_needs,
        world_bible=world_sections.world_bible,
        chapter_plan=chapter_plan,
        enhancements=chapter_artifacts.enhancements,
        random_details=chapter_artifacts.random_details,
        chapter_summaries=chapter_artifacts.summaries,
        chapters=chapter_artifacts.drafts,
    )


def save_alternate_story_output(
    output_dir: str, artifacts: AlternateStoryArtifacts
) -> str:
    """Save all approved alternate-pipeline artifacts to markdown."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "alternate_story_output.md")
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(render_alternate_story_output(artifacts))
    return output_path


def render_alternate_story_output(artifacts: AlternateStoryArtifacts) -> str:
    """Render all alternate-pipeline artifacts as markdown."""
    sections = [
        "# Alternate Story Output",
        f"## Initial Idea\n{artifacts.idea}",
        f"## Clarification QA\n{artifacts.qa_text}",
        f"## Core Premise\n{artifacts.core_premise}",
        f"## Story Spine\n{artifacts.spine.full_text}",
        f"## World Bible\n{artifacts.world_bible.full_text}",
        f"## Location Needs\n{artifacts.location_needs.full_text}",
        f"## Chapter Plan\n{render_chapter_plan(artifacts.chapter_plan)}",
        render_chapter_artifacts(artifacts),
        f"## Final Story\n{artifacts.final_story}",
    ]
    return "\n\n".join(sections).strip() + "\n"


def render_chapter_artifacts(artifacts: AlternateStoryArtifacts) -> str:
    """Render per-chapter auxiliary artifacts."""
    blocks = ["## Chapter Drafting Artifacts"]
    for index, chapter in enumerate(artifacts.chapters):
        blocks.append(
            "\n\n".join(
                [
                    f"### Chapter {index + 1}: {chapter.title}",
                    f"#### Enhancement\n{artifacts.enhancements[index].full_text}",
                    f"#### Random Detail\n{artifacts.random_details[index]}",
                    f"#### Summary\n{artifacts.chapter_summaries[index]}",
                ]
            )
        )
    return "\n\n".join(blocks)
