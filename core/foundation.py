"""Shared pipeline foundation: idea → premise → spine → world bible → chapter plan.

Every chapter-drafting generator consumes the output of this module — the
variants differ only in how they turn (spine, world_bible, chapters_plan)
into prose. All `run_*` stages here are still interactive (they call into
`ui.review_answer` and loop on user feedback); decoupling UI from these
stages so the orchestrator can inject the reviewer is a follow-up.
"""
from __future__ import annotations

import logging
import math

import dspy

import ui
from _compat import observe
from core.artifact import update_artifact
from core.types import PlanEntry, WorldBible

logger = logging.getLogger(__name__)


def _snip(value, limit: int = 240) -> str:
    text = str(value).replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + "…"


# --- ideation ---------------------------------------------------------------


@observe()
def run_clarify_idea(story_idea: str = None, story_title: str = None) -> tuple[str, str]:
    class ClarifyStoryIdea(dspy.Signature):
        """Generate questions to clarify the story story_idea"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        questions: list[str] = dspy.OutputField(
            desc="List of questions to clarify the story story_idea",
        )
        proposed_answers: list[str] = dspy.OutputField(
            desc="List of proposed answers to the questions",
        )

    class UpdateIdea(dspy.Signature):
        """Update the story idea based on the questions and answers"""

        story_idea: str = dspy.InputField()
        qas: list[str] = dspy.InputField()
        updated_idea: str = dspy.OutputField()

    class GenerateStoryTitle(dspy.Signature):
        """Generate a story_title for the story"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.OutputField()

    if not story_idea:
        story_idea = ui.ask_idea()

    clarify_idea = dspy.ChainOfThought(ClarifyStoryIdea)
    result = clarify_idea(story_idea=story_idea, story_title=story_title)
    qas = []
    for i in range(len(result.questions)):
        proposed_answer, _ = ui.review_answer(
            result.questions[i],
            result.proposed_answers[i],
        )
        qas.append(f"question: {result.questions[i]}\nproposed_answer: {proposed_answer}")

    update_idea = dspy.ChainOfThought(UpdateIdea)
    updated_idea = update_idea(story_idea=story_idea, qas=qas).updated_idea
    if not story_title:
        generate_title = dspy.ChainOfThought(GenerateStoryTitle)
        result = generate_title(story_idea=updated_idea)
        story_title = result.story_title
    return updated_idea, story_title


@observe()
def run_generate_core_premise(story_idea: str) -> str:
    class GenerateCorePremise(dspy.Signature):
        """Generate a core premise for the story"""

        story_idea: str = dspy.InputField()
        previous_result: str = dspy.InputField(desc="Previous result of the core premise generation")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the core premise of the story",
        )
        core_premise: str = dspy.OutputField()

    feedback = ""
    previous_result = ""
    core_prem_func = dspy.ChainOfThought(GenerateCorePremise)
    while True:
        generate_core_premise = core_prem_func(story_idea=story_idea, previous_result=previous_result, feedback=feedback)

        previous_result = generate_core_premise.core_premise
        feedback, is_correct = ui.review_answer(
            "Core premise:",
            previous_result,
        )
        if is_correct:
            return feedback


@observe()
def run_generate_spine(story_idea: str, core_premise: str) -> str:
    class GenerateStorySpine(dspy.Signature):
        """Generate a a structured story story_spine using pixar's 7 step formula for building a compelling narrative arc"""

        story_idea: str = dspy.InputField()
        core_premise: str = dspy.InputField()
        previous_result: str = dspy.InputField(desc="Previous result of the story_spine generation")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the story_spine of the story",
        )
        once_upon_a_time: str = dspy.OutputField()
        every_day: str = dspy.OutputField()
        until_one_day: str = dspy.OutputField()
        because_of_that: str = dspy.OutputField()
        and_because_of_that: str = dspy.OutputField()
        until_finally: str = dspy.OutputField()
        ever_since_that_day: str = dspy.OutputField()

    feedback = ""
    previous_result = ""
    generate_spine_func = dspy.ChainOfThought(GenerateStorySpine)
    while True:
        story_spine = generate_spine_func(
            story_idea=story_idea,
            core_premise=core_premise,
            previous_result=previous_result,
            feedback=feedback,
        )

        previous_result = "\n".join(
            [
                story_spine.once_upon_a_time,
                story_spine.every_day,
                story_spine.until_one_day,
                story_spine.because_of_that,
                story_spine.and_because_of_that,
                story_spine.until_finally,
                story_spine.ever_since_that_day,
            ]
        )

        feedback, is_correct = ui.review_answer(
            "spine:",
            previous_result,
        )
        if is_correct:
            return previous_result


# --- world bible -------------------------------------------------------------


@observe()
def run_generate_rules_of_the_world(story_idea: str, story_title: str, story_spine: str) -> list[str]:
    class GenerateRulesOfTheWorld(dspy.Signature):
        """Generate rules of the world for the story based on the story_idea and story_spine"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        previous_result: str = dspy.InputField(desc="Previous result of the rules of the world generation")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the rules of the world",
        )
        rules_of_the_world: list[str] = dspy.OutputField()

    feedback = ""
    previous_result = ""
    generate_rules_of_the_world_func = dspy.ChainOfThought(GenerateRulesOfTheWorld)
    while True:
        rules_of_the_world = generate_rules_of_the_world_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            previous_result=previous_result,
            feedback=feedback,
        )

        rules_list = list(rules_of_the_world.rules_of_the_world)
        previous_result = "\n".join(rules_list)
        feedback, is_correct = ui.review_answer(
            "Rules of the world:",
            previous_result,
        )
        if is_correct:
            return rules_list


@observe()
def run_generate_characters(story_idea: str, story_title: str, story_spine: str, rules_of_the_world: str) -> list[str]:
    class GenerateCharacters(dspy.Signature):
        """Generate characters for the story based on the story_idea, story_spine and rules of the world"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        previous_result: str = dspy.InputField(desc="Previous result of the characters generation")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the characters",
        )
        characters: list[str] = dspy.OutputField(
            desc="list of characters with short descriptions",
        )

    feedback = ""
    previous_result = ""
    generate_characters_func = dspy.ChainOfThought(GenerateCharacters)
    while True:
        characters = generate_characters_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=rules_of_the_world,
            feedback=feedback,
            previous_result=previous_result,
        )

        previous_result = "\n".join([f"{i+1}. {chr}" for i, chr in enumerate(characters.characters)])
        feedback, is_correct = ui.review_answer("Characters:", previous_result)
        if is_correct:
            return characters.characters


@observe()
def run_enhance_character(
    character: str,
    story_idea: str,
    story_title: str,
    story_spine: str,
    rules_of_the_world: str,
) -> str:
    class EnhanceCharacter(dspy.Signature):
        """Enhance the character with more details, background, motivation, and personality."""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        character: str = dspy.InputField()
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the character",
        )
        enhanced_character: str = dspy.OutputField(
            desc="Elaborate description of the character",
        )

    feedback = ""
    enhance_character_func = dspy.ChainOfThought(EnhanceCharacter)
    while True:
        enhanced_character = enhance_character_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=rules_of_the_world,
            character=character,
            feedback=feedback,
        )

        feedback, is_correct = ui.review_answer(
            "Enhanced character:",
            enhanced_character.enhanced_character,
        )
        if is_correct:
            return enhanced_character.enhanced_character


@observe()
def run_generate_locations(
    story_idea: str,
    story_title: str,
    story_spine: str,
    rules_of_the_world: str,
) -> list[str]:
    class GenerateLocations(dspy.Signature):
        """Generate locations for the story based on the story_idea, story_spine and rules of the world"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        locations_str: str = dspy.InputField(
            desc="Previous locations generated",
        )
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the locations",
        )
        locations: list[str] = dspy.OutputField(
            desc="list of locations with short descriptions",
        )

    feedback = ""
    locations_str = ""
    generate_locations_func = dspy.ChainOfThought(GenerateLocations)
    while True:
        locations = generate_locations_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=rules_of_the_world,
            locations_str=locations_str,
            feedback=feedback,
        )

        locations_str += "\n".join(locations.locations) + "\n"
        feedback, is_correct = ui.review_answer("Locations:", locations_str)
        if is_correct:
            return locations.locations


@observe()
def run_enhance_location(
    location: str,
    story_idea: str,
    story_title: str,
    story_spine: str,
    rules_of_the_world: str,
) -> str:
    class EnhanceLocation(dspy.Signature):
        """Enhance the location given with more details"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        location: str = dspy.InputField()
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the locations",
        )
        enhanced_location: str = dspy.OutputField(
            desc="Elaborate description of the location",
        )

    ui.print_status(f"Enhancing location {location}...")

    feedback = ""
    enhance_location_func = dspy.ChainOfThought(EnhanceLocation)
    while True:
        enhanced_location = enhance_location_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=rules_of_the_world,
            location=location,
            feedback=feedback,
        )

        feedback, is_correct = ui.review_answer(
            "Enhanced location:",
            enhanced_location.enhanced_location,
        )
        if is_correct:
            return enhanced_location.enhanced_location


@observe()
def run_generate_timeline(
    story_idea: str,
    story_title: str,
    story_spine: str,
    rules_of_the_world: list[str],
    locations: list[str],
) -> list[str]:
    class GenerateTimeline(dspy.Signature):
        """Generate a timeline for the story based on the story_idea, story_spine, rules of the world and locations"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        previous_timeline: str = dspy.InputField(desc="Previous timeline generated")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the timeline",
        )
        timeline: list[str] = dspy.OutputField(desc="list of events in the story")

    feedback = ""
    timeline_str = ""
    generate_timeline_func = dspy.ChainOfThought(GenerateTimeline)
    while True:
        timeline = generate_timeline_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=rules_of_the_world,
            locations=locations,
            previous_timeline=timeline_str,
            feedback=feedback,
        )

        timeline_list = list(timeline.timeline)
        timeline_str = "\n".join(timeline_list)
        feedback, is_correct = ui.review_answer("Timeline:", timeline_str)
        if is_correct:
            return timeline_list


@observe()
def run_generate_chapters_plan(
    story_idea: str,
    story_title: str,
    story_spine: str,
    world_bible: WorldBible,
    number_of_chapters: int = 7,
):
    class GenerateChaptersPlan(dspy.Signature):
        """Generate chapters for the story based on the story_idea, story_spine and world bible"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the chapters",
        )
        number_of_chapters: int = dspy.InputField(desc="Number of chapters to generate")
        chapters: list[PlanEntry] = dspy.OutputField(desc="list of chapters in the story")

    feedback = ""
    generate_chapters_func = dspy.ChainOfThought(GenerateChaptersPlan)
    while True:
        chapters = generate_chapters_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            feedback=feedback,
            number_of_chapters=number_of_chapters,
        )

        plan_str = "\n".join([chapter.chapter_title + "\n" + chapter.chapter_beats for chapter in chapters.chapters])
        feedback, is_correct = ui.review_answer("Chapters:", plan_str)
        if is_correct:
            return chapters.chapters


# --- act / spine slicing -----------------------------------------------------


_ACT_LABELS = {
    1: "Act 1: Setup",
    2: "Act 2: Confrontation",
    3: "Act 3: Resolution",
}


def act_hint_for_chapter(i: int, n: int, story_spine: str) -> dict:
    """Map 1-indexed chapter `i` (of `n`) onto a 3-act structure.

    Splits chapters using a ~25/50/25 ratio, with at least one chapter per
    act when n >= 3. Returns the act number, a human-readable label, and
    the spine sliced up to and including the current act (so chapter prose
    only sees beats it should know about, not later-act foreshadowing).
    """
    if n >= 3:
        a1 = max(1, math.ceil(n / 4))
        a3 = max(1, math.ceil(n / 4))
        a2 = n - a1 - a3
        if a2 < 1:
            a2 = 1
            a3 = max(1, n - a1 - a2)
            a1 = max(1, n - a2 - a3)
    elif n == 2:
        a1, a2, a3 = 1, 0, 1
    else:
        a1, a2, a3 = 1, 0, 0

    if i <= a1:
        act = 1
    elif i <= a1 + a2:
        act = 2
    else:
        act = 3

    # Spine is 7 newline-separated beats from Pixar's formula:
    # 0:once_upon_a_time 1:every_day 2:until_one_day  -> Act 1
    # 3:because_of_that 4:and_because_of_that         -> Act 2
    # 5:until_finally 6:ever_since_that_day           -> Act 3
    beats = story_spine.split("\n")
    cutoff = {1: 3, 2: 5, 3: 7}[act]
    spine_through_act = "\n".join(beats[:cutoff])

    return {
        "act": act,
        "label": _ACT_LABELS[act],
        "spine_through_act": spine_through_act,
    }


# --- sanity check ------------------------------------------------------------


@observe()
def sanity_check(story_idea: str, story_title: str, story_spine: str, world_bible: WorldBible) -> bool:
    class SanityCheck(dspy.Signature):
        """Check if the story_idea, story_spine, and world bible are consistent"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        world_bible: WorldBible = dspy.InputField()
        is_consistent: bool = dspy.OutputField(desc="Whether the story_idea, story_spine, and world bible are consistent")

    sanity_check_func = dspy.ChainOfThought(SanityCheck)
    return sanity_check_func(
        story_idea=story_idea,
        story_title=story_title,
        story_spine=story_spine,
        world_bible=world_bible,
    ).is_consistent


# --- world-bible + foundation orchestrators ----------------------------------


@observe()
def build_world_bible(updated_idea: str, story_title: str, spine: str, output_file: str) -> WorldBible:
    # 1. Generate rules of the world
    logger.info("STEP rules_of_the_world | inputs idea=%s | title=%s | spine=%s",
                _snip(updated_idea), _snip(story_title, 80), _snip(spine))
    rules_of_the_world = run_generate_rules_of_the_world(updated_idea, story_title, spine)
    logger.info("STEP rules_of_the_world | output=%s", _snip(rules_of_the_world, 600))
    update_artifact(output_file, "World bible", "", level=2)
    rules_str = "\n".join([f"- {rule}" for rule in rules_of_the_world])
    update_artifact(output_file, "Rules of the World", rules_str, level=3)
    logger.info(f"Rules of the world added to artifact {output_file}")

    # 2. Generate locations
    logger.info("STEP locations | inputs idea=%s | title=%s | spine=%s | rules=%s",
                _snip(updated_idea), _snip(story_title, 80), _snip(spine), _snip(rules_of_the_world))
    locations = run_generate_locations(updated_idea, story_title, spine, rules_of_the_world)
    logger.info("STEP locations | output count=%d list=%s", len(locations), _snip(locations))
    updated_locations = []
    locations_str = "\n".join([f"- {loc}" for i, loc in enumerate(locations)])
    update_artifact(output_file, "Locations", locations_str, level=3)

    for i, location in enumerate(locations, 1):
        logger.info("STEP enhance_location[%d] | location=%s", i, _snip(location))
        updated_location = run_enhance_location(location, updated_idea, story_title, spine, rules_of_the_world)
        logger.info("STEP enhance_location[%d] | output=%s", i, _snip(updated_location, 400))
        updated_locations.append(updated_location)
        update_artifact(output_file, f"Location {i}", updated_location, level=4)
    logger.info(f"Locations added to artifact {output_file}")

    # 3. Generate timeline
    logger.info("STEP timeline | inputs idea=%s | title=%s | spine=%s | rules=%s | locations_count=%d",
                _snip(updated_idea), _snip(story_title, 80), _snip(spine),
                _snip(rules_of_the_world), len(updated_locations))
    timeline = run_generate_timeline(updated_idea, story_title, spine, rules_of_the_world, updated_locations)
    logger.info("STEP timeline | output=%s", _snip(timeline, 600))
    timeline_str = "\n".join([f"- {event}" for event in timeline])
    update_artifact(output_file, "Timeline", timeline_str, level=3)
    logger.info(f"Timeline added to artifact {output_file}")

    # 4. Generate characters
    logger.info("STEP characters | inputs idea=%s | title=%s | spine=%s | rules=%s",
                _snip(updated_idea), _snip(story_title, 80), _snip(spine), _snip(rules_of_the_world))
    characters = run_generate_characters(updated_idea, story_title, spine, rules_of_the_world)
    logger.info("STEP characters | output count=%d list=%s", len(characters), _snip(characters))
    updated_characters = []
    characters_str = "\n".join([f"{i+1}. {char}" for i, char in enumerate(characters)])
    update_artifact(output_file, "Characters", characters_str, level=3)

    for i, character in enumerate(characters, 1):
        logger.info("STEP enhance_character[%d] | character=%s", i, _snip(character))
        updated_character = run_enhance_character(character, updated_idea, story_title, spine, rules_of_the_world)
        logger.info("STEP enhance_character[%d] | output=%s", i, _snip(updated_character, 400))
        updated_characters.append(updated_character)
        update_artifact(output_file, f"Character {i}", updated_character, level=4)
    logger.info(f"Characters added to artifact {output_file}")

    return WorldBible(
        story_title=story_title,
        rules_of_the_world=rules_of_the_world,
        characters=updated_characters,
        locations=updated_locations,
        timeline=timeline,
    )


@observe()
def build_foundation(idea: str, title: str, output_file: str):
    """Run the shared idea → premise → spine → world bible steps.

    Returns ``(updated_idea, story_title, spine, world_bible)``. Each variant's
    drafting pipeline starts from this 4-tuple.
    """
    updated_idea, story_title = run_clarify_idea(idea, title)
    update_artifact(output_file, "Story Title", story_title)

    core_premise = run_generate_core_premise(updated_idea)
    update_artifact(output_file, "Core Premise", core_premise)
    logger.info("STEP core_premise | output=%s", _snip(core_premise, 600))

    spine = run_generate_spine(updated_idea, core_premise)
    update_artifact(output_file, "Spine", spine)
    logger.info("STEP spine | output=%s", _snip(spine, 800))

    world_bible = build_world_bible(updated_idea, story_title, spine, output_file)
    return updated_idea, story_title, spine, world_bible
