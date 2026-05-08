import random
from dataclasses import dataclass

import dspy

import ui
from _compat import observe


@dataclass
class Character:
    name: str
    description: str
    role: str
    relationships: list[str]
    arc: str


@dataclass
class WorldBible:
    story_title: str
    rules_of_the_world: list[str]
    characters: list[str]
    locations: list[str]
    timeline: list[str]


@observe()
def run_clarify_idea(idea: str = None, title: str = None) -> tuple[str, str]:
    class ClarifyIdea(dspy.Signature):
        """Generate questions to clarify the story idea"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        questions: list[str] = dspy.OutputField(
            desc="List of questions to clarify the story idea",
        )
        proposed_answers: list[str] = dspy.OutputField(
            desc="List of proposed answers to the questions",
        )

    class GenerateTitle(dspy.Signature):
        """Generate a title for the story"""

        idea: str = dspy.InputField()
        title: str = dspy.OutputField()

    if not idea:
        idea = ui.ask_idea()

    clarify_idea = dspy.ChainOfThought(ClarifyIdea)
    result = clarify_idea(idea=idea, title=title)
    qas = []
    for i in range(len(result.questions)):
        proposed_answer, _ = ui.review_answer(
            result.questions[i],
            result.proposed_answers[i],
        )
        qas.append({"question": result.questions[i], "proposed_answer": proposed_answer})

    updated_idea = idea + "\n" + "\n".join([f"{qa['question']}: {qa['proposed_answer']}" for qa in qas])
    if not title:
        generate_title = dspy.ChainOfThought(GenerateTitle)
        result = generate_title(idea=updated_idea)
        title = result.title
    return updated_idea, title


@observe()
def run_generate_core_premise(idea: str) -> str:
    class GenerateCorePremise(dspy.Signature):
        """Generate a core premise for the story"""

        idea: str = dspy.InputField()
        previous_result: str = dspy.InputField(desc="Previous result of the core premise generation")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the core premise of the story",
        )
        core_premise: str = dspy.OutputField()

    feedback = ""
    previous_result = ""
    core_prem_func = dspy.ChainOfThought(GenerateCorePremise)
    while True:
        generate_core_premise = core_prem_func(idea=idea, previous_result=previous_result, feedback=feedback)

        previous_result = generate_core_premise.core_premise
        feedback, is_correct = ui.review_answer(
            "Core premise:",
            previous_result,
        )
        if is_correct:
            return feedback


@observe()
def run_generate_spine(idea: str, core_premise: str) -> str:
    class GenerateSpine(dspy.Signature):
        """Generate a a structured story spine using pixar's 7 step formula for building a compelling narrative arc"""

        idea: str = dspy.InputField()
        core_premise: str = dspy.InputField()
        previous_result: str = dspy.InputField(desc="Previous result of the spine generation")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the spine of the story",
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
    generate_spine_func = dspy.ChainOfThought(GenerateSpine)
    while True:
        spine = generate_spine_func(
            idea=idea,
            core_premise=core_premise,
            previous_result=previous_result,
            feedback=feedback,
        )

        previous_result = "\n".join(
            [
                spine.once_upon_a_time,
                spine.every_day,
                spine.until_one_day,
                spine.because_of_that,
                spine.and_because_of_that,
                spine.until_finally,
                spine.ever_since_that_day,
            ]
        )

        feedback, is_correct = ui.review_answer(
            "Spine:",
            previous_result,
        )
        if is_correct:
            return previous_result
        

@observe()
def run_generate_rules_of_the_world(idea: str, title: str, spine: str) -> list[str]:
    class GenerateRulesOfTheWorld(dspy.Signature):
        """Generate rules of the world for the story based on the idea and spine"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
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
            idea=idea,
            title=title,
            spine=spine,
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
def run_generate_characters(idea: str, title: str, spine: str, rules_of_the_world: str) -> list[str]:
    class GenerateCharacters(dspy.Signature):
        """Generate characters for the story based on the idea, spine and rules of the world"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
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
            idea=idea,
            title=title,
            spine=spine,
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
    idea: str,
    title: str,
    spine: str,
    rules_of_the_world: str,
) -> str:
    class EnhanceCharacter(dspy.Signature):
        """Enhance the character with more details, background, motivation, and personality."""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
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
            idea=idea,
            title=title,
            spine=spine,
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
    idea: str,
    title: str,
    spine: str,
    rules_of_the_world: str,
) -> list[str]:
    class GenerateLocations(dspy.Signature):
        """Generate locations for the story based on the idea, spine and rules of the world"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
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
            idea=idea,
            title=title,
            spine=spine,
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
    idea: str,
    title: str,
    spine: str,
    rules_of_the_world: str,
) -> str:
    class EnhanceLocation(dspy.Signature):
        """Enhance the location given with more details"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
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
            idea=idea,
            title=title,
            spine=spine,
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
    idea: str,
    title: str,
    spine: str,
    rules_of_the_world: list[str],
    locations: list[str],
) -> list[str]:
    class GenerateTimeline(dspy.Signature):
        """Generate a timeline for the story based on the idea, spine, rules of the world and locations"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
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
            idea=idea,
            title=title,
            spine=spine,
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
    idea: str,
    title: str,
    spine: str,
    world_bible: WorldBible,
):
    class GenerateChaptersPlan(dspy.Signature):
        """Generate chapters for the story based on the idea, spine and world bible"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the chapters",
        )
        chapters: list[str] = dspy.OutputField(desc="list of chapters in the story")

    feedback = ""
    generate_chapters_func = dspy.ChainOfThought(GenerateChaptersPlan)
    while True:
        chapters = generate_chapters_func(
            idea=idea,
            title=title,
            spine=spine,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            feedback=feedback,
        )

        feedback, is_correct = ui.review_answer("Chapters:", chapters.chapters)
        if is_correct:
            return chapters.chapters


@observe()
def run_enhance_chapter(
    chapter: str,
    idea: str,
    title: str,
    spine: str,
    world_bible: WorldBible,
    story_so_far: str,
):
    class EnhanceChapter(dspy.Signature):
        """Enhance the chapter based on the idea, spine, rules of the world, characters, locations and timeline"""

        chapter: str = dspy.InputField()
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        detail: str = dspy.InputField(default="")
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the chapter",
        )
        story_so_far: str = dspy.InputField(desc="Story so far")
        enhanced_chapter: str = dspy.OutputField(desc="Enhanced chapter")

    class GenerateRandomDetail(dspy.Signature):
        """Generate a random detail for the chapter"""

        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        chapter: str = dspy.InputField()
        random_detail: str = dspy.OutputField(
            desc="Random detail that doesn't influence the chapter but makes it more interesting. Scenry description or quirky item or a meal or something else fitting the setting",
        )

    random_gen = dspy.ChainOfThought(GenerateRandomDetail)
    if random.random() < 0.33:
        random_detail = random_gen(
            idea=idea,
            title=title,
            spine=spine,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            chapter=chapter,
        ).random_detail
    else:
        random_detail = ""

    feedback = ""
    enhance_chapter_func = dspy.ChainOfThought(EnhanceChapter)
    while True:
        enhanced_chapter = enhance_chapter_func(
            chapter=chapter,
            idea=idea,
            title=title,
            spine=spine,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            detail=random_detail,
            feedback=feedback,
            story_so_far=story_so_far,
        )

        feedback, is_correct = ui.review_answer(
            "Enhanced Chapter:",
            enhanced_chapter.enhanced_chapter,
        )
        if is_correct:
            return enhanced_chapter.enhanced_chapter


@observe()
def run_generate_story_so_far(story_so_far: str, chapter: str) -> str:
    class Summarize(dspy.Signature):
        """Summarize the story so far"""

        story_so_far: str = dspy.InputField()
        chapter: str = dspy.InputField()
        summary: str = dspy.OutputField(desc="Summary of the story so far")

    generate_story_so_far = dspy.ChainOfThought(Summarize)
    return generate_story_so_far(
        story_so_far=story_so_far,
        chapter=chapter,
    ).summary


@observe()
def sanity_check(idea: str, title: str, spine: str, world_bible: WorldBible) -> bool:
    class SanityCheck(dspy.Signature):
        """Check if the idea, spine, and world bible are consistent"""
        
        idea: str = dspy.InputField()
        title: str = dspy.InputField()
        spine: str = dspy.InputField()
        world_bible: WorldBible = dspy.InputField()
        is_consistent: bool = dspy.OutputField(desc="Whether the idea, spine, and world bible are consistent")
    
    sanity_check = dspy.ChainOfThought(SanityCheck)
    return sanity_check(
        idea=idea,
        title=title,
        spine=spine,
        world_bible=world_bible,
    ).is_consistent
