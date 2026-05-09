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
class PlanEntry:
    chapter_title: str
    chapter_beats: str


@dataclass
class WorldBible:
    story_title: str
    rules_of_the_world: list[str]
    characters: list[str]
    locations: list[str]
    timeline: list[str]


@observe()
def run_clarify_idea(story_idea: str = None, story_title: str = None) -> tuple[str, str]:
    class Clarifystory_idea(dspy.Signature):
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

    clarify_idea = dspy.ChainOfThought(Clarifystory_idea)
    result = clarify_idea(story_idea=story_idea, story_title=story_title)
    qas = []
    for i in range(len(result.questions)):
        proposed_answer, _ = ui.review_answer(
            result.questions[i],
            result.proposed_answers[i],
        )
        qas.append(f"question: {result.questions[i]}\nproposed_answer: {proposed_answer}")

    update_idea = dspy.ChainOfThought(UpdateIdea)
    #updated_idea = story_idea + "\n" + "\n".join([f"{qa['question']}: {qa['proposed_answer']}" for qa in qas])
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
    class Generatestory_spine(dspy.Signature):
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
    generate_spine_func = dspy.ChainOfThought(Generatestory_spine)
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


@observe()
def run_enhance_chapter(
    chapter: str,
    story_idea: str,
    story_title: str,
    story_spine: str,
    world_bible: WorldBible,
    story_so_far: str,
):
    class DraftChapter(dspy.Signature):
        """Draft or enhance the chapter prose"""
        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        additional_detail_to_include: str = dspy.InputField(default="")
        chapter: str = dspy.InputField()
        feedback: str = dspy.InputField(
            desc="Feedback from the user for the chapter",
        )
        story_so_far: str = dspy.InputField(desc="Story so far")
        prose: str = dspy.OutputField(desc="Chapter prose")

    class GenerateRandomDetail(dspy.Signature):
        """Generate a random detail for the chapter"""

        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        rules_of_the_world: list[str] = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        chapter: str = dspy.InputField()
        random_detail: str = dspy.OutputField(
            desc="Random detail that doesn't influence the chapter but makes it more interesting. Scenry description or quirky item or a meal or something else fitting the setting",
        )

    random_detail = ""
    random_gen = dspy.ChainOfThought(GenerateRandomDetail)
    if random.random() < 0.33:
        random_detail = random_gen(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            chapter=chapter,
        ).random_detail

    feedback = ""
    draft_chapter_func = dspy.ChainOfThought(DraftChapter)
    while True:
        result = draft_chapter_func(
            story_idea=story_idea,
            story_title=story_title,
            story_spine=story_spine,
            rules_of_the_world=world_bible.rules_of_the_world,
            characters=world_bible.characters,
            locations=world_bible.locations,
            timeline=world_bible.timeline,
            additional_detail_to_include=random_detail,
            chapter=chapter,
            feedback=feedback,
            story_so_far=story_so_far,
        )

        feedback, is_correct = ui.review_answer(
            "Drafted Chapter:",
            result.prose,
        )
        if is_correct:
            return result.prose


@observe()
def run_generate_story_so_far(
    chapter_plan_so_far: list[str],
    story_so_far: str,
    chapter: str,
) -> str:
    class Summarize(dspy.Signature):
        """Summarize story progress based on chapter plan, previous summary and last chapter."""

        chapter_plan_so_far: list[str] = dspy.InputField(
            desc="Chapters and beats that were already written"
        )
        story_so_far: str = dspy.InputField(
            desc="previous summary of the story"
        )
        chapter: str = dspy.InputField(
            desc="The newly-written chapter N. Summarize its events and APPEND to the prior log."
        )
        summary: str = dspy.OutputField(
            desc="new summary of the story"
        )

    generate_story_so_far = dspy.ChainOfThought(Summarize)
    return generate_story_so_far(
        chapter_plan_so_far=chapter_plan_so_far,
        story_so_far=story_so_far,
        chapter=chapter,
    ).summary


@observe()
def sanity_check(story_idea: str, story_title: str, story_spine: str, world_bible: WorldBible) -> bool:
    class SanityCheck(dspy.Signature):
        """Check if the story_idea, story_spine, and world bible are consistent"""
        
        story_idea: str = dspy.InputField()
        story_title: str = dspy.InputField()
        story_spine: str = dspy.InputField()
        world_bible: WorldBible = dspy.InputField()
        is_consistent: bool = dspy.OutputField(desc="Whether the story_idea, story_spine, and world bible are consistent")
    
    sanity_check = dspy.ChainOfThought(SanityCheck)
    return sanity_check(
        story_idea=story_idea,
        story_title=story_title,
        story_spine=story_spine,
        world_bible=world_bible,
    ).is_consistent
