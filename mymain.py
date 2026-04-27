import argparse
import logging
import random
import os

from rich.console import Console
from rich.prompt import Confirm, Prompt

import dspy
from _compat import observe
from dspy_runtime import DSPyConfig, configure_dspy
from logging_config import setup_logging

logger = logging.getLogger(__name__)
console = Console()


def initialize_artifact(fname: str) -> None:
    with open(fname, "w") as f:
        f.write("# Story\n\n")


def format_generation_parameters(args: argparse.Namespace) -> str:
    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    return "\n".join(
        [
            f"- model: `{model_name}`",
            f"- provider: `{args.provider}`",
            f"- max_tokens: `{args.max_tokens}`",
            f"- cache: `{args.cache}`",
            f"- memory_cache: `{args.memory_cache}`",
            f"- cache_dir: `{args.cache_dir}`",
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--api-key")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="The maximum number of tokens to use for the model. Defaults to 4096.",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DSPy disk cache.",
    )
    parser.add_argument(
        "--memory-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable DSPy in-memory cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.environ.get("DSPY_CACHE_DIR", ".cache/dspy"),
        help="Override DSPy disk cache directory.",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=".tmp/mymain.log")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv full firehose.",
    )
    parser.add_argument("--idea", )
    parser.add_argument("--output-file", default=".tmp/story.md")
    return parser.parse_args()


def configure_logging(args) -> None:
    setup_logging(
        verbosity=args.verbose,
        log_level=args.log_level,
        log_file=args.log_file,
    )


def configure_runtime(args) -> None:
    model_name = args.model if "/" in args.model else f"{args.provider}/{args.model}"
    configure_dspy(
        DSPyConfig(
            model_name=model_name,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            cache=args.cache,
            memory_cache=args.memory_cache,
            cache_dir=args.cache_dir,
        )
    )


@observe()
def ask(question: str, proposed_answer: str):
    console.print(question)
    console.print(proposed_answer)
    if not Confirm.ask("Is this correct?", default=True):
        answer = Prompt.ask("Enter your answer")
        return answer, False
    return proposed_answer, True


def update_artifact(fname: str, section: str, value: str, level: int = 2):
    if not os.path.exists(fname):
        initialize_artifact(fname)
    
    with open(fname, "a") as f:
        # Add heading with appropriate level
        f.write(f"{'#' * level} {section}\n\n{value}\n\n")


@observe()
def run_clarify_idea(idea: str):
    class ClarifyIdea(dspy.Signature):
        """Generate questions to clarify the story idea"""
        idea: str = dspy.InputField()
        questions: list[str] = dspy.OutputField(desc="List of questions to clarify the story idea")
        proposed_answers: list[str] = dspy.OutputField(desc="List of proposed answers to the questions")

    clarify_idea = dspy.ChainOfThought(ClarifyIdea)
    result = clarify_idea(idea=idea)
    qas = []
    for i in range(len(result.questions)):
        proposed_answer, _ = ask(result.questions[i], result.proposed_answers[i])
        qas.append({
            "question": result.questions[i],
            "proposed_answer": proposed_answer
        })
    return qas


@observe()
def run_generate_core_premise(idea: str):
    class GenerateCorePremise(dspy.Signature):
        """Generate a core premise for the story"""
        idea: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the core premise of the story")
        core_premise: str = dspy.OutputField()
    
    feedback = ""
    core_prem_func = dspy.ChainOfThought(GenerateCorePremise)
    while True:
        generate_core_premise = core_prem_func(idea=idea, feedback=feedback)
        
        feedback, is_correct = ask("Core premise:", generate_core_premise.core_premise)
        if is_correct:
            return feedback
        

@observe()
def run_generate_spine(idea: str, core_premise: str):
    class GenerateSpine(dspy.Signature):
        """Generate a a structured story spine using 7 step formula for building a compelling narrative arc"""
        idea: str = dspy.InputField()
        core_premise: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the spine of the story")
        spine: str = dspy.OutputField()

    feedback = ""
    generate_spine_func = dspy.ChainOfThought(GenerateSpine)
    while True:
        spine = generate_spine_func(idea=idea, core_premise=core_premise, feedback=feedback)
        
        feedback, is_correct = ask("Spine:", spine.spine)
        if is_correct:
            return spine.spine


@observe()
def run_generate_rules_of_the_world(idea: str, spine: str):
    class GenerateRulesOfTheWorld(dspy.Signature):
        """Generate rules of the world for the story based on the idea and spine"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the rules of the world")
        rules_of_the_world: str = dspy.OutputField()

    feedback = ""
    generate_rules_of_the_world_func = dspy.ChainOfThought(GenerateRulesOfTheWorld)
    while True:
        rules_of_the_world = generate_rules_of_the_world_func(idea=idea, spine=spine, feedback=feedback)
        
        feedback, is_correct = ask("Rules of the world:", rules_of_the_world.rules_of_the_world)
        if is_correct:
            return rules_of_the_world.rules_of_the_world


@observe()
def run_generate_characters(idea: str, spine: str, rules_of_the_world: str):
    class GenerateCharacters(dspy.Signature):
        """Generate characters for the story based on the idea, spine and rules of the world"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the characters")
        characters: list[str] = dspy.OutputField(desc="list of characters with short descriptions")

    feedback = ""
    generate_characters_func = dspy.ChainOfThought(GenerateCharacters)
    while True:
        characters = generate_characters_func(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, feedback=feedback)
        
        feedback, is_correct = ask("Characters:", characters.characters)
        if is_correct:
            return characters.characters


@observe()
def run_enhance_character(character: str, idea: str, spine: str, rules_of_the_world: str):
    class EnhanceCharacter(dspy.Signature):
        """Enhance the character with more details"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        character: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the character")
        enhanced_character: str = dspy.OutputField(desc="Elaborate description of the character")
    
    feedback = ""
    enhance_character_func = dspy.ChainOfThought(EnhanceCharacter)
    while True:
        enhanced_character = enhance_character_func(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, character=character, feedback=feedback)
        
        feedback, is_correct = ask("Enhanced character:", enhanced_character.enhanced_character)
        if is_correct:
            return enhanced_character.enhanced_character


@observe()
def run_generate_locations(idea: str, spine: str, rules_of_the_world: str, characters: list[str]):
    class GenerateLocations(dspy.Signature):
        """Generate locations for the story based on the idea, spine and rules of the world"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        characters: list[str] = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the locations")
        locations: list[str] = dspy.OutputField(desc="list of locations with short descriptions")

    feedback = ""
    generate_locations_func = dspy.ChainOfThought(GenerateLocations)
    while True:
        locations = generate_locations_func(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, characters=characters, feedback=feedback)
        
        feedback, is_correct = ask("Locations:", locations.locations)
        if is_correct:
            return locations.locations


@observe()
def run_enhance_location(location: str, idea: str, spine: str, rules_of_the_world: str, characters: list[str]):
    class EnhanceLocation(dspy.Signature):
        """Enhance the location with more details"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        characters: list[str] = dspy.InputField()
        location: str = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the locations")
        enhanced_location: str = dspy.OutputField(desc="Elaborate description of the location")
    
    feedback = ""
    enhance_location_func = dspy.ChainOfThought(EnhanceLocation)
    while True:
        enhanced_location = enhance_location_func(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, characters=characters, location=location, feedback=feedback)
        
        feedback, is_correct = ask("Enhanced location:", enhanced_location.enhanced_location)
        if is_correct:
            return enhanced_location.enhanced_location


@observe()
def run_generate_timeline(idea: str, spine: str, rules_of_the_world: str, characters: list[str], locations: list[str]):
    class GenerateTimeline(dspy.Signature):
        """Generate a timeline for the story based on the idea, spine, rules of the world, characters and locations"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the timeline")
        timeline: list[str] = dspy.OutputField(desc="list of events in the story")
    
    feedback = ""
    generate_timeline_func = dspy.ChainOfThought(GenerateTimeline)
    while True:
        timeline = generate_timeline_func(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, characters=characters, locations=locations, feedback=feedback)
        
        feedback, is_correct = ask("Timeline:", timeline.timeline)
        if is_correct:
            return timeline.timeline


@observe()
def run_generate_chapters_plan(idea: str, spine: str, rules_of_the_world: str, characters: list[str], locations: list[str], timeline: list[str]):
    class GenerateChapters(dspy.Signature):
        """Generate chapters for the story based on the idea, spine, rules of the world, characters, locations and timeline"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        feedback: str = dspy.InputField(desc="Feedback from the user for the chapters")
        chapters: list[str] = dspy.OutputField(desc="list of chapters in the story")
    
    feedback = ""
    generate_chapters_func = dspy.ChainOfThought(GenerateChapters)
    while True:
        chapters = generate_chapters_func(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, characters=characters, locations=locations, timeline=timeline, feedback=feedback)
        
        feedback, is_correct = ask("Chapters:", chapters.chapters)
        if is_correct:
            return chapters.chapters


@observe()
def run_enhance_chapter(chapter: str, idea: str, spine: str, rules_of_the_world: str, characters: list[str], locations: list[str], timeline: list[str], story_so_far: str):
    class EnhanceChapter(dspy.Signature):
        """Enhance the chapter based on the idea, spine, rules of the world, characters, locations and timeline"""
        chapter: str = dspy.InputField()
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        detail: str = dspy.InputField(default="")
        feedback: str = dspy.InputField(desc="Feedback from the user for the chapter")
        story_so_far: str = dspy.InputField(desc="Story so far")
        enhanced_chapter: str = dspy.OutputField(desc="Enhanced chapter")
    
    class GenerateRandomDetail(dspy.Signature):
        """Generate a random detail for the chapter"""
        idea: str = dspy.InputField()
        spine: str = dspy.InputField()
        rules_of_the_world: str = dspy.InputField()
        characters: list[str] = dspy.InputField()
        locations: list[str] = dspy.InputField()
        timeline: list[str] = dspy.InputField()
        chapter: str = dspy.InputField()
        random_detail: str = dspy.OutputField(desc="Random detail that doesn't influence the chapter but makes it more interesting. Scenry description or quirky item or a meal or something else fitting the setting")


    random_gen = dspy.ChainOfThought(GenerateRandomDetail)
    if random.random() < 0.33:
        random_detail = random_gen(idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, characters=characters, locations=locations, timeline=timeline, chapter=chapter).random_detail
    else:
        random_detail = ""

    feedback = ""
    enhance_chapter_func = dspy.ChainOfThought(EnhanceChapter)
    while True:
        enhanced_chapter = enhance_chapter_func(chapter=chapter, idea=idea, spine=spine, rules_of_the_world=rules_of_the_world, characters=characters, locations=locations, timeline=timeline, detail=random_detail, feedback=feedback, story_so_far=story_so_far)
        
        feedback, is_correct = ask("Enhanced Chapter:", enhanced_chapter.enhanced_chapter)
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
    return generate_story_so_far(story_so_far=story_so_far, chapter=chapter).summary


@observe()
def write(idea: str, output_file: str):
    if not idea:
        idea = Prompt.ask("Enter your story idea")  

    questions_with_answers = run_clarify_idea(idea)
    updated_idea = idea + "\n" + "\n".join([f"{qa['question']}: {qa['proposed_answer']}" for qa in questions_with_answers])

    core_premise = run_generate_core_premise(updated_idea)
    update_artifact(output_file, "Core Premise", core_premise)
    logger.info(f"Core premise added to artifact {output_file}")

    spine = run_generate_spine(updated_idea, core_premise)
    update_artifact(output_file, "Spine", spine)
    logger.info(f"Spine added to artifact {output_file}")

    rules_of_the_world = run_generate_rules_of_the_world(updated_idea, spine)
    update_artifact(output_file, "World bible", "", level=2)
    update_artifact(output_file, "Rules of the World", rules_of_the_world, level=3)
    logger.info(f"Rules of the world added to artifact {output_file}")

    characters = run_generate_characters(updated_idea, spine, rules_of_the_world)
    updated_characters = []
    update_artifact(output_file, "Characters", "", level=3)
    
    for i, character in enumerate(characters, 1):
        updated_character = run_enhance_character(character, updated_idea, spine, rules_of_the_world) 
        updated_characters.append(updated_character)
        update_artifact(output_file, f"Character {i}", updated_character, level=4)
    logger.info(f"Characters added to artifact {output_file}")

    locations = run_generate_locations(updated_idea, spine, rules_of_the_world, updated_characters)
    updated_locations = []
    update_artifact(output_file, "Locations", "", level=3)
    
    for i, location in enumerate(locations, 1):
        updated_location = run_enhance_location(location, updated_idea, spine, rules_of_the_world, updated_characters)
        updated_locations.append(updated_location)
        update_artifact(output_file, f"Location {i}", updated_location, level=4)
    logger.info(f"Locations added to artifact {output_file}")

    timeline = run_generate_timeline(updated_idea, spine, rules_of_the_world, updated_characters, updated_locations)
    for i, entry in enumerate(timeline, 1):
        update_artifact(output_file, f"Timeline {i}", entry, level=4)
    logger.info(f"Timeline added to artifact {output_file}")

    chapters_plan = run_generate_chapters_plan(updated_idea, spine, rules_of_the_world, updated_characters, updated_locations, timeline)
    update_artifact(output_file, "Chapters Plan", "", level=3)
    
    for i, chapter in enumerate(chapters_plan, 1):
        update_artifact(output_file, f"Chapter {i}", chapter, level=4)
    logger.info(f"Chapters plan added to artifact {output_file}")

    story_so_far = ""
    update_artifact(output_file, "Finale Story", "", level=2)
    
    for i, chapter in enumerate(chapters_plan, 1):
        chapter = run_enhance_chapter(chapter, updated_idea, spine, rules_of_the_world, updated_characters, updated_locations, timeline, story_so_far)
        story_so_far = run_generate_story_so_far(story_so_far, chapter)
        update_artifact(output_file, f"Chapter {i}", chapter, level=3)
        logger.info(f"Chapter {i} added to artifact {output_file}")


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args)
    configure_runtime(args)

    initialize_artifact(args.output_file)
    update_artifact(args.output_file, "Generation Parameters", format_generation_parameters(args))
    
    logger.info("Starting story writer")
    write(args.idea, args.output_file)
