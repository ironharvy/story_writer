import logging

from _compat import observe
from artifact import update_artifact
from story import (
    WorldBible,
    run_clarify_idea,
    run_enhance_chapter,
    run_enhance_character,
    run_enhance_location,
    run_generate_chapters_plan,
    run_generate_characters,
    run_generate_core_premise,
    run_generate_locations,
    run_generate_rules_of_the_world,
    run_generate_spine,
    run_generate_story_so_far,
    run_generate_timeline,
    sanity_check,
)


def _snip(value, limit: int = 240) -> str:
    text = str(value).replace("\n", " ⏎ ")
    return text if len(text) <= limit else text[:limit] + "…"


logger = logging.getLogger(__name__)


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
def write(idea: str, title: str, output_file: str, number_of_chapters: int = 7):
    logger.info("STEP clarify_idea | inputs idea=%s | title=%s",
                _snip(idea), _snip(title, 80))
    updated_idea, story_title = run_clarify_idea(idea, title)
    logger.info("STEP clarify_idea | output title=%s | updated_idea=%s",
                _snip(story_title, 80), _snip(updated_idea, 600))

    update_artifact(output_file, "Story Title", story_title)
    logger.info(f"Story title {story_title} added to artifact {output_file}")

    # Core premise
    logger.info("STEP core_premise | inputs idea=%s", _snip(updated_idea))
    core_premise = run_generate_core_premise(updated_idea)
    logger.info("STEP core_premise | output=%s", _snip(core_premise, 600))
    update_artifact(output_file, "Core Premise", core_premise)
    logger.info(f"Core premise added to artifact {output_file}")

    # Spine
    logger.info("STEP spine | inputs idea=%s | core_premise=%s",
                _snip(updated_idea), _snip(core_premise))
    spine = run_generate_spine(updated_idea, core_premise)
    logger.info("STEP spine | output=%s", _snip(spine, 800))
    update_artifact(output_file, "Spine", spine)
    logger.info(f"Spine added to artifact {output_file}")

    # World bible
    world_bible = build_world_bible(updated_idea, story_title, spine, output_file)

    logger.info("STEP sanity_check | running on world_bible (chars=%d, locs=%d)",
                len(world_bible.characters), len(world_bible.locations))
    is_consistent = sanity_check(updated_idea, story_title, spine, world_bible)
    logger.info("STEP sanity_check | output is_consistent=%s", is_consistent)
    if not is_consistent:
        logger.error("Idea, spine, and world bible are not consistent")
        return

    # Chapters plan
    logger.info("STEP chapters_plan | inputs idea=%s | title=%s | spine=%s | world_bible(chars=%d, locs=%d) | number_of_chapters=%d",
                _snip(updated_idea), _snip(story_title, 80), _snip(spine),
                len(world_bible.characters), len(world_bible.locations), number_of_chapters)
    chapters_plan = run_generate_chapters_plan(updated_idea, story_title, spine, world_bible, number_of_chapters)
    logger.info("STEP chapters_plan | output count=%d", len(chapters_plan))
    update_artifact(output_file, "Chapters Plan", "", level=3)

    for i, chapter in enumerate(chapters_plan, 1):
        logger.info("STEP chapters_plan | chapter %d=%s", i, _snip(chapter, 300))
        update_artifact(output_file, f"Chapter {i}: {chapter.chapter_title}", chapter.chapter_beats, level=4)
    logger.info(f"Chapters plan added to artifact {output_file}")

    # Final story
    story_so_far = ""
    update_artifact(output_file, "Final Story", "", level=2)

    for i, chapter in enumerate(chapters_plan, 1):
        chapter_plan_str = f"{chapter.chapter_title}\n{chapter.chapter_beats}"
        logger.info("STEP enhance_chapter[%d/%d] | chapter_outline=%s | story_so_far_len=%d",
                    i, len(chapters_plan), _snip(chapter_plan_str, 200), len(story_so_far))
        enhanced = run_enhance_chapter(
            chapter_plan_str, updated_idea, story_title, spine, world_bible, story_so_far,
            chapter_index=i, total_chapters=len(chapters_plan),
        )
        logger.info("STEP enhance_chapter[%d/%d] | output_chars=%d preview=%s",
                    i, len(chapters_plan), len(enhanced), _snip(enhanced, 240))
        update_artifact(output_file, f"Chapter {i}: {chapter.chapter_title}", enhanced, level=3)
        logger.info(f"Chapter {i} added to artifact {output_file}")
        plan_so_far = chapters_plan[:i]
        logger.info("STEP story_so_far[%d/%d] | summarizing | plan_so_far_count=%d",
                    i, len(chapters_plan), len(plan_so_far))

        chapter_plan_strs = [c.chapter_title + "\n" + c.chapter_beats for c in plan_so_far]
        story_so_far = run_generate_story_so_far(chapter_plan_strs, story_so_far, enhanced)
        logger.info("STEP story_so_far[%d/%d] | new_len=%d", i, len(chapters_plan), len(story_so_far))
