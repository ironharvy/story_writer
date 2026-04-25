from types import SimpleNamespace

from alternate_models import (
    ChapterDraft,
    ChapterEnhancement,
    ChapterPlanEntry,
    LocationNeeds,
    StorySpine,
)
from alternate_pipeline import (
    AlternateGenerators,
    build_artifacts,
    generate_chapter_artifacts,
    generate_chapter_plan,
    generate_world_sections,
    render_alternate_story_output,
)
from story_modules import QuestionWithAnswer
from world_bible import WorldBible


class RecordingGenerator:
    """Callable test double that records keyword arguments."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.result


def approve(
    generator: RecordingGenerator,
    inputs: dict[str, object],
    _field_name: str,
) -> object:
    """Test feedback runner that approves first generated result."""
    return generator(**inputs)


def make_generators() -> AlternateGenerators:
    """Create mock alternate generators with stable outputs."""
    chapter = ChapterPlanEntry(
        number=1,
        title="First Door",
        purpose="Open the central conflict.",
        beats=["Hero finds the door.", "Hero chooses to enter."],
    )
    enhancement = ChapterEnhancement(
        pacing="Start quiet, then accelerate.",
        tension="Let the door feel dangerous.",
        imagery="Use cold brass and dust.",
        theme="Choice over safety.",
    )
    return AlternateGenerators(
        question=RecordingGenerator(
            SimpleNamespace(
                questions_with_answers=[
                    QuestionWithAnswer(question="Why?", proposed_answer="Because.")
                ]
            )
        ),
        core_premise=RecordingGenerator(SimpleNamespace(core_premise="Premise")),
        spine=RecordingGenerator(
            SimpleNamespace(
                spine=StorySpine(
                    setup="Setup",
                    disruption="Disruption",
                    escalation="Escalation",
                    crisis="Crisis",
                    climax="Climax",
                    resolution="Resolution",
                )
            )
        ),
        world_questions=RecordingGenerator(SimpleNamespace(questions_with_answers=[])),
        world_rules=RecordingGenerator(SimpleNamespace(world_rules="Rules")),
        characters=RecordingGenerator(SimpleNamespace(characters="Characters")),
        location_needs=RecordingGenerator(
            SimpleNamespace(location_needs=LocationNeeds(needs=["A refuge"]))
        ),
        locations=RecordingGenerator(SimpleNamespace(locations="Locations")),
        timeline=RecordingGenerator(SimpleNamespace(plot_timeline="Timeline")),
        chapter_plan=RecordingGenerator(SimpleNamespace(chapter_plan=[chapter])),
        enhancement=RecordingGenerator(SimpleNamespace(enhancement=enhancement)),
        random_detail=RecordingGenerator(SimpleNamespace(random_detail="A blue moth.")),
        chapter_draft=RecordingGenerator(
            SimpleNamespace(
                chapter=ChapterDraft(
                    title="First Door",
                    chapter_text="The hero entered.",
                )
            )
        ),
        chapter_summary=RecordingGenerator(
            SimpleNamespace(chapter_summary="The hero entered the door.")
        ),
    )


def test_world_sections_use_premise_for_world_and_spine_only_for_timeline() -> None:
    generators = make_generators()
    spine = generators.spine(core_premise="Premise", qa_pairs="QA").spine

    sections = generate_world_sections(
        generators,
        core_premise="Premise",
        qa_text="QA",
        spine=spine,
        world_qa="World QA",
        run_with_feedback=approve,
    )

    assert sections.world_bible.full_text
    assert "spine" not in generators.world_rules.calls[0]
    assert "spine" not in generators.characters.calls[0]
    assert "spine" not in generators.locations.calls[0]
    assert generators.timeline.calls[0]["spine"] == spine.full_text


def test_locations_receive_location_needs_not_full_characters() -> None:
    generators = make_generators()
    spine = generators.spine(core_premise="Premise", qa_pairs="QA").spine

    generate_world_sections(
        generators,
        core_premise="Premise",
        qa_text="QA",
        spine=spine,
        world_qa="World QA",
        run_with_feedback=approve,
    )

    location_call = generators.locations.calls[0]
    assert location_call["location_needs"] == "- A refuge"
    assert "characters" not in location_call


def test_chapter_generation_passes_beats_and_auxiliary_guidance_to_draft() -> None:
    generators = make_generators()
    world_bible = WorldBible(
        rules="Rules",
        characters="Characters",
        locations="Locations",
        plot_timeline="Timeline",
    )
    chapter_plan = generate_chapter_plan(
        generators,
        "Premise",
        generators.spine(core_premise="Premise", qa_pairs="QA").spine,
        world_bible,
        approve,
    )

    artifacts = generate_chapter_artifacts(
        generators,
        world_bible,
        chapter_plan,
        approve,
    )

    draft_call = generators.chapter_draft.calls[0]
    assert "Hero finds the door." in draft_call["current_chapter"]
    assert "Start quiet" in draft_call["enhancement"]
    assert draft_call["random_detail"] == "A blue moth."
    assert artifacts.summaries == ["The hero entered the door."]


def test_render_output_includes_all_major_artifacts() -> None:
    generators = make_generators()
    spine = generators.spine(core_premise="Premise", qa_pairs="QA").spine
    world_bible = WorldBible(
        rules="Rules",
        characters="Characters",
        locations="Locations",
        plot_timeline="Timeline",
    )
    chapter_plan = generators.chapter_plan().chapter_plan
    chapter_artifacts = generate_chapter_artifacts(
        generators,
        world_bible,
        chapter_plan,
        approve,
    )
    artifacts = build_artifacts(
        "Idea",
        "QA",
        "Premise",
        spine,
        SimpleNamespace(
            location_needs=LocationNeeds(needs=["A refuge"]), world_bible=world_bible
        ),
        chapter_plan,
        chapter_artifacts,
    )

    output = render_alternate_story_output(artifacts)

    assert "## Core Premise" in output
    assert "## Story Spine" in output
    assert "## World Bible" in output
    assert "## Chapter Plan" in output
    assert "## Chapter Drafting Artifacts" in output
    assert "## Final Story" in output
