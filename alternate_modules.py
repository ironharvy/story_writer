"""DSPy modules for the alternate interactive story pipeline."""

from typing import List

import dspy

from _compat import observe
from alternate_models import (
    ChapterDraft,
    ChapterEnhancement,
    ChapterPlanEntry,
    LocationNeeds,
    StorySpine,
)
from story_modules import QuestionWithAnswer


class GenerateAlternateSpineSignature(dspy.Signature):
    """Generate a structured story spine from the approved premise."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    qa_pairs: str = dspy.InputField(desc="Approved clarification answers.")
    spine: StorySpine = dspy.OutputField(desc="Structured narrative spine.")


class AlternateSpineGenerator(dspy.Module):
    """Generate a structured story spine."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(GenerateAlternateSpineSignature)

    @observe()
    def forward(self, core_premise: str, qa_pairs: str) -> object:
        """Create a narrative spine from approved premise context."""
        return self.generate(core_premise=core_premise, qa_pairs=qa_pairs)


class GenerateAlternateWorldQuestionsSignature(dspy.Signature):
    """Generate world-building clarification questions."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    qa_pairs: str = dspy.InputField(desc="Approved premise clarification answers.")
    questions_with_answers: List[QuestionWithAnswer] = dspy.OutputField(
        desc="Up to 3 world-building questions with proposed answers.",
    )


class AlternateWorldQuestionGenerator(dspy.Module):
    """Generate world-bible clarification questions."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(GenerateAlternateWorldQuestionsSignature)

    @observe()
    def forward(self, core_premise: str, qa_pairs: str) -> object:
        """Generate world-building questions from approved premise context."""
        return self.generate(core_premise=core_premise, qa_pairs=qa_pairs)


class GenerateAlternateWorldRulesSignature(dspy.Signature):
    """Generate rules for the story world."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    qa_pairs: str = dspy.InputField(desc="Approved premise clarification answers.")
    world_qa: str = dspy.InputField(desc="Approved world-building answers.")
    world_rules: str = dspy.OutputField(desc="Rules and constraints of the world.")


class AlternateWorldRulesGenerator(dspy.Module):
    """Generate world rules."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAlternateWorldRulesSignature)

    @observe()
    def forward(self, core_premise: str, qa_pairs: str, world_qa: str) -> object:
        """Generate world rules from premise context."""
        return self.generate(
            core_premise=core_premise,
            qa_pairs=qa_pairs,
            world_qa=world_qa,
        )


class GenerateAlternateCharactersSignature(dspy.Signature):
    """Generate character profiles from premise and approved rules."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    qa_pairs: str = dspy.InputField(desc="Approved premise clarification answers.")
    world_rules: str = dspy.InputField(desc="Approved world rules.")
    characters: str = dspy.OutputField(desc="Character profiles and relationships.")


class AlternateCharactersGenerator(dspy.Module):
    """Generate character profiles."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAlternateCharactersSignature)

    @observe()
    def forward(self, core_premise: str, qa_pairs: str, world_rules: str) -> object:
        """Generate characters without using the story spine."""
        return self.generate(
            core_premise=core_premise,
            qa_pairs=qa_pairs,
            world_rules=world_rules,
        )


class GenerateLocationNeedsSignature(dspy.Signature):
    """Summarize location needs implied by characters."""

    characters: str = dspy.InputField(desc="Approved character profiles.")
    location_needs: LocationNeeds = dspy.OutputField(
        desc="Compact character-driven location requirements.",
    )


class LocationNeedsGenerator(dspy.Module):
    """Generate compact location needs from character profiles."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(GenerateLocationNeedsSignature)

    @observe()
    def forward(self, characters: str) -> object:
        """Summarize location needs without passing full bios downstream."""
        return self.generate(characters=characters)


class GenerateAlternateLocationsSignature(dspy.Signature):
    """Generate locations from premise, rules, and compact needs."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    world_rules: str = dspy.InputField(desc="Approved world rules.")
    location_needs: str = dspy.InputField(desc="Compact location requirements.")
    locations: str = dspy.OutputField(desc="Key locations and setting details.")


class AlternateLocationsGenerator(dspy.Module):
    """Generate location profiles."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAlternateLocationsSignature)

    @observe()
    def forward(
        self,
        core_premise: str,
        world_rules: str,
        location_needs: str,
    ) -> object:
        """Generate locations without receiving full character profiles."""
        return self.generate(
            core_premise=core_premise,
            world_rules=world_rules,
            location_needs=location_needs,
        )


class GenerateAlternateTimelineSignature(dspy.Signature):
    """Generate a plot timeline from approved story and world artifacts."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    spine: str = dspy.InputField(desc="Approved structured story spine.")
    world_rules: str = dspy.InputField(desc="Approved world rules.")
    characters: str = dspy.InputField(desc="Approved character profiles.")
    locations: str = dspy.InputField(desc="Approved location profiles.")
    plot_timeline: str = dspy.OutputField(desc="Plot timeline for the story.")


class AlternateTimelineGenerator(dspy.Module):
    """Generate the plot timeline."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAlternateTimelineSignature)

    @observe()
    def forward(
        self,
        core_premise: str,
        spine: str,
        world_rules: str,
        characters: str,
        locations: str,
    ) -> object:
        """Generate timeline using both premise and spine."""
        return self.generate(
            core_premise=core_premise,
            spine=spine,
            world_rules=world_rules,
            characters=characters,
            locations=locations,
        )


class GenerateAlternateChapterPlanSignature(dspy.Signature):
    """Generate chapters with ordered beats."""

    core_premise: str = dspy.InputField(desc="The confirmed core premise.")
    spine: str = dspy.InputField(desc="Approved story spine.")
    world_bible: str = dspy.InputField(desc="Approved world bible.")
    chapter_plan: List[ChapterPlanEntry] = dspy.OutputField(
        desc="Chapters with number, title, purpose, and ordered beats.",
    )


class AlternateChapterPlanGenerator(dspy.Module):
    """Generate structured chapter plans."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAlternateChapterPlanSignature)

    @observe()
    def forward(self, core_premise: str, spine: str, world_bible: str) -> object:
        """Generate chapters with fixed beats for prose drafting."""
        return self.generate(
            core_premise=core_premise,
            spine=spine,
            world_bible=world_bible,
        )


class GenerateChapterEnhancementSignature(dspy.Signature):
    """Generate prose-only enhancement notes for a chapter."""

    world_bible: str = dspy.InputField(desc="Approved world bible.")
    chapter_plan: str = dspy.InputField(desc="Full approved chapter plan.")
    current_chapter: str = dspy.InputField(desc="Approved chapter beats.")
    enhancement: ChapterEnhancement = dspy.OutputField(
        desc="Guidance that improves prose without changing events.",
    )


class ChapterEnhancementGenerator(dspy.Module):
    """Generate chapter-specific enhancement guidance."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateChapterEnhancementSignature)

    @observe()
    def forward(
        self,
        world_bible: str,
        chapter_plan: str,
        current_chapter: str,
    ) -> object:
        """Generate prose-only chapter enhancement notes."""
        return self.generate(
            world_bible=world_bible,
            chapter_plan=chapter_plan,
            current_chapter=current_chapter,
        )


class GenerateAlternateRandomDetailSignature(dspy.Signature):
    """Generate an optional prose flourish for a chapter."""

    world_bible: str = dspy.InputField(desc="Approved world bible.")
    current_chapter: str = dspy.InputField(desc="Approved chapter beats.")
    random_detail: str = dspy.OutputField(
        desc="A vivid detail that does not change chapter events.",
    )


class AlternateRandomDetailGenerator(dspy.Module):
    """Generate a random chapter detail."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(GenerateAlternateRandomDetailSignature)

    @observe()
    def forward(self, world_bible: str, current_chapter: str) -> object:
        """Generate a prose flourish that preserves approved beats."""
        return self.generate(world_bible=world_bible, current_chapter=current_chapter)


class GenerateAlternateChapterDraftSignature(dspy.Signature):
    """Write prose from approved beats plus auxiliary prose guidance."""

    world_bible: str = dspy.InputField(desc="Approved world bible.")
    chapter_plan: str = dspy.InputField(desc="Full approved chapter plan.")
    current_chapter: str = dspy.InputField(desc="Approved chapter beats.")
    previous_summaries: str = dspy.InputField(desc="Approved prior chapter summaries.")
    enhancement: str = dspy.InputField(desc="Prose-only enhancement notes.")
    random_detail: str = dspy.InputField(desc="Optional prose flourish.")
    chapter: ChapterDraft = dspy.OutputField(desc="Chapter title and prose.")


class AlternateChapterDraftGenerator(dspy.Module):
    """Write one chapter from approved planning artifacts."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateAlternateChapterDraftSignature)

    @observe()
    def forward(
        self,
        world_bible: str,
        chapter_plan: str,
        current_chapter: str,
        previous_summaries: str,
        enhancement: str,
        random_detail: str,
    ) -> object:
        """Write chapter prose without changing approved beats."""
        return self.generate(
            world_bible=world_bible,
            chapter_plan=chapter_plan,
            current_chapter=current_chapter,
            previous_summaries=previous_summaries,
            enhancement=enhancement,
            random_detail=random_detail,
        )


class GenerateAlternateChapterSummarySignature(dspy.Signature):
    """Summarize approved chapter prose for rolling context."""

    current_chapter: str = dspy.InputField(desc="Approved chapter beats.")
    chapter_text: str = dspy.InputField(desc="Approved chapter prose.")
    chapter_summary: str = dspy.OutputField(desc="Factual 2-3 sentence summary.")


class AlternateChapterSummaryGenerator(dspy.Module):
    """Summarize one approved chapter."""

    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.Predict(GenerateAlternateChapterSummarySignature)

    @observe()
    def forward(self, current_chapter: str, chapter_text: str) -> object:
        """Summarize approved chapter prose."""
        return self.generate(
            current_chapter=current_chapter,
            chapter_text=chapter_text,
        )
