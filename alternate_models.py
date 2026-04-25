"""Typed artifacts for the alternate interactive story pipeline."""

from pydantic import BaseModel, Field

from world_bible import WorldBible


class StorySpine(BaseModel):
    """Structured narrative movement for planning, not world building."""

    setup: str = Field(description="The initial story state and normal world.")
    disruption: str = Field(description="The event that breaks the normal world.")
    escalation: str = Field(description="How conflict and stakes intensify.")
    crisis: str = Field(description="The hardest irreversible choice or low point.")
    climax: str = Field(description="The decisive confrontation or resolution action.")
    resolution: str = Field(description="The resulting new state of the story world.")

    @property
    def full_text(self) -> str:
        """Render the spine as markdown for review and saving."""
        return (
            f"### Setup\n{self.setup}\n\n"
            f"### Disruption\n{self.disruption}\n\n"
            f"### Escalation\n{self.escalation}\n\n"
            f"### Crisis\n{self.crisis}\n\n"
            f"### Climax\n{self.climax}\n\n"
            f"### Resolution\n{self.resolution}"
        )


class LocationNeeds(BaseModel):
    """Compact character-derived location requirements."""

    needs: list[str] = Field(
        description="Location needs implied by character roles and conflicts.",
    )

    @property
    def full_text(self) -> str:
        """Render location needs as a markdown list."""
        return "\n".join(f"- {need}" for need in self.needs)


class ChapterPlanEntry(BaseModel):
    """Approved event plan for one chapter."""

    number: int = Field(description="One-based chapter number.")
    title: str = Field(description="Working chapter title.")
    purpose: str = Field(description="Narrative job this chapter performs.")
    beats: list[str] = Field(description="Ordered events that must happen.")

    @property
    def full_text(self) -> str:
        """Render the chapter plan entry as markdown."""
        beats_text = "\n".join(f"- {beat}" for beat in self.beats)
        return (
            f"### Chapter {self.number}: {self.title}\n"
            f"Purpose: {self.purpose}\n\n"
            f"{beats_text}"
        )


class ChapterEnhancement(BaseModel):
    """Prose-only enhancement guidance for one approved chapter plan."""

    pacing: str = Field(description="Pacing guidance that preserves the beats.")
    tension: str = Field(description="Tension or suspense guidance.")
    imagery: str = Field(description="Imagery and sensory guidance.")
    theme: str = Field(description="Theme or emotional resonance guidance.")

    @property
    def full_text(self) -> str:
        """Render enhancement guidance as markdown."""
        return (
            f"- Pacing: {self.pacing}\n"
            f"- Tension: {self.tension}\n"
            f"- Imagery: {self.imagery}\n"
            f"- Theme: {self.theme}"
        )


class ChapterDraft(BaseModel):
    """Generated prose for one chapter."""

    title: str = Field(description="Final chapter title.")
    chapter_text: str = Field(description="Full prose for the chapter.")

    @property
    def full_text(self) -> str:
        """Render chapter prose as markdown."""
        return f"### {self.title}\n\n{self.chapter_text}"


class AlternateStoryArtifacts(BaseModel):
    """Artifacts collected by the alternate story pipeline."""

    idea: str
    qa_text: str
    core_premise: str
    spine: StorySpine
    location_needs: LocationNeeds
    world_bible: WorldBible
    chapter_plan: list[ChapterPlanEntry]
    enhancements: list[ChapterEnhancement]
    random_details: list[str]
    chapter_summaries: list[str]
    chapters: list[ChapterDraft]

    @property
    def final_story(self) -> str:
        """Render the approved chapter drafts as one story."""
        rendered = []
        for index, chapter in enumerate(self.chapters, start=1):
            rendered.append(
                f"### Chapter {index}: {chapter.title}\n\n{chapter.chapter_text}"
            )
        return "\n\n".join(rendered)
