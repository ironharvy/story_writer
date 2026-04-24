"""Structured world-bible model shared across the pipeline."""

from pydantic import BaseModel, Field


class WorldBible(BaseModel):
    """Discrete world-bible sections plus a markdown rendering helper."""

    rules: str = Field(description="Rules of the world and its governing systems.")
    characters: str = Field(description="Character bios and relationship context.")
    locations: str = Field(description="Key places and setting details.")
    plot_timeline: str = Field(description="Plot beats organized across the story arc.")

    @property
    def full_text(self) -> str:
        """Render the legacy markdown format used for display and export."""
        return (
            f"### Rules of the World\n{self.rules}\n\n"
            f"### Characters\n{self.characters}\n\n"
            f"### Locations\n{self.locations}\n\n"
            f"### Plot Timeline\n{self.plot_timeline}"
        )
