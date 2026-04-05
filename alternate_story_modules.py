"""Alternate story generation pipeline using a 3-phase architecture.

Phase 1 — The Foundation (System State):
    Architect Agent takes a vague idea and generates the "global variables":
    logline, spine (internal want + external need), and world bible rules.

Phase 2 — The Macro Structure (Control Flow):
    Director Agent takes the Foundation and breaks it into 3-Act structure
    with dramatic markers (inciting incident, midpoint, climax).

Phase 3 — The Granular Breakdown (Execution):
    Scripter Agent breaks acts into sequences, sequences into scenes.
    Writer Agent expands scene beats into full prose.
    Beats follow: Input → Action → Conflict → Climax → Resolution.
"""

import dspy
import logging
from typing import List
from pydantic import BaseModel, Field
from _compat import observe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1 Pydantic models — The Foundation
# ---------------------------------------------------------------------------

class Spine(BaseModel):
    internal_want: str = Field(
        description="What the protagonist consciously desires (ego, justice, love, power, etc.)."
    )
    external_need: str = Field(
        description="What the protagonist must do to survive or succeed in the external world."
    )


class WorldRule(BaseModel):
    rule_number: int = Field(description="Sequential rule number.")
    description: str = Field(description="A concise statement of the rule or constraint.")


class Foundation(BaseModel):
    logline: str = Field(
        description="A 1-2 sentence summary capturing protagonist, conflict, and stakes."
    )
    spine: Spine = Field(
        description="The core conflict expressed as internal want vs external need."
    )
    world_rules: List[WorldRule] = Field(
        description="The key rules, constraints, or laws that govern this story's world."
    )
    observer: str = Field(
        description="The neutral or antagonistic observer figure who watches the protagonist."
    )


# ---------------------------------------------------------------------------
# Phase 2 Pydantic models — The Macro Structure
# ---------------------------------------------------------------------------

class ActBreakdown(BaseModel):
    act_number: int = Field(description="The act number (1, 2, or 3).")
    phase: str = Field(description="The dramatic phase: Setup, Confrontation, or Resolution.")
    title: str = Field(description="A short title for the act.")
    summary: str = Field(description="A 2-3 sentence summary of what happens in this act.")
    dramatic_marker: str = Field(
        description=(
            "The key dramatic turning point for this act. "
            "Act 1: Inciting Incident. Act 2: Midpoint reversal. Act 3: Climax."
        )
    )


# ---------------------------------------------------------------------------
# Phase 3 Pydantic models — The Granular Breakdown
# ---------------------------------------------------------------------------

class Sequence(BaseModel):
    sequence_number: int = Field(description="The sequence number within the act.")
    title: str = Field(description="A short title for the sequence.")
    summary: str = Field(
        description="A series of scenes with a unified dramatic purpose."
    )


class SceneBeat(BaseModel):
    label: str = Field(
        description="The beat type: Input, Action, Conflict, Climax, or Resolution."
    )
    description: str = Field(description="What happens in this beat.")


class Scene(BaseModel):
    scene_number: int = Field(description="The scene number within the sequence.")
    title: str = Field(description="A short title for the scene.")
    location: str = Field(description="Where this scene takes place.")
    characters: List[str] = Field(description="Characters present in this scene.")
    goal: str = Field(
        description="What the protagonist must accomplish in this scene."
    )
    beats: List[SceneBeat] = Field(
        description=(
            "The logic steps of the scene following the structure: "
            "Input → Action → Conflict → Climax → Resolution."
        )
    )


# ---------------------------------------------------------------------------
# Phase 1 — The Architect
# ---------------------------------------------------------------------------

class FoundationSignature(dspy.Signature):
    """Generates the story Foundation from a vague idea.
    The Foundation includes: a logline (1-2 sentences capturing protagonist,
    conflict, and stakes), a spine (internal want vs external need), world
    rules (the key constraints/laws governing the world), and an observer
    figure who watches the protagonist."""
    idea: str = dspy.InputField(desc="The user's vague story idea.")
    foundation: Foundation = dspy.OutputField(
        desc="The complete Foundation: logline, spine, world_rules, and observer."
    )


class Architect(dspy.Module):
    """Phase 1: Takes a vague idea and produces the Foundation —
    logline, spine, world rules, and observer."""

    def __init__(self):
        super().__init__()
        self.generate_foundation = dspy.ChainOfThought(FoundationSignature)

    @observe()
    def forward(self, idea: str):
        logger.info("Architect: generating Foundation (logline, spine, world rules)...")
        result = self.generate_foundation(idea=idea)
        return dspy.Prediction(foundation=result.foundation)


# ---------------------------------------------------------------------------
# Phase 2 — The Director
# ---------------------------------------------------------------------------

class MacroStructureSignature(dspy.Signature):
    """Breaks the Foundation into a 3-Act macro structure.
    Each act has a phase (Setup/Confrontation/Resolution), title, summary,
    and a dramatic marker (Inciting Incident / Midpoint / Climax).
    The acts should form a complete dramatic arc."""
    foundation: str = dspy.InputField(
        desc="The Foundation: logline, spine, world rules, and observer."
    )
    act_breakdowns: List[ActBreakdown] = dspy.OutputField(
        desc="Exactly 3 act breakdowns forming the macro structure."
    )


class Director(dspy.Module):
    """Phase 2: Takes the Foundation and produces the 3-Act macro structure."""

    def __init__(self):
        super().__init__()
        self.generate_structure = dspy.ChainOfThought(MacroStructureSignature)

    @observe()
    def forward(self, foundation: str):
        logger.info("Director: generating 3-Act macro structure...")
        result = self.generate_structure(foundation=foundation)
        return dspy.Prediction(act_breakdowns=result.act_breakdowns)


# ---------------------------------------------------------------------------
# Phase 3a — The Scripter (sequences + scenes)
# ---------------------------------------------------------------------------

class SequenceSignature(dspy.Signature):
    """Breaks a single act into sequences. Each sequence is a self-contained
    dramatic movement — a series of scenes with a unified purpose."""
    foundation: str = dspy.InputField(desc="The Foundation for world context.")
    act_breakdown: str = dspy.InputField(
        desc="The act to break down, including phase, summary, and dramatic marker."
    )
    full_structure: str = dspy.InputField(
        desc="The full 3-act structure for overall story arc context."
    )
    previous_context: str = dspy.InputField(
        desc="Brief summary of everything that happened before this act."
    )
    sequences: List[Sequence] = dspy.OutputField(
        desc="The sequences for this act (typically 2-4)."
    )


class SceneSignature(dspy.Signature):
    """Generates detailed scenes for a single sequence.
    Each scene has a location, characters, goal, and structured beats
    following: Input → Action → Conflict → Climax → Resolution."""
    foundation: str = dspy.InputField(desc="The Foundation for world context.")
    act_breakdown: str = dspy.InputField(desc="The act this sequence belongs to.")
    sequence_summary: str = dspy.InputField(
        desc="The sequence to break into scenes."
    )
    previous_context: str = dspy.InputField(
        desc="Brief summary of everything that happened before this sequence."
    )
    scenes: List[Scene] = dspy.OutputField(
        desc="The scenes for this sequence (typically 2-4), each with structured beats."
    )


class Scripter(dspy.Module):
    """Phase 3a: Breaks acts into sequences, sequences into scenes with
    structured beats (Input → Action → Conflict → Climax → Resolution)."""

    def __init__(self):
        super().__init__()
        self.generate_sequences = dspy.ChainOfThought(SequenceSignature)
        self.generate_scenes = dspy.ChainOfThought(SceneSignature)

    @observe()
    def forward(self, foundation: str, act_breakdown: str,
                full_structure: str, previous_context: str):
        logger.info("Scripter: breaking act into sequences...")
        seq_result = self.generate_sequences(
            foundation=foundation,
            act_breakdown=act_breakdown,
            full_structure=full_structure,
            previous_context=previous_context,
        )
        return dspy.Prediction(sequences=seq_result.sequences)

    @observe()
    def generate_scene_beats(self, foundation: str, act_breakdown: str,
                             sequence_summary: str, previous_context: str):
        logger.info("Scripter: generating scenes with structured beats...")
        result = self.generate_scenes(
            foundation=foundation,
            act_breakdown=act_breakdown,
            sequence_summary=sequence_summary,
            previous_context=previous_context,
        )
        return dspy.Prediction(scenes=result.scenes)


# ---------------------------------------------------------------------------
# Phase 3b — The Writer (prose)
# ---------------------------------------------------------------------------

class WriterSignature(dspy.Signature):
    """Writes vivid, immersive prose for a single scene based on its structured
    beats. The output should include dialogue, description, and internal
    character thoughts as appropriate. The prose should flow naturally and
    read like a published novel. Each beat (Input, Action, Conflict, Climax,
    Resolution) should be woven into the narrative seamlessly."""
    foundation: str = dspy.InputField(desc="The Foundation for world context.")
    scene_title: str = dspy.InputField(desc="The title of this scene.")
    location: str = dspy.InputField(desc="Where this scene takes place.")
    characters: str = dspy.InputField(desc="Characters present in this scene.")
    goal: str = dspy.InputField(desc="What the protagonist must accomplish.")
    beats: str = dspy.InputField(
        desc="The structured beats (Input → Action → Conflict → Climax → Resolution)."
    )
    previous_context: str = dspy.InputField(
        desc="Brief summary of everything that happened before this scene."
    )
    scene_prose: str = dspy.OutputField(
        desc="Full, vivid prose for the scene (multiple paragraphs with dialogue and description)."
    )


class Writer(dspy.Module):
    """Phase 3b: Takes structured scene beats and produces prose paragraphs."""

    def __init__(self):
        super().__init__()
        self.write_scene = dspy.ChainOfThought(WriterSignature)

    @observe()
    def forward(self, foundation: str, scene_title: str, location: str,
                characters: str, goal: str, beats: str, previous_context: str):
        logger.info("Writer: producing prose for scene '%s'...", scene_title)
        result = self.write_scene(
            foundation=foundation,
            scene_title=scene_title,
            location=location,
            characters=characters,
            goal=goal,
            beats=beats,
            previous_context=previous_context,
        )
        return dspy.Prediction(scene_prose=result.scene_prose)


# ---------------------------------------------------------------------------
# Orchestrator — runs the full Phase 1 → 2 → 3 pipeline
# ---------------------------------------------------------------------------

class AlternateStoryOrchestrator(dspy.Module):
    """Runs the full 3-phase pipeline:

    Phase 1 (Foundation):  Idea → Logline + Spine + World Rules + Observer
    Phase 2 (Structure):   Foundation → 3-Act Breakdown
    Phase 3 (Execution):   Acts → Sequences → Scenes (with beats) → Prose
    """

    def __init__(self):
        super().__init__()
        self.architect = Architect()
        self.director = Director()
        self.scripter = Scripter()
        self.writer = Writer()

    def _format_foundation(self, foundation: Foundation) -> str:
        rules_str = "\n".join(
            f"  Rule {r.rule_number}: {r.description}" for r in foundation.world_rules
        )
        return (
            f"Logline: {foundation.logline}\n\n"
            f"Spine:\n"
            f"  Internal Want: {foundation.spine.internal_want}\n"
            f"  External Need: {foundation.spine.external_need}\n\n"
            f"World Rules:\n{rules_str}\n\n"
            f"Observer: {foundation.observer}"
        )

    def _format_act(self, act: ActBreakdown) -> str:
        return (
            f"Act {act.act_number} ({act.phase}): {act.title}\n"
            f"  Summary: {act.summary}\n"
            f"  Dramatic Marker: {act.dramatic_marker}"
        )

    def _format_sequence(self, seq: Sequence) -> str:
        return f"Sequence {seq.sequence_number}: {seq.title} — {seq.summary}"

    def _format_beats(self, scene: Scene) -> str:
        lines = []
        for b in scene.beats:
            lines.append(f"  {b.label}: {b.description}")
        return "\n".join(lines)

    @observe()
    def forward(self, idea: str):
        # --- Phase 1: Foundation ---
        logger.info("=== Phase 1: The Foundation ===")
        arch_result = self.architect(idea=idea)
        foundation = arch_result.foundation
        foundation_str = self._format_foundation(foundation)

        logger.info("Foundation complete: logline=%d chars, %d world rules",
                     len(foundation.logline), len(foundation.world_rules))

        # --- Phase 2: Macro Structure ---
        logger.info("=== Phase 2: The Macro Structure ===")
        dir_result = self.director(foundation=foundation_str)
        act_breakdowns = dir_result.act_breakdowns

        full_structure = "\n\n".join(self._format_act(a) for a in act_breakdowns)
        logger.info("Macro structure complete: %d acts", len(act_breakdowns))

        # --- Phase 3: Granular Breakdown ---
        logger.info("=== Phase 3: The Granular Breakdown ===")
        full_story = ""
        running_context = ""
        scene_counter = 0
        all_sequences = []
        all_scenes = []

        for act in act_breakdowns:
            act_str = self._format_act(act)
            logger.info("Processing %s", act_str.split("\n")[0])

            # Phase 3a: Act → Sequences
            script_result = self.scripter(
                foundation=foundation_str,
                act_breakdown=act_str,
                full_structure=full_structure,
                previous_context=running_context or "This is the beginning of the story.",
            )
            sequences = script_result.sequences
            all_sequences.extend(sequences)

            full_story += f"\n\n# Act {act.act_number}: {act.title}\n"

            for seq in sequences:
                seq_str = self._format_sequence(seq)
                logger.info("  Processing %s", seq_str[:60])

                # Phase 3a: Sequence → Scenes with beats
                scene_result = self.scripter.generate_scene_beats(
                    foundation=foundation_str,
                    act_breakdown=act_str,
                    sequence_summary=seq_str,
                    previous_context=running_context or "This is the beginning of the story.",
                )
                scenes = scene_result.scenes
                all_scenes.extend(scenes)

                for scene in scenes:
                    scene_counter += 1
                    beats_str = self._format_beats(scene)
                    characters_str = ", ".join(scene.characters)

                    # Phase 3b: Scene → Prose
                    writer_result = self.writer(
                        foundation=foundation_str,
                        scene_title=scene.title,
                        location=scene.location,
                        characters=characters_str,
                        goal=scene.goal,
                        beats=beats_str,
                        previous_context=running_context or "This is the beginning of the story.",
                    )

                    full_story += f"\n\n## Scene {scene_counter}: {scene.title}\n"
                    full_story += f"*{scene.location}*\n\n"
                    full_story += writer_result.scene_prose

                    # Update running context
                    beat_summary = "; ".join(b.description for b in scene.beats)
                    running_context += (
                        f"Scene {scene_counter} ({scene.title} at {scene.location}): "
                        f"{beat_summary}\n"
                    )

        logger.info("Story complete: %d scenes generated.", scene_counter)

        return dspy.Prediction(
            foundation=foundation,
            foundation_text=foundation_str,
            act_breakdowns=act_breakdowns,
            full_structure=full_structure,
            story=full_story.strip(),
            scene_count=scene_counter,
        )
