"""Alternate story generation pipeline: Architect → Director → Scripter → Writer.

Module A (The Architect): Vague Idea → World Bible + 3-Act Outline
Module B (The Director): Act + World Bible → 4 Sequences
Module C (The Scripter): Sequence + World Bible → Beats for 3 Scenes
Module D (The Writer): Beats + World Bible → Prose Paragraphs
"""

import dspy
import logging
from typing import List
from pydantic import BaseModel, Field
from _compat import observe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ActOutline(BaseModel):
    act_number: int = Field(description="The act number (1, 2, or 3).")
    title: str = Field(description="A short title for the act.")
    summary: str = Field(description="A 2-3 sentence summary of what happens in this act.")


class Sequence(BaseModel):
    sequence_number: int = Field(description="The sequence number (1-4) within the act.")
    title: str = Field(description="A short title for the sequence.")
    summary: str = Field(description="A 2-3 sentence summary of the sequence's purpose and events.")


class Beat(BaseModel):
    beat_summary: str = Field(description="A concise description of what happens in this beat.")
    emotion: str = Field(description="The dominant emotion or tone of this beat.")
    purpose: str = Field(description="Why this beat matters to the scene and overall story.")


class SceneBeats(BaseModel):
    scene_number: int = Field(description="The scene number (1-3) within the sequence.")
    scene_title: str = Field(description="A short title for the scene.")
    beats: List[Beat] = Field(description="The 3-5 narrative beats that make up this scene.")


# ---------------------------------------------------------------------------
# Module A — The Architect
# ---------------------------------------------------------------------------

class ArchitectWorldBibleSignature(dspy.Signature):
    """Generates a comprehensive World Bible from a vague story idea.
    The World Bible covers: world rules (magic, science, laws),
    characters (names, descriptions, relationships),
    locations (places, climates, relationships between them),
    and a high-level plot timeline."""
    idea: str = dspy.InputField(desc="The user's vague story idea.")
    world_bible: str = dspy.OutputField(
        desc="A comprehensive World Bible covering world rules, characters, locations, and plot timeline."
    )


class ArchitectOutlineSignature(dspy.Signature):
    """Generates a 3-Act story outline based on the idea and World Bible.
    Each act has a number, title, and 2-3 sentence summary."""
    idea: str = dspy.InputField(desc="The user's story idea.")
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    act_outlines: List[ActOutline] = dspy.OutputField(
        desc="Exactly 3 act outlines forming a complete 3-act structure."
    )


class Architect(dspy.Module):
    """Module A: Takes a vague idea and produces a World Bible + 3-Act Outline."""

    def __init__(self):
        super().__init__()
        self.generate_world_bible = dspy.ChainOfThought(ArchitectWorldBibleSignature)
        self.generate_outline = dspy.ChainOfThought(ArchitectOutlineSignature)

    @observe()
    def forward(self, idea: str):
        logger.info("Architect: generating World Bible...")
        wb_result = self.generate_world_bible(idea=idea)

        logger.info("Architect: generating 3-Act Outline...")
        outline_result = self.generate_outline(
            idea=idea, world_bible=wb_result.world_bible
        )

        return dspy.Prediction(
            world_bible=wb_result.world_bible,
            act_outlines=outline_result.act_outlines,
        )


# ---------------------------------------------------------------------------
# Module B — The Director
# ---------------------------------------------------------------------------

class DirectorSignature(dspy.Signature):
    """Breaks a single act into exactly 4 sequences. Each sequence is a
    self-contained dramatic movement within the act."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    act_outline: str = dspy.InputField(
        desc="The act number, title, and summary to break into sequences."
    )
    full_outline: str = dspy.InputField(
        desc="The full 3-act outline for context on the overall story arc."
    )
    sequences: List[Sequence] = dspy.OutputField(
        desc="Exactly 4 sequences for this act."
    )


class Director(dspy.Module):
    """Module B: Takes an act outline and produces 4 sequences."""

    def __init__(self):
        super().__init__()
        self.generate_sequences = dspy.ChainOfThought(DirectorSignature)

    @observe()
    def forward(self, world_bible: str, act_outline: str, full_outline: str):
        logger.info("Director: breaking act into 4 sequences...")
        result = self.generate_sequences(
            world_bible=world_bible,
            act_outline=act_outline,
            full_outline=full_outline,
        )
        return dspy.Prediction(sequences=result.sequences)


# ---------------------------------------------------------------------------
# Module C — The Scripter
# ---------------------------------------------------------------------------

class ScripterSignature(dspy.Signature):
    """Generates the beats for exactly 3 scenes within a single sequence.
    Each scene has 3-5 beats describing what happens, the emotion, and purpose."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    act_outline: str = dspy.InputField(desc="The act this sequence belongs to.")
    sequence_summary: str = dspy.InputField(
        desc="The sequence number, title, and summary to break into scenes."
    )
    previous_context: str = dspy.InputField(
        desc="Brief summary of everything that happened before this sequence."
    )
    scene_beats: List[SceneBeats] = dspy.OutputField(
        desc="Exactly 3 scenes, each with 3-5 narrative beats."
    )


class Scripter(dspy.Module):
    """Module C: Takes a sequence and produces beats for 3 scenes."""

    def __init__(self):
        super().__init__()
        self.generate_scene_beats = dspy.ChainOfThought(ScripterSignature)

    @observe()
    def forward(self, world_bible: str, act_outline: str,
                sequence_summary: str, previous_context: str):
        logger.info("Scripter: generating beats for 3 scenes...")
        result = self.generate_scene_beats(
            world_bible=world_bible,
            act_outline=act_outline,
            sequence_summary=sequence_summary,
            previous_context=previous_context,
        )
        return dspy.Prediction(scene_beats=result.scene_beats)


# ---------------------------------------------------------------------------
# Module D — The Writer
# ---------------------------------------------------------------------------

class WriterSignature(dspy.Signature):
    """Writes vivid, immersive prose for a single scene based on its beats.
    The output should include dialogue, description, and internal character
    thoughts as appropriate. The prose should flow naturally and read like
    a published novel."""
    world_bible: str = dspy.InputField(desc="The comprehensive World Bible.")
    scene_title: str = dspy.InputField(desc="The title of this scene.")
    beats: str = dspy.InputField(
        desc="The narrative beats to expand into full prose."
    )
    previous_context: str = dspy.InputField(
        desc="Brief summary of everything that happened before this scene."
    )
    scene_prose: str = dspy.OutputField(
        desc="Full, vivid prose for the scene (multiple paragraphs with dialogue and description)."
    )


class Writer(dspy.Module):
    """Module D: Takes beats and produces actual prose paragraphs."""

    def __init__(self):
        super().__init__()
        self.write_scene = dspy.ChainOfThought(WriterSignature)

    @observe()
    def forward(self, world_bible: str, scene_title: str,
                beats: str, previous_context: str):
        logger.info("Writer: producing prose for scene '%s'...", scene_title)
        result = self.write_scene(
            world_bible=world_bible,
            scene_title=scene_title,
            beats=beats,
            previous_context=previous_context,
        )
        return dspy.Prediction(scene_prose=result.scene_prose)


# ---------------------------------------------------------------------------
# Orchestrator — runs the full A → B → C → D pipeline
# ---------------------------------------------------------------------------

class AlternateStoryOrchestrator(dspy.Module):
    """Runs the full hierarchical pipeline:
    Architect → Director → Scripter → Writer

    Produces a complete story by iterating:
    - 3 acts (from Architect)
    - 4 sequences per act (from Director)
    - 3 scenes per sequence (from Scripter)
    - Prose per scene (from Writer)
    = 36 scenes total
    """

    def __init__(self):
        super().__init__()
        self.architect = Architect()
        self.director = Director()
        self.scripter = Scripter()
        self.writer = Writer()

    def _format_act(self, act: ActOutline) -> str:
        return f"Act {act.act_number}: {act.title} — {act.summary}"

    def _format_sequence(self, seq: Sequence) -> str:
        return f"Sequence {seq.sequence_number}: {seq.title} — {seq.summary}"

    def _format_beats(self, scene: SceneBeats) -> str:
        lines = []
        for b in scene.beats:
            lines.append(f"- {b.beat_summary} [{b.emotion}] ({b.purpose})")
        return "\n".join(lines)

    @observe()
    def forward(self, idea: str):
        # --- Module A: Architect ---
        arch_result = self.architect(idea=idea)
        world_bible = arch_result.world_bible
        act_outlines = arch_result.act_outlines

        full_outline = "\n".join(self._format_act(a) for a in act_outlines)
        logger.info("Architect complete: %d acts, world bible length=%d",
                     len(act_outlines), len(world_bible))

        full_story = ""
        running_context = ""
        scene_counter = 0

        for act in act_outlines:
            act_str = self._format_act(act)
            logger.info("Processing %s", act_str[:60])

            # --- Module B: Director ---
            dir_result = self.director(
                world_bible=world_bible,
                act_outline=act_str,
                full_outline=full_outline,
            )
            sequences = dir_result.sequences

            full_story += f"\n\n# Act {act.act_number}: {act.title}\n"

            for seq in sequences:
                seq_str = self._format_sequence(seq)
                logger.info("  Processing %s", seq_str[:60])

                # --- Module C: Scripter ---
                script_result = self.scripter(
                    world_bible=world_bible,
                    act_outline=act_str,
                    sequence_summary=seq_str,
                    previous_context=running_context or "This is the beginning of the story.",
                )
                scene_beats_list = script_result.scene_beats

                for scene in scene_beats_list:
                    scene_counter += 1
                    beats_str = self._format_beats(scene)

                    # --- Module D: Writer ---
                    writer_result = self.writer(
                        world_bible=world_bible,
                        scene_title=scene.scene_title,
                        beats=beats_str,
                        previous_context=running_context or "This is the beginning of the story.",
                    )

                    full_story += f"\n\n## Scene {scene_counter}: {scene.scene_title}\n\n"
                    full_story += writer_result.scene_prose

                    # Update running context
                    running_context += (
                        f"Scene {scene_counter} ({scene.scene_title}): "
                        f"{'; '.join(b.beat_summary for b in scene.beats)}\n"
                    )

        logger.info("Story complete: %d scenes generated.", scene_counter)

        return dspy.Prediction(
            world_bible=world_bible,
            act_outlines=act_outlines,
            full_outline=full_outline,
            story=full_story.strip(),
            scene_count=scene_counter,
        )
