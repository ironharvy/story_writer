"""Tests for the alternate story pipeline: Phase 1 → Phase 2 → Phase 3."""

import dspy
import os
import argparse
import logging
import coloredlogs
import pytest
from dotenv import load_dotenv
from alternate_story_modules import (
    Architect,
    Director,
    Scripter,
    Writer,
    AlternateStoryOrchestrator,
    Foundation,
    Spine,
    WorldRule,
    ActBreakdown,
    Sequence,
    Scene,
    SceneBeat,
)

load_dotenv()


class AlternateMockLM(dspy.LM):
    """Mock LM tuned for the 3-phase alternate pipeline's signatures."""

    def __init__(self):
        super().__init__(model="mock")

    def __call__(self, prompt=None, messages=None, **kwargs):
        content = prompt if prompt else str(messages)

        # Match on DSPy signature docstrings — most reliable discriminator.

        # Phase 3b — Writer (check before Scripter since both mention "scene")
        if "Writes vivid, immersive prose" in content:
            return [self._scene_prose_response()]

        # Phase 3a — Scripter: Scenes
        if "Generates detailed scenes for a single sequence" in content:
            return [self._scenes_response()]

        # Phase 3a — Scripter: Sequences
        if "Breaks a single act into sequences" in content:
            return [self._sequences_response()]

        # Phase 2 — Director
        if "Breaks the Foundation into a 3-Act macro structure" in content:
            return [self._act_breakdowns_response()]

        # Phase 1 — Architect
        if "Generates the story Foundation from a vague idea" in content:
            return [self._foundation_response()]

        # Fallback chain using field name heuristics
        if "foundation" in content and "logline" in content:
            return [self._foundation_response()]
        if "act_breakdowns" in content:
            return [self._act_breakdowns_response()]
        if "sequences" in content:
            return [self._sequences_response()]
        if "scenes" in content and "beats" in content:
            return [self._scenes_response()]
        if "scene_prose" in content:
            return [self._scene_prose_response()]

        return ["Mock response"]

    @staticmethod
    def _foundation_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "foundation": {'
            '"logline": "A genius high school student finds a supernatural notebook '
            'that kills anyone whose name is written in it; he uses it to cleanse '
            'the world of criminals while playing a cat-and-mouse game with a '
            'world-class detective.", '
            '"spine": {'
            '"internal_want": "Light wants to be a God of the new world (Justice/Ego).", '
            '"external_need": "Light needs to avoid being caught and maintain secrecy (Survival)."'
            '}, '
            '"world_rules": ['
            '{"rule_number": 1, "description": "You must have the person\'s face in mind when writing their name."}, '
            '{"rule_number": 2, "description": "Death occurs in 40 seconds after the name is written."}, '
            '{"rule_number": 3, "description": "The cause of death can be specified within 6 minutes and 40 seconds."}'
            '], '
            '"observer": "Ryuk, a Shinigami (death god) who dropped the notebook into the human world out of boredom."'
            '}}\n```'
        )

    @staticmethod
    def _act_breakdowns_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "act_breakdowns": ['
            '{"act_number": 1, "phase": "Setup", "title": "The God Complex Awakens", '
            '"summary": "Light finds the Death Note, kills his first criminal, and decides to become Kira. '
            'The world notices the pattern of criminal deaths.", '
            '"dramatic_marker": "Inciting Incident: L broadcasts globally and challenges Kira, narrowing his location to Japan."},'
            '{"act_number": 2, "phase": "Confrontation", "title": "The Mind Game", '
            '"summary": "Light joins the investigation team to get close to L. He eliminates FBI agents and Naomi Misora. '
            'Light and L meet face-to-face at the university.", '
            '"dramatic_marker": "Midpoint: Light and L meet face-to-face, each suspecting the other."},'
            '{"act_number": 3, "phase": "Resolution", "title": "Checkmate", '
            '"summary": "The Second Kira (Misa) appears. Light manipulates the Shinigami Rem into sacrificing herself. '
            'Light successfully eliminates L.", '
            '"dramatic_marker": "Climax: Light orchestrates L\'s death through Rem, becoming the unchallenged Kira."}'
            ']}\n```'
        )

    @staticmethod
    def _sequences_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "sequences": ['
            '{"sequence_number": 1, "title": "The Discovery", '
            '"summary": "Light finds the Death Note in the school yard and tests it on a criminal he sees on TV."},'
            '{"sequence_number": 2, "title": "The First Kills", '
            '"summary": "Light begins systematically killing criminals, establishing the Kira pattern."},'
            '{"sequence_number": 3, "title": "The Challenge", '
            '"summary": "L appears on television and challenges Kira, narrowing the search to the Kanto region."}'
            ']}\n```'
        )

    @staticmethod
    def _scenes_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "scenes": ['
            '{"scene_number": 1, "title": "The Train Station Death", '
            '"location": "Yamanote Line Train", '
            '"characters": ["Light Yagami", "Raye Penber"], '
            '"goal": "Light must force Raye to write the names of all other FBI agents without Raye seeing Light\'s face.", '
            '"beats": ['
            '{"label": "Input", "description": "Light enters the train and stands behind Raye Penber."}, '
            '{"label": "Action", "description": "Light speaks to Raye, proving he knows Raye is FBI."}, '
            '{"label": "Conflict", "description": "Raye tries to turn around; Light threatens to kill everyone on the train."}, '
            '{"label": "Climax", "description": "Light hands Raye a file with a Death Note page inside and tells him to write the names."}, '
            '{"label": "Resolution", "description": "Raye leaves the train and dies of a heart attack as the doors close."}'
            ']},'
            '{"scene_number": 2, "title": "Naomi\'s Investigation", '
            '"location": "Tokyo Streets", '
            '"characters": ["Light Yagami", "Naomi Misora"], '
            '"goal": "Light must prevent Naomi from reaching L with her critical deduction about Kira.", '
            '"beats": ['
            '{"label": "Input", "description": "Naomi approaches the task force building, carrying evidence."}, '
            '{"label": "Action", "description": "Light intercepts Naomi and pretends to be an ally."}, '
            '{"label": "Conflict", "description": "Naomi is suspicious but Light earns her trust by sharing insider details."}, '
            '{"label": "Climax", "description": "Light convinces Naomi to reveal her real name."}, '
            '{"label": "Resolution", "description": "Naomi walks away in a daze and is never seen again."}'
            ']}]}\n```'
        )

    @staticmethod
    def _scene_prose_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", '
            '"scene_prose": "The Yamanote Line train swayed gently as it pulled away from '
            'Shinjuku Station. Light Yagami stood near the rear door, his school bag slung '
            'over one shoulder, eyes fixed on the back of the man in the dark suit three '
            'seats ahead.\\n\\n'
            'Raye Penber. FBI. One of twelve agents sent to investigate the Kira case in '
            'Japan. Light had known about the tail for three days now — ever since Ryuk had '
            'casually mentioned someone was following him.\\n\\n'
            '\\"I know who you are,\\" Light said quietly, stepping closer. His voice was '
            'barely audible above the rattle of the train.\\n\\n'
            'Raye stiffened but did not turn around. \\"I don\'t know what you\'re talking '
            'about.\\"\\n\\n'
            '\\"Turn around and everyone on this train dies.\\" Light\'s tone was calm, '
            'almost conversational. \\"I have a piece of the Death Note in my watch. '
            'One wrong move and I write a name.\\""}\n```'
        )


logger = logging.getLogger(__name__)


def configure_logging(verbosity: int = 0, log_file: str | None = None):
    level_map = {0: logging.INFO, 1: logging.INFO, 2: logging.DEBUG, 3: logging.DEBUG}
    level = level_map.get(verbosity, logging.DEBUG)
    log_format = "%(asctime)s.%(msecs)03d %(levelname)s [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    coloredlogs.install(level=level, fmt=log_format, datefmt=date_format)

    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logging.getLogger().addHandler(file_handler)

    logger.setLevel(level)
    _http_loggers = ("httpx", "httpcore", "urllib3", "requests")
    _llm_loggers = ("litellm", "dspy", "langfuse", "openai", "anthropic")

    if verbosity <= 1:
        for name in _http_loggers + _llm_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    elif verbosity == 2:
        for name in _llm_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)
        for name in _http_loggers:
            logging.getLogger(name).setLevel(logging.WARNING)
    else:
        for name in _llm_loggers + _http_loggers:
            logging.getLogger(name).setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Integration test — full pipeline
# ---------------------------------------------------------------------------

def test_alternate_pipeline(
    model_name="mock",
    api_base="http://localhost:11434",
    api_key=None,
    output_dir=".tmp",
    max_tokens=1024,
    cache=True,
    memory_cache=True,
    cache_dir=None,
):
    kwargs = {"max_tokens": max_tokens}
    if api_base:
        kwargs["api_base"] = api_base

    if api_key is not None:
        kwargs["api_key"] = api_key
    elif "openai" in model_name.lower():
        env_key = os.environ.get("OPENAI_API_KEY")
        if env_key:
            kwargs["api_key"] = env_key

    logger.info("Testing alternate pipeline with model: %s...", model_name)

    dspy.configure_cache(
        enable_disk_cache=cache,
        enable_memory_cache=memory_cache,
        disk_cache_dir=cache_dir,
    )

    if model_name in ("mock", "test_mock"):
        lm = AlternateMockLM()
        dspy.configure(lm=lm)
    else:
        if "openai" in model_name.lower() and not kwargs.get("api_key"):
            logger.warning("OPENAI_API_KEY not found. Skipping integration test.")
            return

        lm = dspy.LM(model_name, cache=cache, **kwargs)
        dspy.configure(lm=lm)

    idea = (
        "A genius high school student finds a supernatural notebook that kills "
        "anyone whose name is written in it. He decides to use it to rid the "
        "world of criminals, but a brilliant detective begins hunting him."
    )

    # Run the full orchestrator
    orchestrator = AlternateStoryOrchestrator()
    result = orchestrator(idea=idea)

    logger.info("Foundation logline: %s", result.foundation.logline[:80])
    logger.info("Acts: %d", len(result.act_breakdowns))
    logger.info("Scenes generated: %d", result.scene_count)
    logger.info("Story length: %d characters", len(result.story))

    assert result.foundation.logline, "Logline should not be empty"
    assert result.foundation.spine.internal_want, "Internal want should not be empty"
    assert result.foundation.spine.external_need, "External need should not be empty"
    assert len(result.foundation.world_rules) > 0, "Should have at least one world rule"
    assert result.foundation.observer, "Observer should not be empty"
    assert len(result.act_breakdowns) == 3, f"Expected 3 acts, got {len(result.act_breakdowns)}"
    assert result.scene_count > 0, "Should have generated at least one scene"
    assert result.story.strip(), "Story should not be empty"

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "alternate_story_output.md")
    logger.info("Saving story output to %s...", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Alternate Pipeline Story Output\n\n")

        f.write("## Phase 1: Foundation\n\n")
        f.write(f"**Logline:** {result.foundation.logline}\n\n")
        f.write("**Spine:**\n")
        f.write(f"- Internal Want: {result.foundation.spine.internal_want}\n")
        f.write(f"- External Need: {result.foundation.spine.external_need}\n\n")
        f.write("**World Rules:**\n")
        for rule in result.foundation.world_rules:
            f.write(f"- Rule {rule.rule_number}: {rule.description}\n")
        f.write(f"\n**Observer:** {result.foundation.observer}\n\n")

        f.write("## Phase 2: Macro Structure\n\n")
        for act in result.act_breakdowns:
            f.write(f"### Act {act.act_number} ({act.phase}): {act.title}\n")
            f.write(f"{act.summary}\n")
            f.write(f"**Dramatic Marker:** {act.dramatic_marker}\n\n")

        f.write("## Phase 3: Full Story\n")
        f.write(f"{result.story}\n")

    logger.info("Alternate pipeline test passed!")


# ---------------------------------------------------------------------------
# Unit tests — Pydantic models
# ---------------------------------------------------------------------------

def test_spine_model():
    spine = Spine.model_validate({
        "internal_want": "Wants to be a god",
        "external_need": "Must avoid capture",
    })
    assert spine.internal_want == "Wants to be a god"
    assert spine.external_need == "Must avoid capture"


def test_world_rule_model():
    rule = WorldRule.model_validate({
        "rule_number": 1,
        "description": "Must know the face",
    })
    assert rule.rule_number == 1


def test_foundation_model():
    foundation = Foundation.model_validate({
        "logline": "A student finds a deadly notebook.",
        "spine": {
            "internal_want": "Wants justice",
            "external_need": "Must survive",
        },
        "world_rules": [
            {"rule_number": 1, "description": "Must know the face."},
            {"rule_number": 2, "description": "Death in 40 seconds."},
        ],
        "observer": "Ryuk the Shinigami",
    })
    assert foundation.logline
    assert len(foundation.world_rules) == 2
    assert foundation.observer == "Ryuk the Shinigami"


def test_act_breakdown_model():
    act = ActBreakdown.model_validate({
        "act_number": 1,
        "phase": "Setup",
        "title": "The Beginning",
        "summary": "Everything starts here.",
        "dramatic_marker": "Inciting Incident: The notebook is found.",
    })
    assert act.act_number == 1
    assert act.phase == "Setup"
    assert act.dramatic_marker.startswith("Inciting")


def test_sequence_model():
    seq = Sequence.model_validate({
        "sequence_number": 2,
        "title": "Rising Action",
        "summary": "Tension builds.",
    })
    assert seq.sequence_number == 2
    assert seq.title == "Rising Action"


def test_scene_beat_model():
    beat = SceneBeat.model_validate({
        "label": "Conflict",
        "description": "Raye tries to turn around.",
    })
    assert beat.label == "Conflict"


def test_scene_model():
    scene = Scene.model_validate({
        "scene_number": 1,
        "title": "The Train Station Death",
        "location": "Yamanote Line Train",
        "characters": ["Light Yagami", "Raye Penber"],
        "goal": "Force Raye to write the names.",
        "beats": [
            {"label": "Input", "description": "Light enters the train."},
            {"label": "Action", "description": "Light reveals he knows."},
            {"label": "Conflict", "description": "Raye tries to resist."},
            {"label": "Climax", "description": "Light hands Raye the file."},
            {"label": "Resolution", "description": "Raye dies on the platform."},
        ],
    })
    assert len(scene.beats) == 5
    assert scene.characters == ["Light Yagami", "Raye Penber"]
    assert scene.location == "Yamanote Line Train"


# ---------------------------------------------------------------------------
# Unit tests — Agents with mock LM
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_lm_configured():
    """Configure DSPy with the AlternateMockLM for unit tests."""
    lm = AlternateMockLM()
    dspy.configure(lm=lm)
    return lm


def test_architect_produces_foundation(mock_lm_configured):
    _ = mock_lm_configured
    architect = Architect()
    result = architect(idea="A student finds a notebook that can kill people.")
    assert result.foundation.logline
    assert result.foundation.spine.internal_want
    assert result.foundation.spine.external_need
    assert len(result.foundation.world_rules) >= 1
    assert result.foundation.observer


def test_director_produces_three_acts(mock_lm_configured):
    _ = mock_lm_configured
    director = Director()
    result = director(
        foundation="Logline: A student finds a deadly notebook.\n"
                   "Spine:\n  Internal Want: God complex\n  External Need: Avoid capture\n"
                   "World Rules:\n  Rule 1: Must know the face\n"
                   "Observer: Ryuk",
    )
    assert len(result.act_breakdowns) == 3
    assert result.act_breakdowns[0].phase == "Setup"


def test_scripter_produces_sequences(mock_lm_configured):
    _ = mock_lm_configured
    scripter = Scripter()
    result = scripter(
        foundation="Mock foundation text.",
        act_breakdown="Act 1 (Setup): The Beginning — Light finds the notebook.",
        full_structure="Act 1: Setup\nAct 2: Confrontation\nAct 3: Resolution",
        previous_context="This is the beginning of the story.",
    )
    assert len(result.sequences) >= 2


def test_scripter_produces_scenes_with_beats(mock_lm_configured):
    _ = mock_lm_configured
    scripter = Scripter()
    result = scripter.generate_scene_beats(
        foundation="Mock foundation text.",
        act_breakdown="Act 1 (Setup): The Beginning — Light finds the notebook.",
        sequence_summary="Sequence 1: The Discovery — Light finds the Death Note.",
        previous_context="This is the beginning of the story.",
    )
    assert len(result.scenes) >= 1
    scene = result.scenes[0]
    assert scene.location
    assert scene.characters
    assert scene.goal
    assert len(scene.beats) >= 3
    # Verify beat structure
    labels = [b.label for b in scene.beats]
    assert "Input" in labels
    assert "Resolution" in labels


def test_writer_produces_prose(mock_lm_configured):
    _ = mock_lm_configured
    writer = Writer()
    result = writer(
        foundation="Mock foundation text.",
        scene_title="The Train Station Death",
        location="Yamanote Line Train",
        characters="Light Yagami, Raye Penber",
        goal="Force Raye to write the names.",
        beats="Input: Light enters the train.\nAction: Light reveals he knows.\n"
              "Conflict: Raye tries to resist.\nClimax: Light hands the file.\n"
              "Resolution: Raye dies.",
        previous_context="This is the beginning of the story.",
    )
    assert result.scene_prose
    assert len(result.scene_prose) > 10


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Alternate Story Pipeline (3-Phase)")
    parser.add_argument("--model", type=str, default=os.environ.get("MODEL", "mock"),
                        help="The language model to use. Defaults to MODEL env var or mock.")
    parser.add_argument("--llm-url", type=str, default=os.environ.get("LLM_URL", "http://localhost:11434"),
                        help="The custom API base URL. Defaults to LLM_URL env var or http://localhost:11434.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("API_KEY"),
                        help="The API key for the model. Defaults to API_KEY env var.")
    parser.add_argument("--output-dir", type=str, default=".tmp",
                        help="Path to save output to")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="The number of tokens for an LLM to output")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable DSPy disk cache.")
    parser.add_argument("--memory-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable/disable DSPy in-memory cache.")
    parser.add_argument("--cache-dir", type=str, default=os.environ.get("DSPY_CACHE_DIR"),
                        help="Override DSPy disk cache directory.")
    parser.add_argument("--log-file", type=str, default=os.environ.get("LOG_FILE"),
                        help="Path to write detailed logs.")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv full firehose.")

    args = parser.parse_args()
    configure_logging(verbosity=args.verbose, log_file=args.log_file)

    test_alternate_pipeline(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        cache=args.cache,
        memory_cache=args.memory_cache,
        cache_dir=args.cache_dir,
    )
