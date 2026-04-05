"""Tests for the alternate story pipeline: Architect → Director → Scripter → Writer."""

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
    ActOutline,
    Sequence,
    SceneBeats,
    Beat,
)

load_dotenv()


class AlternateMockLM(dspy.LM):
    """Mock LM tuned for the alternate pipeline's signatures."""

    def __init__(self):
        super().__init__(model="mock")

    def __call__(self, prompt=None, messages=None, **kwargs):
        content = prompt if prompt else str(messages)

        # Match on DSPy signature docstrings — these are the most reliable
        # discriminator since each signature has a unique description.

        # Module D — Writer (check before Scripter since both mention "scene")
        if "Writes vivid, immersive prose" in content:
            return [self._scene_prose_response()]

        # Module C — Scripter
        if "Generates the beats for exactly 3 scenes" in content:
            return [self._scene_beats_response()]

        # Module B — Director
        if "Breaks a single act into exactly 4 sequences" in content:
            return [self._sequences_response()]

        # Module A — Architect: Act Outlines (check before World Bible)
        if "Generates a 3-Act story outline" in content:
            return [self._act_outlines_response()]

        # Module A — Architect: World Bible
        if "Generates a comprehensive World Bible" in content:
            return [self._world_bible_response()]

        # Fallback chain using field name heuristics
        if "act_outlines" in content:
            return [self._act_outlines_response()]
        if "sequences" in content:
            return [self._sequences_response()]
        if "scene_beats" in content:
            return [self._scene_beats_response()]
        if "scene_prose" in content:
            return [self._scene_prose_response()]
        if "world_bible" in content:
            return [self._world_bible_response()]

        return ["Mock response"]

    @staticmethod
    def _world_bible_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", '
            '"world_bible": "### Rules\\nDemon-binding magic powered by faith. '
            'The Church controls all sanctioned exorcisms.\\n\\n'
            '### Characters\\nKael: orphan raised as a weapon. '
            'Archbishop Dران: corrupt leader.\\n\\n'
            '### Locations\\nThe Sanctum: fortified cathedral. '
            'The Ashlands: demon-scarred wasteland.\\n\\n'
            '### Timeline\\nYear 1: Kael found. Year 10: first mission. '
            'Year 15: discovers truth."}\n```'
        )

    @staticmethod
    def _act_outlines_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "act_outlines": ['
            '{"act_number": 1, "title": "The Weapon Forged", '
            '"summary": "Kael is raised by the Church as their ultimate weapon. '
            'He trains relentlessly and completes his first demon hunt."},'
            '{"act_number": 2, "title": "Cracks in the Faith", '
            '"summary": "Kael discovers the Church breeds demons for profit. '
            'His loyalty fractures as he uncovers the conspiracy."},'
            '{"act_number": 3, "title": "The Reckoning", '
            '"summary": "Overpowered and disillusioned, Kael turns against the Church '
            'and dismantles their corrupt empire."}'
            ']}\n```'
        )

    @staticmethod
    def _sequences_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "sequences": ['
            '{"sequence_number": 1, "title": "Awakening", '
            '"summary": "The protagonist is introduced in their ordinary world."},'
            '{"sequence_number": 2, "title": "Training", '
            '"summary": "Rigorous preparation reveals hidden strengths and costs."},'
            '{"sequence_number": 3, "title": "First Trial", '
            '"summary": "A dangerous mission tests everything learned so far."},'
            '{"sequence_number": 4, "title": "Turning Point", '
            '"summary": "A revelation changes the protagonist\'s understanding of their world."}'
            ']}\n```'
        )

    @staticmethod
    def _scene_beats_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", "scene_beats": ['
            '{"scene_number": 1, "scene_title": "The Cold Morning", "beats": ['
            '{"beat_summary": "Kael wakes before dawn in the stone dormitory.", '
            '"emotion": "isolation", '
            '"purpose": "Establish the protagonist\'s lonely routine."},'
            '{"beat_summary": "He recites the Litany of Binding from memory.", '
            '"emotion": "discipline", '
            '"purpose": "Show the Church\'s indoctrination."},'
            '{"beat_summary": "A senior cleric watches from the shadows, taking notes.", '
            '"emotion": "unease", '
            '"purpose": "Hint at surveillance and control."}'
            ']},'
            '{"scene_number": 2, "scene_title": "The Sparring Ring", "beats": ['
            '{"beat_summary": "Kael fights three older acolytes simultaneously.", '
            '"emotion": "intensity", '
            '"purpose": "Demonstrate his combat superiority."},'
            '{"beat_summary": "He holds back a killing blow at the last instant.", '
            '"emotion": "restraint", '
            '"purpose": "Show his humanity despite training."},'
            '{"beat_summary": "The weapons-master nods approval but his eyes are cold.", '
            '"emotion": "ambiguity", '
            '"purpose": "Foreshadow the Church viewing Kael as a tool."}'
            ']},'
            '{"scene_number": 3, "scene_title": "The Archive Whisper", "beats": ['
            '{"beat_summary": "Kael sneaks into the restricted archives after hours.", '
            '"emotion": "curiosity", '
            '"purpose": "Plant seeds of his eventual rebellion."},'
            '{"beat_summary": "He finds a half-burned ledger with strange entries.", '
            '"emotion": "suspicion", '
            '"purpose": "First concrete clue of the Church\'s corruption."},'
            '{"beat_summary": "Footsteps echo down the corridor and he barely escapes.", '
            '"emotion": "tension", '
            '"purpose": "Raise stakes and keep the reader hooked."}'
            ']}]}\n```'
        )

    @staticmethod
    def _scene_prose_response():
        return (
            '```json\n{"reasoning": "Mock reasoning", '
            '"scene_prose": "The first light of dawn had not yet breached the narrow '
            'windows of the dormitory when Kael opened his eyes. He lay still for a '
            'moment, listening to the breathing of the other acolytes, then swung his '
            'legs over the side of the stone cot and stood.\\n\\n'
            '\\"Another day in paradise,\\" he muttered to no one in particular.\\n\\n'
            'The cold bit into his bare feet as he crossed the flagstones to the washbasin. '
            'He splashed icy water on his face and stared at his reflection — hollow '
            'cheeks, sharp eyes, the faint scar across his jaw from last month\'s trial. '
            'The Church had made him into something precise and dangerous, and he was '
            'only beginning to understand what that meant."}\n```'
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
        "An unnamed child is raised by the Church as the ultimate weapon against demons. "
        "As the child grows he learns that the church itself is corrupt and breeds demons "
        "for controlled chaos. The church receives funding for protection and as such "
        "decides who should receive help. The child eventually becomes overpowered and "
        "turns against the Church."
    )

    # Run the full orchestrator
    orchestrator = AlternateStoryOrchestrator()
    result = orchestrator(idea=idea)

    logger.info("World Bible length: %d characters", len(result.world_bible))
    logger.info("Acts: %d", len(result.act_outlines))
    logger.info("Scenes generated: %d", result.scene_count)
    logger.info("Story length: %d characters", len(result.story))

    assert result.world_bible, "World Bible should not be empty"
    assert len(result.act_outlines) == 3, f"Expected 3 acts, got {len(result.act_outlines)}"
    assert result.scene_count > 0, "Should have generated at least one scene"
    assert result.story.strip(), "Story should not be empty"

    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "alternate_story_output.md")
    logger.info("Saving story output to %s...", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Alternate Pipeline Story Output\n\n")
        f.write("## World Bible\n")
        f.write(f"{result.world_bible}\n\n")
        f.write("## 3-Act Outline\n")
        for act in result.act_outlines:
            f.write(f"### Act {act.act_number}: {act.title}\n")
            f.write(f"{act.summary}\n\n")
        f.write("## Full Story\n")
        f.write(f"{result.story}\n")

    logger.info("Alternate pipeline test passed!")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_act_outline_model():
    act = ActOutline.model_validate({
        "act_number": 1,
        "title": "The Beginning",
        "summary": "Everything starts here.",
    })
    assert act.act_number == 1
    assert act.title == "The Beginning"


def test_sequence_model():
    seq = Sequence.model_validate({
        "sequence_number": 2,
        "title": "Rising Action",
        "summary": "Tension builds.",
    })
    assert seq.sequence_number == 2
    assert seq.title == "Rising Action"


def test_beat_model():
    beat = Beat.model_validate({
        "beat_summary": "Hero draws sword.",
        "emotion": "determination",
        "purpose": "Show resolve.",
    })
    assert beat.emotion == "determination"


def test_scene_beats_model():
    scene = SceneBeats.model_validate({
        "scene_number": 1,
        "scene_title": "Opening",
        "beats": [
            {"beat_summary": "Dawn breaks.", "emotion": "calm", "purpose": "Set the tone."},
            {"beat_summary": "A scream.", "emotion": "shock", "purpose": "Inciting incident."},
        ],
    })
    assert len(scene.beats) == 2
    assert scene.scene_title == "Opening"


@pytest.fixture
def mock_lm_configured():
    """Configure DSPy with the AlternateMockLM for unit tests."""
    lm = AlternateMockLM()
    dspy.configure(lm=lm)
    return lm


def test_architect_produces_world_bible_and_acts(mock_lm_configured):
    architect = Architect()
    result = architect(idea="A child raised as a weapon discovers the truth.")
    assert result.world_bible
    assert len(result.act_outlines) == 3


def test_director_produces_four_sequences(mock_lm_configured):
    director = Director()
    result = director(
        world_bible="Mock world bible.",
        act_outline="Act 1: The Beginning — The hero is introduced.",
        full_outline="Act 1: Beginning\nAct 2: Middle\nAct 3: End",
    )
    assert len(result.sequences) == 4


def test_scripter_produces_three_scenes(mock_lm_configured):
    scripter = Scripter()
    result = scripter(
        world_bible="Mock world bible.",
        act_outline="Act 1: The Beginning — The hero is introduced.",
        sequence_summary="Sequence 1: Awakening — The protagonist wakes.",
        previous_context="This is the beginning of the story.",
    )
    assert len(result.scene_beats) == 3


def test_writer_produces_prose(mock_lm_configured):
    writer = Writer()
    result = writer(
        world_bible="Mock world bible.",
        scene_title="The Cold Morning",
        beats="- Kael wakes before dawn. [isolation] (Establish routine.)",
        previous_context="This is the beginning of the story.",
    )
    assert result.scene_prose
    assert len(result.scene_prose) > 10


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Alternate Story Pipeline (A→B→C→D)")
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
