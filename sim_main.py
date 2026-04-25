"""CLI entry point for the multi-agent story simulation."""

import argparse
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Confirm, Prompt

from logging_config import setup_logging
from main import DSPyConfig, configure_dspy
from story_sim.engine import SimulationEngine
from story_sim.models import RoundRecord, SimulationState

load_dotenv()

logger = logging.getLogger(__name__)
console = Console()

MAX_CHARACTERS = 4


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for simulation mode."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Story Simulation (TRPG-style)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "openai/gpt-4o-mini"),
        help="The language model to use. Defaults to MODEL env var.",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default=os.environ.get("LLM_URL"),
        help="Custom API base URL. Defaults to LLM_URL env var.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("API_KEY"),
        help="API key for the model. Defaults to API_KEY env var.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help="Maximum tokens per LLM call. Defaults to 8000.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".tmp",
        help="Directory to save generated content. Defaults to '.tmp'.",
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
        default=os.environ.get("DSPY_CACHE_DIR"),
        help="Override DSPy disk cache directory.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.environ.get("LOG_FILE", ".tmp/sim_debug.log"),
        help="Path to write detailed logs.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Logging verbosity: -v INFO, -vv LLM debug, -vvv firehose.",
    )
    return parser


def setup_runtime(args: Namespace) -> None:
    """Configure logging and DSPy runtime from CLI arguments."""
    setup_logging(verbosity=args.verbose, log_file=args.log_file)
    config = DSPyConfig(
        model_name=args.model,
        api_base=args.llm_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        cache=args.cache,
        memory_cache=args.memory_cache,
        cache_dir=args.cache_dir,
    )
    configure_dspy(config)


def get_answers_for_questions(questions: Sequence[Any]) -> str:
    """Collect accepted or user-edited answers for GM questions."""
    qa_pairs = []
    for i, qa in enumerate(questions):
        console.print(f"\n[bold cyan]Question {i + 1}:[/bold cyan] {qa.question}")
        console.print(
            f"[bold green]Proposed Answer:[/bold green] {qa.proposed_answer}",
        )
        accept = Confirm.ask("Accept this proposed answer?")
        answer = qa.proposed_answer if accept else Prompt.ask("Enter your answer")
        qa_pairs.append(f"Q: {qa.question}\nA: {answer}")
    return "\n\n".join(qa_pairs)


def collect_user_input() -> tuple[str, int]:
    """Collect the story idea and character count from the user."""
    idea = Prompt.ask(
        "\n[bold yellow]What is your story idea?[/bold yellow]",
    )
    num_characters = int(
        Prompt.ask(
            "[bold yellow]How many characters? (2-4)[/bold yellow]",
            default="3",
        ),
    )
    num_characters = max(2, min(num_characters, MAX_CHARACTERS))
    return idea, num_characters


def run_setup_phase(
    engine: SimulationEngine,
    idea: str,
    num_characters: int,
) -> SimulationState:
    """Run GM setup: questions -> world -> characters -> plot outline."""
    console.print("\n[italic]GM is preparing clarifying questions...[/italic]")
    q_result = engine.gm_setup.generate_questions(
        idea=idea,
        num_characters=num_characters,
    )
    qa_context = get_answers_for_questions(q_result.questions)

    console.print("\n[italic]GM is building the world...[/italic]")
    state = engine.setup(
        idea=idea,
        qa_context=qa_context,
        num_characters=num_characters,
    )

    console.print("\n[bold green]--- World ---[/bold green]")
    console.print(f"[bold]Setting:[/bold] {state.world.setting}")
    console.print(f"[bold]Genre:[/bold] {state.world.genre}")
    console.print(f"[bold]Tone:[/bold] {state.world.tone}")

    console.print("\n[bold cyan]--- Characters ---[/bold cyan]")
    for char in state.characters:
        console.print(f"\n[bold]{char.name}[/bold] ({char.role})")
        console.print(f"  {char.personality}")
        console.print(f"  Goals: {', '.join(char.goals)}")

    console.print("\n[dim]Plot outline generated (GM-private).[/dim]")
    return state


def print_round_progress(record: RoundRecord) -> None:
    """Print a brief progress line for each completed round."""
    console.print(
        f"  [dim]Round {record.round_number} (Ch.{record.chapter_number}): "
        f"{record.resolution.pacing.value}[/dim]",
    )


def save_simulation_output(
    output_dir: str,
    story: str,
    state: SimulationState,
) -> str:
    """Write the generated story to a markdown file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sim_story_output.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Simulated Story\n\n")
        f.write(f"**Genre:** {state.world.genre}\n")
        f.write(f"**Tone:** {state.world.tone}\n")
        f.write(f"**Setting:** {state.world.setting}\n\n")
        f.write("## Characters\n\n")
        for char in state.characters:
            f.write(f"- **{char.name}** ({char.role}): {char.personality}\n")
        f.write(f"\n---\n\n{story}\n")

    return output_path


def main() -> None:
    """Run the multi-agent story simulation CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_runtime(args)

    console.print(
        "[bold magenta]Multi-Agent Story Simulation[/bold magenta]",
    )
    console.print(
        "[dim]TRPG-style story generation with GM, Characters, and Narrator[/dim]\n"
    )

    idea, num_characters = collect_user_input()
    engine = SimulationEngine()
    state = run_setup_phase(engine, idea, num_characters)

    Confirm.ask(
        "\nReady to begin the simulation? Press Enter to start",
        default=True,
        show_default=False,
    )

    console.print("\n[bold magenta]--- Simulation Running ---[/bold magenta]")
    story = engine.run(state, on_round=print_round_progress)

    console.print("\n[bold red]--- Generated Story ---[/bold red]")
    console.print(story)

    output_path = save_simulation_output(args.output_dir, story, state)
    console.print(
        f"\n[bold magenta]Simulation complete! "
        f"Story saved to {output_path}[/bold magenta]",
    )


if __name__ == "__main__":
    main()
