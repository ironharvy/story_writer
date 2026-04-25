"""Interactive CLI for the alternate step-by-step story pipeline."""

import logging
from argparse import Namespace
from collections.abc import Sequence

from rich.console import Console
from rich.prompt import Confirm, Prompt

from alternate_models import AlternateStoryArtifacts
from alternate_pipeline import (
    AlternateGenerators,
    build_alternate_generators,
    build_artifacts,
    generate_chapter_artifacts,
    generate_chapter_plan,
    generate_core_premise,
    generate_spine,
    generate_world_sections,
    render_chapter_plan,
    save_alternate_story_output,
)
from main import build_arg_parser, setup_runtime
from story_modules import QuestionWithAnswer


logger = logging.getLogger(__name__)
console = Console()


def ask_answers_for_questions(questions: Sequence[QuestionWithAnswer]) -> str:
    """Collect accepted or user-edited answers for generated questions."""
    qa_pairs: list[str] = []
    for index, item in enumerate(questions, start=1):
        console.print(f"\n[bold cyan]Question {index}:[/bold cyan] {item.question}")
        console.print(
            f"[bold green]Proposed Answer:[/bold green] {item.proposed_answer}"
        )
        answer = item.proposed_answer
        if not Confirm.ask("Accept this proposed answer?", default=True):
            answer = Prompt.ask("Enter your answer")
        qa_pairs.append(f"Q: {item.question}\nA: {answer}")
    return "\n\n".join(qa_pairs)


def render_artifact(value: object) -> str:
    """Render generated values for console review."""
    if hasattr(value, "full_text"):
        return str(getattr(value, "full_text"))
    if isinstance(value, list):
        return "\n\n".join(render_artifact(item) for item in value)
    return str(value)


def extract_artifact(result: object, field_name: str) -> object:
    """Extract a named DSPy result field for review."""
    return getattr(result, field_name)


def append_feedback(inputs: dict[str, object], feedback: str) -> dict[str, object]:
    """Append feedback to an existing accepted text input for regeneration."""
    updated = dict(inputs)
    target = choose_feedback_target(updated)
    previous = str(updated[target])
    updated[target] = f"{previous}\n\nUser feedback for regeneration:\n{feedback}"
    return updated


def choose_feedback_target(inputs: dict[str, object]) -> str:
    """Pick the safest existing text field to carry regeneration feedback."""
    preferred = [
        "qa_pairs",
        "world_qa",
        "core_premise",
        "spine",
        "world_bible",
        "current_chapter",
        "characters",
        "location_needs",
    ]
    for key in preferred:
        if isinstance(inputs.get(key), str):
            return key
    for key, value in inputs.items():
        if isinstance(value, str):
            return key
    raise ValueError("No string input available for feedback regeneration.")


def run_with_feedback(
    generator: object,
    inputs: dict[str, object],
    field_name: str,
) -> object:
    """Run a generator until the user approves its selected output field."""
    current_inputs = dict(inputs)
    while True:
        result = generator(**current_inputs)
        artifact = extract_artifact(result, field_name)
        console.print(f"\n[bold magenta]--- {field_name} ---[/bold magenta]")
        console.print(render_artifact(artifact))
        if Confirm.ask("Approve this artifact?", default=True):
            return result
        feedback = Prompt.ask("Provide feedback for regeneration")
        current_inputs = append_feedback(current_inputs, feedback)


def collect_initial_context(generators: AlternateGenerators, idea: str) -> str:
    """Generate premise clarification questions and collect approved answers."""
    console.print("\n[italic]Generating clarification questions...[/italic]")
    result = generators.question(idea=idea)
    return ask_answers_for_questions(result.questions_with_answers)


def collect_world_context(
    generators: AlternateGenerators,
    core_premise: str,
    qa_text: str,
) -> str:
    """Generate world-building questions and collect approved answers."""
    console.print("\n[italic]Generating world-building questions...[/italic]")
    questions = generators.world_questions(
        core_premise=core_premise,
        qa_pairs=qa_text,
    ).questions_with_answers
    if not questions:
        return ""
    return ask_answers_for_questions(questions)


def run_alternate_workflow(
    generators: AlternateGenerators,
    idea: str,
) -> AlternateStoryArtifacts:
    """Run the alternate interactive story workflow."""
    qa_text = collect_initial_context(generators, idea)
    core_premise = generate_core_premise(
        generators,
        idea,
        qa_text,
        run_with_feedback,
    )
    spine = generate_spine(generators, core_premise, qa_text, run_with_feedback)
    world_qa = collect_world_context(generators, core_premise, qa_text)
    world_sections = generate_world_sections(
        generators,
        core_premise,
        qa_text,
        spine,
        world_qa,
        run_with_feedback,
    )
    chapter_plan = generate_chapter_plan(
        generators,
        core_premise,
        spine,
        world_sections.world_bible,
        run_with_feedback,
    )
    console.print("\n[bold blue]Approved chapter plan:[/bold blue]")
    console.print(render_chapter_plan(chapter_plan))
    chapter_artifacts = generate_chapter_artifacts(
        generators,
        world_sections.world_bible,
        chapter_plan,
        run_with_feedback,
    )
    return build_artifacts(
        idea,
        qa_text,
        core_premise,
        spine,
        world_sections,
        chapter_plan,
        chapter_artifacts,
    )


def run_cli(args: Namespace) -> str:
    """Run the alternate CLI and save output."""
    setup_runtime(args)
    console.print("[bold magenta]Alternate AI DSPy Story Writer[/bold magenta]")
    idea = Prompt.ask(
        "\n[bold yellow]What is your initial story idea/prompt?[/bold yellow]"
    )
    artifacts = run_alternate_workflow(build_alternate_generators(), idea)
    output_path = save_alternate_story_output(args.output_dir, artifacts)
    console.print(f"\n[bold magenta]Saved to {output_path}[/bold magenta]")
    return output_path


def main() -> None:
    """Run the alternate interactive Story Writer CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
