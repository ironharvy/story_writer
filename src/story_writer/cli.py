from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from story_writer.models.story import StoryLength
from story_writer.run_store import RunStore

app = typer.Typer(help="Local-first DSPy story generation pipeline.")
console = Console()
PROFILE_HELP = "Named model bundle: fast, quality, or tiny."


@app.command()
def new(
    idea: Annotated[str, typer.Option(help="Raw story idea.")],
    length: Annotated[StoryLength, typer.Option()] = StoryLength.SHORT,
    slug: Annotated[str | None, typer.Option()] = None,
    embellish_probability: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.15,
    non_interactive: Annotated[bool, typer.Option()] = False,
    allow_paid: Annotated[bool, typer.Option()] = False,
    strict: Annotated[bool, typer.Option()] = False,
    skip_qa_embeddings: Annotated[bool, typer.Option()] = False,
    model: Annotated[str | None, typer.Option()] = None,
    profile: Annotated[str | None, typer.Option(help=PROFILE_HELP)] = None,
) -> None:
    _ = skip_qa_embeddings
    _check_profile(profile)
    from story_writer.orchestrator import start_run

    resolved = start_run(
        idea,
        length=length,
        slug=slug,
        embellish_probability=embellish_probability,
        allow_paid=allow_paid,
        non_interactive=non_interactive,
        strict=strict,
        model_override=model,
        profile=profile,
    )
    console.print(f"Run complete: runs/{resolved}")


@app.command()
def resume(
    slug: str,
    non_interactive: Annotated[bool, typer.Option()] = False,
    strict: Annotated[bool, typer.Option()] = False,
    model: Annotated[str | None, typer.Option()] = None,
    profile: Annotated[str | None, typer.Option(help=PROFILE_HELP)] = None,
) -> None:
    _check_profile(profile)
    from story_writer.orchestrator import resume_run

    resume_run(
        slug,
        non_interactive=non_interactive,
        strict=strict,
        model_override=model,
        profile=profile,
    )
    console.print(f"Run complete: runs/{slug}")


@app.command()
def render(
    slug: str,
    fmt: Annotated[str, typer.Option()] = "md",
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    from story_writer.render import render_story

    if fmt not in {"md", "txt"}:
        raise typer.BadParameter("fmt must be md or txt")
    path = render_story(slug, RunStore(), fmt=fmt, out=out)
    console.print(str(path))


@app.command()
def inspect(
    slug: str,
    stage: Annotated[str | None, typer.Option()] = None,
) -> None:
    store = RunStore()
    if stage is None:
        console.print(store.manifest_path(slug).read_text(encoding="utf-8"))
        return
    path = store.artifact_path(slug, stage)
    data = json.loads(path.read_text(encoding="utf-8"))
    console.print_json(data=data)


def _check_profile(profile: str | None) -> None:
    if profile is not None and profile not in {"fast", "quality", "tiny"}:
        raise typer.BadParameter("profile must be fast, quality, or tiny")


if __name__ == "__main__":
    app()
