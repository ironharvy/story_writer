from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from story_writer.config import context_warning, estimated_input_tokens, routing_for
from story_writer.interactive import answer_questions, review_artifact
from story_writer.models.qa import RuleResult
from story_writer.models.run import Manifest, StageRecord, utc_now
from story_writer.models.story import (
    Chapter,
    ChapterPlan,
    Clarification,
    EnhancementGuide,
    Premise,
    Spine,
    StoryLength,
    WorldBible,
)
from story_writer.qa import detect_chapter, has_hard_failures
from story_writer.run_store import RunStore, slugify
from story_writer.stages import (
    GENERATION_STAGE_TYPES,
    EmbellishStage,
    ProseStage,
    StageBase,
    StageContext,
    build_stage,
)

ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "premise": Premise,
    "spine": Spine,
    "world_bible": WorldBible,
    "chapter_plan": ChapterPlan,
    "enhancement": EnhancementGuide,
}

PROFILE_MODELS = {
    "fast": "qwen3:latest",
    "quality": "gemma4:26b",
    "tiny": "gemma3:4b",
}

MAX_HARD_QA_REVISIONS = 2


class StrictQaError(RuntimeError):
    pass


def start_run(
    idea: str,
    *,
    length: StoryLength = StoryLength.SHORT,
    slug: str | None = None,
    embellish_probability: float = 0.15,
    allow_paid: bool = False,
    non_interactive: bool = False,
    strict: bool = False,
    model_override: str | None = None,
    profile: str | None = None,
    store: RunStore | None = None,
) -> str:
    store = store or RunStore()
    _validate_profile(profile)
    resolved_slug = slug or slugify(idea)
    manifest = Manifest(
        slug=resolved_slug,
        length=length.value,
        embellish_probability=embellish_probability,
        allow_paid=allow_paid,
    )
    store.create(resolved_slug, idea, manifest)
    resume_run(
        resolved_slug,
        non_interactive=non_interactive,
        strict=strict,
        model_override=model_override,
        profile=profile,
        store=store,
    )
    return resolved_slug


def resume_run(
    slug: str,
    *,
    non_interactive: bool = False,
    strict: bool = False,
    model_override: str | None = None,
    profile: str | None = None,
    store: RunStore | None = None,
) -> None:
    store = store or RunStore()
    _validate_profile(profile)
    manifest = store.read_manifest(slug)
    context = StageContext(
        slug=slug,
        idea=store.read_idea(slug),
        length=StoryLength(manifest.length),
        embellish_probability=manifest.embellish_probability,
        allow_paid=manifest.allow_paid,
    )
    override = model_override or (PROFILE_MODELS[profile] if profile else None)
    _load_existing_artifacts(context, store)
    _run_generation_stages(context, manifest, store, non_interactive, override)
    _run_chapters(context, manifest, store, override)
    _run_qa(context, store, strict)
    store.write_manifest(manifest)


def _run_generation_stages(
    context: StageContext,
    manifest: Manifest,
    store: RunStore,
    non_interactive: bool,
    model_override: str | None,
) -> None:
    for stage_type in GENERATION_STAGE_TYPES:
        name = stage_type.name
        if name in context.artifacts:
            continue
        stage = build_stage(stage_type, model_override)
        output = _run_stage(stage, context, manifest, store)
        if name == "clarify":
            output = answer_questions(output, non_interactive=non_interactive)
            store.write_json(
                context.slug, "clarify", [item.model_dump() for item in output]
            )
        if name in {"premise", "spine", "world_bible"}:
            replacement = review_artifact(
                title=name.replace("_", " ").title(),
                rendered=_render_for_review(output),
                regenerate=lambda feedback, stage=stage: _rerun_with_feedback(
                    stage,
                    context,
                    manifest,
                    store,
                    feedback,
                ),
                non_interactive=non_interactive,
            )
            if replacement is not None:
                output = replacement
                _write_stage_output(context.slug, name, output, store)
        context.artifacts[name] = output


def _run_chapters(
    context: StageContext,
    manifest: Manifest,
    store: RunStore,
    model_override: str | None,
) -> None:
    plan = context.artifacts["chapter_plan"]
    guide = context.artifacts["enhancement"]
    prior_summary = ""
    prior_chapters: list[Chapter] = []
    for planned in plan.chapters:
        artifact_name = f"chapter:{planned.number}"
        if store.has_artifact(context.slug, artifact_name):
            chapter = store.read_model(context.slug, artifact_name, Chapter)
            prior_chapters.append(chapter)
            prior_summary = _summarize_prior(prior_chapters)
            continue
        notes = next(
            item for item in guide.notes if item.chapter_number == planned.number
        )
        context.artifacts.update(
            {
                "current_chapter": planned,
                "current_enhancement": notes,
                "prior_summary": prior_summary,
                "prior_openings": _opening_context(prior_chapters),
                "qa_feedback": "",
            }
        )
        embellish_stage = EmbellishStage(
            routing_for("embellish", model_override=model_override)
        )
        embellishment = _run_stage(
            embellish_stage,
            context,
            manifest,
            store,
            stage_key=f"embellish:{planned.number:02d}",
            artifact_name=None,
        )
        context.artifacts["current_embellishment"] = embellishment
        prose_stage = ProseStage(routing_for("prose", model_override=model_override))
        chapter = _run_stage(
            prose_stage,
            context,
            manifest,
            store,
            stage_key=f"prose:{planned.number:02d}",
            artifact_name=None,
        )
        chapter = _revise_hard_qa_failures(
            chapter,
            prose_stage,
            context,
            manifest,
            store,
            prior_chapters,
        )
        store.write_model(context.slug, artifact_name, chapter)
        prior_chapters.append(chapter)
        prior_summary = _summarize_prior(prior_chapters)


def _run_qa(
    context: StageContext,
    store: RunStore,
    strict: bool,
) -> None:
    world_bible = context.artifacts["world_bible"]
    prior: list[Chapter] = []
    all_results: list[RuleResult] = []
    for path in sorted((store.run_dir(context.slug) / "chapters").glob("*.json")):
        chapter = Chapter.model_validate_json(path.read_text(encoding="utf-8"))
        results = detect_chapter(chapter, world_bible, prior)
        store.write_json(
            context.slug,
            f"qa:{chapter.number}",
            [result.model_dump() for result in results],
        )
        all_results.extend(results)
        prior.append(chapter)
    if strict and has_hard_failures(all_results):
        msg = "Strict QA failed with hard rule violations."
        raise StrictQaError(msg)


def _revise_hard_qa_failures(
    chapter: Chapter,
    prose_stage: ProseStage,
    context: StageContext,
    manifest: Manifest,
    store: RunStore,
    prior_chapters: list[Chapter],
) -> Chapter:
    world_bible = context.artifacts["world_bible"]
    revised = chapter
    for attempt in range(1, MAX_HARD_QA_REVISIONS + 1):
        results = detect_chapter(revised, world_bible, prior_chapters)
        hard_failures = [
            result
            for result in results
            if not result.passed and result.severity == "hard"
        ]
        if not hard_failures:
            break
        context.artifacts["qa_feedback"] = _qa_feedback(hard_failures)
        revised = _run_stage(
            prose_stage,
            context,
            manifest,
            store,
            stage_key=f"prose:{chapter.number:02d}:revision:{attempt:02d}",
            artifact_name=None,
        )
    remaining = [
        result
        for result in detect_chapter(revised, world_bible, prior_chapters)
        if not result.passed and result.rule_id == "R5"
    ]
    if remaining:
        revised = _remove_unknown_names(revised, remaining)
    context.artifacts["qa_feedback"] = ""
    return revised


def _run_stage(
    stage: StageBase,
    context: StageContext,
    manifest: Manifest,
    store: RunStore,
    *,
    stage_key: str | None = None,
    artifact_name: str | None = "",
) -> Any:
    key = stage_key or stage.name
    record = StageRecord(
        status="running",
        model=stage.config.model,
        provider=stage.config.provider,
        temperature=stage.config.temperature,
        max_tokens=stage.config.max_tokens,
        context_window=stage.config.context_window,
        estimated_input_tokens=estimated_input_tokens(stage.name, context.length),
        context_warning=context_warning(stage.name, context.length, stage.config),
        started_at=utc_now(),
    )
    manifest.stages[key] = record
    store.write_manifest(manifest)
    try:
        output = stage.run(context)
        record.status = "complete"
        record.finished_at = utc_now()
        if artifact_name is not None:
            _write_stage_output(
                context.slug,
                artifact_name or stage.name,
                output,
                store,
            )
        return output
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        record.finished_at = utc_now()
        store.write_manifest(manifest)
        raise
    finally:
        store.write_manifest(manifest)


def _rerun_with_feedback(
    stage: StageBase,
    context: StageContext,
    manifest: Manifest,
    store: RunStore,
    feedback: str,
) -> BaseModel:
    prior_feedback = context.feedback
    context.feedback = feedback
    try:
        return _run_stage(stage, context, manifest, store)
    finally:
        context.feedback = prior_feedback


def _write_stage_output(slug: str, name: str, output: Any, store: RunStore) -> None:
    if isinstance(output, BaseModel):
        store.write_model(slug, name, output)
    elif name == "clarify":
        store.write_json(slug, name, [item.model_dump() for item in output])


def _load_existing_artifacts(context: StageContext, store: RunStore) -> None:
    if store.has_artifact(context.slug, "clarify"):
        context.artifacts["clarify"] = [
            Clarification.model_validate(item)
            for item in store.read_json(context.slug, "clarify")
        ]
    for name, model in ARTIFACT_MODELS.items():
        if store.has_artifact(context.slug, name):
            context.artifacts[name] = store.read_model(context.slug, name, model)


def _render_for_review(output: Any) -> str:
    if isinstance(output, BaseModel):
        return output.model_dump_json(indent=2)
    return str(output)


def _summarize_prior(chapters: list[Chapter]) -> str:
    lines = [
        f"Chapter {chapter.number}: {chapter.title} - {chapter.prose[:500]}"
        for chapter in chapters[-2:]
    ]
    return "\n".join(lines)


def _opening_context(chapters: list[Chapter]) -> str:
    openings: list[str] = []
    for chapter in chapters[-5:]:
        opening = chapter.prose.strip().split("\n", maxsplit=1)[0].strip()
        if opening:
            openings.append(f"Chapter {chapter.number}: {opening[:500]}")
    return "\n".join(openings)


def _qa_feedback(results: list[RuleResult]) -> str:
    lines: list[str] = []
    for result in results:
        lines.append(
            f"{result.rule_id} ({result.severity}): {result.message}; "
            f"forbidden_spans={result.spans}"
        )
        if result.rule_id == "R5":
            lines.append(
                "Remove every forbidden span exactly. Do not replace forbidden "
                "names with another invented name. If a referenced person is "
                "unnamed in the world bible, keep the reference generic."
            )
    return "\n".join(lines)


def _remove_unknown_names(chapter: Chapter, results: list[RuleResult]) -> Chapter:
    prose = chapter.prose
    forbidden = {
        span
        for result in results
        for span in result.spans
        if span and span.lower() not in {"i"}
    }
    for span in forbidden:
        prose = re.sub(
            rf"\b{re.escape(span)}\b",
            "a name she could not read",
            prose,
        )
    return chapter.model_copy(update={"prose": prose})


def run_path(slug: str, store: RunStore | None = None) -> Path:
    return (store or RunStore()).run_dir(slug)


def _validate_profile(profile: str | None) -> None:
    if profile is not None and profile not in PROFILE_MODELS:
        allowed = ", ".join(sorted(PROFILE_MODELS))
        msg = f"profile must be one of: {allowed}"
        raise ValueError(msg)
