# Story Writer — Software Design Document (SDD)

This document describes *how* Story Writer is built. It is the implementation
counterpart to `SYSTEM_ANALYSIS.md` (which describes *what* and *why*).

Audience: contributors, integrators, and anyone tuning the DSPy pipeline.

> **Status:** in progress. Sections marked _TBD_ will be filled in
> incremental commits.

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [High-Level Architecture](#2-high-level-architecture)
3. Pipeline Stages — _TBD_
4. DSPy Signature Reference — _TBD_
5. DSPy Module Reference — _TBD_
6. Cross-Cutting Concerns — _TBD_
7. Extensibility — _TBD_
8. Key Design Decisions — _TBD_

## 1. Purpose & Scope

### What this document covers

- The static structure of the codebase: layers, modules, and their
  responsibilities.
- The runtime structure: how the CLI, the DSPy pipeline, and the optional
  image generator wire together.
- The contracts between components: dataclasses, DSPy signatures, and the
  Markdown export shape.
- Cross-cutting concerns (logging, observability, caching, optimization).
- Extension points for adding new pipeline stages or swapping providers.

### What this document does *not* cover

- User-facing usage instructions — see `README.md`.
- Project goals, success criteria, and non-goals — see `SYSTEM_ANALYSIS.md`.
- Coding standards, agent rules, and review discipline — see `AGENTS.md`.

### Conventions

- File references use `path:line` notation (e.g. `main.py:840`).
- "Stage" = one logical phase of the pipeline (e.g. *premise*, *spine*).
- "Module" = a `dspy.Module` subclass; "signature" = a `dspy.Signature`
  subclass.
- "Generator" is used loosely to mean a DSPy module that produces text or
  structured output.

## 2. High-Level Architecture

The system is a single-process Python CLI organized into three layers:

```
┌──────────────────────────────────────────────────────────────────────┐
│                       CLI / UI Layer (main.py)                       │
│  argparse · rich.Console · Prompt/Confirm · interactive flow funcs   │
└──────────────────────────────────────────────────────────────────────┘
                              │  calls
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      Pipeline Layer (DSPy modules)                   │
│   story_modules.py · world_bible_modules.py  (UI-free, pure-ish)     │
│   QuestionGenerator · CorePremiseGenerator · SpineTemplateGenerator  │
│   WorldBibleQuestionGenerator · WorldBibleGenerator                  │
│   StoryGenerator · ChapterInpaintingGenerator · ChapterSummarizer    │
│   CharacterVisualDescriber · SceneImagePromptGenerator               │
└──────────────────────────────────────────────────────────────────────┘
                              │  delegates
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       Cross-Cutting / Infra                          │
│  logging_config.py  · _compat.py (Langfuse @observe)                 │
│  dspy_optimization.py (manifest load) · exceptions.py                │
│  postprocessing.py  · image_gen.py (Replicate)                       │
│  world_bible.py (shared Pydantic model)                              │
└──────────────────────────────────────────────────────────────────────┘
```

### Module map

| File | Layer | Responsibility |
|---|---|---|
| `main.py` | CLI/UI | Arg parsing, runtime setup, interactive flow, file output. Contains all `console.print` / `Prompt.ask` calls. |
| `story_modules.py` | Pipeline | Core story DSPy signatures + modules; chapter parsing/cleaning helpers; `QuestionWithAnswer` and `CharacterVisual` Pydantic models. |
| `world_bible_modules.py` | Pipeline | World-bible DSPy signatures + modules; plot-timeline normalization. |
| `world_bible.py` | Shared model | `WorldBible` Pydantic model with `full_text` rendering. |
| `image_gen.py` | Infra | Replicate-based portrait + scene illustration generation. |
| `postprocessing.py` | Infra | Sentence-similarity duplicate detection. |
| `dspy_optimization.py` | Infra | Optimization manifest load/save and per-module artifact loading. |
| `logging_config.py` | Infra | Centralized logging + token-usage callback. |
| `exceptions.py` | Infra | `RECOVERABLE_MODEL_EXCEPTIONS` / `RECOVERABLE_RUNTIME_EXCEPTIONS` tuples. |
| `_compat.py` | Infra | Langfuse `@observe` shim when the SDK is absent. |
| `scripts/` | Tooling | `optimize_text_pipeline.py`, `fetch_langfuse_traces.py`, `render_story.py`, `word_count.py`. |
| `test_story.py`, `test_postprocessing.py` | Tests | Pytest coverage with `MockLM`. |

### Layering rules (enforced by `AGENTS.md`)

- The Pipeline layer **must not** import `rich` or call `console.print` /
  `Prompt.ask`. Interaction is the CLI/UI layer's job.
- The CLI/UI layer **owns** all file I/O for the final Markdown bundle and
  all argument-parsing logic.
- Cross-cutting modules are leaves: they may be imported by either layer but
  must not depend on `main.py`.
- Shared types (`WorldBible`, `QuestionWithAnswer`, `CharacterVisual`) live
  next to the pipeline and are imported by both layers.

### Process model

Single-process, synchronous, single-user. There is no concurrency model
beyond what DSPy and the underlying LLM SDK provide. Each pipeline run is
one `python main.py` invocation that reads from stdin (via `rich.Prompt`)
and writes one Markdown file.
