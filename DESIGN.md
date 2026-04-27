# Story Writer — Software Design Document (SDD)

This document describes *how* Story Writer is built. It is the implementation
counterpart to `SYSTEM_ANALYSIS.md` (which describes *what* and *why*).

Audience: contributors, integrators, and anyone tuning the DSPy pipeline.

> **Status:** in progress. Sections marked _TBD_ will be filled in
> incremental commits.

## Table of Contents

1. [Purpose & Scope](#1-purpose--scope)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Pipeline Stages](#3-pipeline-stages)
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

## 3. Pipeline Stages

A run progresses through a fixed sequence of stages. Most stages produce a
named artifact that is fed into one or more later stages; a few are
side-effecting (file I/O, image generation) or optional.

### 3.1 Stage diagram

```
                ┌─────────────────────────────────────────────┐
                │  S0 Bootstrap                               │
                │  argparse → logging → DSPy LM → generators  │
                └─────────────────────────────────────────────┘
                                   │
                                   ▼
                       ┌──────────────────────┐
                       │  S1 Idea capture     │  ← user stdin
                       └──────────────────────┘
                                   │ idea
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S2 Premise refinement (loop)               │
                │  QuestionGenerator → user Q&A               │
                │  → CorePremiseGenerator → user accept?      │
                └─────────────────────────────────────────────┘
                                   │ core_premise, qa_text
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S3 Spine template (loop)                   │
                │  SpineTemplateGenerator → user accept?      │
                └─────────────────────────────────────────────┘
                                   │ spine_template
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S4 World bible                             │
                │  WorldBibleQuestionGenerator → user Q&A     │
                │  → WorldBibleGenerator (rules→chars→        │
                │    locs→timeline, CoT each)                 │
                └─────────────────────────────────────────────┘
                                   │ WorldBible
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S5 Character visuals  (opt: --enable-images)│
                │  CharacterVisualDescriber → ImageGenerator   │
                │  .generate_character_portrait                │
                └─────────────────────────────────────────────┘
                                   │ character_visuals,
                                   │ portrait paths
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S6 Story generation                        │
                │  StoryGenerator.forward:                    │
                │    per-act ChapterPlan (×3 acts)            │
                │    → EnhancersGuide                         │
                │    → per-chapter loop (RandomDetail?,       │
                │       SingleChapter, ChapterSummarizer)     │
                └─────────────────────────────────────────────┘
                                   │ chapter_plan,
                                   │ enhancers_guide, story
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S7 Chapter inpainting (opt: --inpaint-…)   │
                │  ChapterInpaintingGenerator (per chapter)   │
                └─────────────────────────────────────────────┘
                                   │ final_story_text
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S8 Scene illustrations (opt: --enable-…)   │
                │  SceneImagePromptGenerator →                │
                │  ImageGenerator.generate_scene_illustration │
                └─────────────────────────────────────────────┘
                                   │ scene_image_paths
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S9 Similarity check                        │
                │  postprocessing.find_similar_sentences      │
                └─────────────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────────────┐
                │  S10 Markdown export                        │
                │  save_story_output → .tmp/story_output.md   │
                └─────────────────────────────────────────────┘
```

### 3.2 Stage reference

Inputs/outputs use the dataclass and Pydantic types defined in the codebase.
"Side effects" calls out anything beyond pure transformation.

#### S0 — Bootstrap

- **Entry:** `main()` at `main.py:840`.
- **Input:** `sys.argv`, environment variables, `.env` file.
- **Steps:** `build_arg_parser` → `setup_runtime` (logging + DSPy LM via
  `configure_dspy`) → `initialize_text_generators` (with optional optimized
  artifact loading via `try_load_optimized_module`).
- **Output:** parsed `Namespace`, `dict[str, dspy.Module]` of generators.
- **Side effects:** configures `dspy.LM`, registers `TokenUsageCallback`,
  optionally instruments DSPy for Langfuse.

#### S1 — Idea capture

- **Entry:** `Prompt.ask(...)` in `main()`.
- **Input:** user stdin.
- **Output:** `idea: str`.

#### S2 — Premise refinement

- **Entry:** `run_core_premise_flow` at `main.py:344`.
- **Loop:** `QuestionGenerator(idea)` → `get_answers_for_questions(...)` →
  `CorePremiseGenerator(idea, qa_pairs)` → user prompts for refinement; if
  the user refines, `idea` is rewritten to embed prior premise + refinement
  details and the loop repeats.
- **Output:** `(idea, core_premise, qa_text)`.

#### S3 — Spine template

- **Entry:** `run_spine_template_flow` at `main.py:386`.
- **Loop:** `SpineTemplateGenerator(idea, qa_pairs, core_premise)` →
  optional refinement (same pattern as S2).
- **Output:** `(idea, spine_template)`.

#### S4 — World bible

- **Entry:** `generate_world_bible` at `main.py:426`.
- **Steps:** `WorldBibleQuestionGenerator(core_premise, spine_template)` →
  Q&A collection → `WorldBibleGenerator(core_premise, spine_template,
  user_additions)`.
- **Internals:** `WorldBibleGenerator` chains four `dspy.ChainOfThought`
  calls in dependency order — rules → characters → locations → plot
  timeline — and post-processes the timeline via `_normalize_plot_timeline`
  to remove duplicate act headings.
- **Output:** `WorldBible` (Pydantic) with `rules`, `characters`,
  `locations`, `plot_timeline`, plus `full_text` rendering.

#### S5 — Character visuals (optional)

- **Entry:** `maybe_generate_character_assets` at `main.py:492`.
- **Gated by:** `--enable-images` and a Replicate token.
- **Steps:** `CharacterVisualDescriber(world_bible)` → list of
  `CharacterVisual` (with normalized `reference_mix`,
  `distinguishing_features`, `full_prompt`). For each character,
  `ImageGenerator.generate_character_portrait` calls Replicate's Animagine
  XL 4.0 and saves a PNG.
- **Output:** `ImageArtifacts` dataclass: visuals, portrait paths, summary
  string, and the `ImageGenerator` instance for reuse in S8.
- **Failure mode:** per-character recoverable exceptions are logged and
  skipped; the run continues without that portrait.

#### S6 — Story generation

- **Entry:** `generate_story_text` at `main.py:578` →
  `StoryGenerator.forward` at `story_modules.py:873`.
- **Sub-steps:**
  1. `_generate_chapter_plan_entries`: loops over `_ACT_SEQUENCE`
     (Setup / Confrontation / Resolution), invoking
     `GenerateChapterPlanSignature` once per act and threading
     `previous_chapters` so later acts continue without repeating beats.
     Final list is normalized via `_normalize_chapter_plan_entries`
     (numbering + title cleanup).
  2. `_generate_enhancers_guide`: one
     `GenerateEnhancersSignature` call producing tension/mystery/twist
     guidance over the whole plan.
  3. `_write_story_chapters`: per-chapter loop. Each iteration:
     - Probabilistically generates a `random_detail` (gated by
       `RANDOM_DETAIL_PROBABILITY`, default 0.35).
     - Calls `GenerateSingleChapterSignature` with world facts, plan,
       enhancers guide, rolling summary, and the optional detail.
     - Calls `ChapterSummarizer` to produce a 2-3 sentence factual summary
       for the rolling context (with a truncation fallback on failure).
     - Optionally appends a verbatim tail of the previous chapter
       (gated by `verbatim_tail_paragraphs`, default 0).
- **Output:** `dspy.Prediction(chapter_plan, enhancers_guide, story)`.

#### S7 — Chapter inpainting (optional)

- **Entry:** `_run_optional_inpainting` at `main.py:537` →
  `ChapterInpaintingGenerator.forward` at `story_modules.py:582`.
- **Gated by:** `--inpaint-chapters`; requires `--inpaint-ratio > 1.0`.
- **Steps:** `_split_story_into_chapters` parses `### Chapter N:` headings
  out of the story; each chapter is re-expanded via
  `GenerateChapterInpaintingSignature` (CoT) with the world bible and the
  full chapter plan as context. Failures fall back to the original
  chapter text.
- **Output:** rebuilt story string + counts of expanded vs total chapters.

#### S8 — Scene illustrations (optional)

- **Entry:** `maybe_generate_scene_images` at `main.py:602`.
- **Gated by:** `--enable-images` and successful S5.
- **Steps:** for each chapter slice (split on `### Chapter `),
  `SceneImagePromptGenerator(chapter_text, character_visuals_summary)` →
  `ImageGenerator.generate_scene_illustration` (FLUX Kontext, using the
  first portrait as a reference image to preserve character identity).
- **Output:** `dict[chapter_index, image_path]`.

#### S9 — Similarity check

- **Entry:** `run_similarity_check` at `main.py:640`.
- **Gated by:** `--check-similar` (default on); threshold from
  `--similar-threshold`.
- **Steps:** `find_similar_sentences` extracts sentences, normalizes them,
  alphabetically sorts, and runs `SequenceMatcher` on adjacent pairs;
  `format_report` prints the result.
- **Output:** console report; `logger.warning` if any pair is found.

#### S10 — Markdown export

- **Entry:** `save_story_output` at `main.py:704`.
- **Output:** `<output_dir>/story_output.md` with sections, in order:
  Core Premise → Spine Template → World Bible → (optional Character
  Visuals with embedded portraits) → Chapter Plan → Enhancers Guide →
  Final Story (with optional per-chapter scene image embeds).

### 3.3 State carried between stages

The CLI/UI layer threads two dataclasses through the pipeline rather than
passing loose locals (per `AGENTS.md` data-flow rules):

- `StoryFoundation` (`main.py:569`): `core_premise`, `spine_template`,
  `world_bible`. Built by S2–S4, consumed by S6.
- `ImageArtifacts` (`main.py:180`): `character_visuals`,
  `character_portrait_paths`, `character_visuals_summary`,
  `image_generator`. Built by S5, consumed by S8 and S10.
- `StoryRunArtifacts` (`main.py:729`): the union of the above plus the
  story result and final text — used only by S10 to write the bundle.

Inside `StoryGenerator`, the per-chapter loop additionally maintains:

- `previous_summary_entries: list[str]` — rolling factual summaries that
  feed `previous_chapters_summary` into the next chapter's signature.
- `ChapterWritingContext` (frozen dataclass) — invariants that don't
  change per chapter (`world_bible`, `chapter_plan_text`, `enhancers_guide`).
