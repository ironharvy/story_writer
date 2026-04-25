# Story Writer — System Analysis

This document is a lightweight Software Requirements Specification (SRS) /
System Concept Document for the Story Writer project. It synthesizes what the
system is for, what it must do, how it does it, and how we know it works.

## 1. Problem Statement

Writing a long-form story end-to-end is hard for both humans and naive LLM
prompts:

- **One-shot LLM prompts** ("write me a 10-chapter novel about X") collapse
  under their own context: characters drift, world rules contradict
  themselves, scenes repeat, and the tonal/structural quality decays as the
  output grows.
- **Pure human authoring** is slow and front-loaded with unstructured
  ideation: turning a vague idea into a coherent premise, world, and
  chapter-by-chapter plan is the part most aspiring writers stall on.
- **Existing AI writing tools** tend to be either chat-only (no structured
  artifacts) or fully automated black boxes (no human steering, no
  intermediate review points, no observability into model behavior).

There is no lightweight, observable, locally-runnable pipeline that walks an
author from a one-line idea through a structured story bible to a full draft,
while keeping the human in the loop and the LLM calls inspectable and
optimizable.

## 2. Goals & Non-Goals

### Goals

- Take a user from a rough idea to a complete, structured, multi-chapter
  story in a single guided session.
- Produce **discrete, reviewable artifacts** at each stage (premise, spine,
  world bible, chapter plan, enhancers guide, full story) instead of one
  opaque blob.
- Keep the **human in the loop** at every ideation step (questions, premise
  approval, world-bible refinement) without forcing them to drive the LLM
  directly.
- Be **provider-agnostic**: work against OpenAI, Ollama, or any
  DSPy-supported LM, configurable from CLI/env.
- Be **observable and tunable**: every LLM call is traceable (Langfuse) and
  every text module is independently optimizable (DSPy compile artifacts).
- Optionally enrich the output with **character portraits and per-chapter
  scene illustrations** via Replicate.

### Non-Goals

- Not a multi-user web app — it is a single-user interactive CLI.
- Not a fine-tuning project — quality comes from prompt structure, DSPy
  optimization, and pipeline decomposition, not from model training.
- Not a publishing/formatting tool beyond a single Markdown export (and a
  helper HTML renderer in `scripts/`).
- Not an autonomous agent — the user is expected to approve/redirect at the
  ideation stages.

## 3. Stakeholders & Users

| Stakeholder | Interest |
|---|---|
| Hobbyist / amateur author | Wants a structured assistant to get from idea to draft. |
| Prompt / DSPy researcher | Wants a real, multi-stage pipeline to optimize and benchmark. |
| Tooling integrator | Wants a clean module boundary (`story_modules`, `world_bible_modules`) to embed elsewhere. |

## 4. Functional Requirements

The system shall:

1. **Ideate.** Generate clarifying questions about a user's seed idea and
   collect human answers (`QuestionGenerator`).
2. **Crystallize a premise.** Synthesize a `CorePremise` from the seed idea
   and Q&A (`CorePremieGenerator`).
3. **Build a spine.** Produce a 3-act spine template
   (`SpineTemplateGenerator`).
4. **Construct a world bible.** Ask world-building questions, then generate a
   structured `WorldBible` with rules / characters / locations / plot
   timeline (`WorldBibleQuestionGenerator`, `WorldBibleGenerator`).
5. **Plan and write chapters.** Generate a chapter plan, an enhancers guide,
   and the full chapter-by-chapter story (`StoryGenerator`,
   `ChapterInpaintingGenerator`).
6. **Post-process.** Detect repeated/near-duplicate sentences across the
   draft (`postprocessing.find_similar_sentences`) and surface a report.
7. **Optionally illustrate.** Generate character portraits and per-chapter
   scene images via Replicate when `--enable-images` is set
   (`CharacterVisualDescriber`, `SceneImagePromptGenerator`, `image_gen`).
8. **Export.** Write all artifacts to a single Markdown file (default
   `.tmp/story_output.md`) with images embedded when present.
9. **Be configurable.** Accept model, endpoint, API key, cache, optimization
   manifest, image toggle, and verbosity via CLI flags and env vars.
10. **Be observable.** Emit structured logs (text or JSON), token-usage
    callbacks, and OpenTelemetry/DSPy traces to Langfuse when configured.

## 5. Non-Functional Requirements

- **Local-first:** runs on a developer laptop with `python main.py`; no
  server component required.
- **Provider-agnostic LLM:** any DSPy-supported `dspy.LM` model works
  (default `openai/gpt-4o-mini`).
- **Caching:** disk + memory caching of LLM calls is on by default to keep
  iteration cheap; toggleable.
- **Optimizability:** any text module can be compiled offline
  (`scripts/optimize_text_pipeline.py`) and loaded at runtime via a
  manifest, without code changes.
- **Code-quality discipline (per `AGENTS.md`):** ≤50 lines/function,
  separation of UI from pipeline, dataclasses for shared state, no
  commented-out code.
- **Testability:** Pipeline modules are pure-ish DSPy modules and are
  covered by `test_story.py` / `test_postprocessing.py` using a `MockLM`.

## 6. Proposed Solution

A staged, **DSPy-orchestrated pipeline** driven by a thin interactive CLI:

```
seed idea
   │
   ▼
[QuestionGenerator] ──► user answers
   │
   ▼
[CorePremiseGenerator] ──► CorePremise (user-approved)
   │
   ▼
[SpineTemplateGenerator] ──► 3-act spine
   │
   ▼
[WorldBibleQuestionGenerator] ──► user answers
   │
   ▼
[WorldBibleGenerator] ──► WorldBible {rules, characters, locations, timeline}
   │
   ▼
[StoryGenerator] ──► chapter plan + enhancers guide + full story
   │
   ├──► [ChapterInpaintingGenerator] (per-chapter refinement)
   │
   ├──► [postprocessing] ──► duplicate-sentence report
   │
   └──► [ImageGenerator] (optional) ──► portraits + scene images
   │
   ▼
.tmp/story_output.md  (Markdown bundle of all artifacts)
```

Key design choices:

- **DSPy signatures** describe each LLM call declaratively (typed inputs /
  outputs, Pydantic models for structured data like `WorldBible`,
  `QuestionWithAnswer`).
- **One module = one responsibility**, each living in `story_modules.py` /
  `world_bible_modules.py`, exported as plain classes that take inputs and
  return structured outputs — making them swappable, testable, and
  optimizable in isolation.
- **CLI orchestration** lives in `main.py` and is the only place that knows
  about `rich` prompts, file I/O, and argument parsing — pipeline modules
  are UI-free.
- **Cross-cutting concerns** are isolated: logging in `logging_config.py`,
  optimizer loading in `dspy_optimization.py`, recoverable-exception lists
  in `exceptions.py`, image generation in `image_gen.py`.
- **Post-processing** uses an alphabetical-sort + `SequenceMatcher` trick to
  cheaply surface duplicated sentences without embeddings.
- **Observability** is opt-in via Langfuse env vars; instrumentation is
  attached only when keys are present.

## 7. Success Criteria

The system is considered successful when:

| Criterion | How we measure it |
|---|---|
| End-to-end run completes for a typical idea on the default model. | `python main.py` produces `.tmp/story_output.md` containing all six sections (Premise, Spine, World Bible, Chapter Plan, Enhancers Guide, Final Story). |
| Output is structurally coherent. | Chapter headings parse cleanly (`_chapter_heading_re` matches every chapter); world-bible sections are populated; chapter count matches the plan. |
| Repetition stays bounded. | `postprocessing.find_similar_sentences` reports zero pairs above ~0.85 similarity on a clean run. |
| Pipeline is provider-portable. | The same flow runs against at least one OpenAI model and one Ollama model with only flag/env changes. |
| Modules are independently testable. | `pytest -q` passes; each generator has at least one test using `MockLM`. |
| Modules are independently optimizable. | `scripts/optimize_text_pipeline.py` can compile any subset via `--modules`, write a manifest, and `--use-optimized` loads them at runtime. |
| Behavior is observable. | When Langfuse env vars are set, every module call appears as a trace; token usage is logged via `TokenUsageCallback`. |
| Code stays within the discipline in `AGENTS.md`. | `ruff check .` clean; no function >50 lines; UI calls (`console.print`, `Prompt.ask`) absent from pipeline modules. |

## 8. Constraints & Assumptions

- Python ≥ 3.10 (uses `X | Y` type unions, `list[...]` generics).
- A DSPy-compatible LM endpoint is reachable (network or local).
- For images, a Replicate token and the `replicate` package are available.
- The user is willing to answer ideation questions interactively; the
  pipeline is not designed for fully unattended runs.
- Output volume fits the configured `--max-tokens` (default 8000) per LLM
  call; very long stories rely on the chapter-by-chapter decomposition
  rather than a single mega-call.

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM drift / contradiction across chapters. | Structured `WorldBible` + chapter plan + enhancers guide are passed as stable context; `ChapterInpaintingGenerator` allows per-chapter refinement. |
| Repetition / "purple prose" loops. | Post-processing duplicate-sentence report; `RANDOM_DETAIL_PROBABILITY` injects variety. |
| Provider/API outage or quota. | `RECOVERABLE_MODEL_EXCEPTIONS` / `RECOVERABLE_RUNTIME_EXCEPTIONS` allow graceful degradation; caching avoids re-paying for identical calls. |
| Observability vendor lock-in (Langfuse). | All instrumentation is opt-in and gated on env vars; the pipeline runs unchanged without it. |
| Pipeline complexity creep. | `AGENTS.md` enforces hard rules (50-line functions, one-domain-per-module, no UI in pipeline) and is the contract for AI agents and humans alike. |

## 10. Out of Scope (for now)

- Multi-user collaboration, accounts, or hosted UI.
- Fine-tuning or RLHF on user feedback.
- Optimization of image-oriented modules
  (`CharacterVisualDescriber`, `SceneImagePromptGenerator`) — deferred per
  the README TODO until text-pipeline metrics are stable.
- Non-Markdown export formats beyond the existing HTML renderer helper.
