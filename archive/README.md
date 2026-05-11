> **Archived.** This documents the pre-rewrite pipeline (`archive/main.py`, `archive/cli.py`,
> `archive/models.py`, `archive/story_modules.py`, …) and is kept for reference alongside that
> code. The current pipeline is documented in the repo-root [`README.md`](../README.md).

---

# Story Writer

Interactive DSPy-based story generation pipeline with optional image generation and Langfuse observability.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## What It Does

- Guides you through an interactive ideation flow with feedback loops at every stage (questions, premise, spine, world bible, chapter plan).
- Generates structured story artifacts (chapter plan, enhancers guide, full story).
- Saves progress incrementally — if the process crashes, you keep everything generated so far.
- Optionally generates character portraits and per-chapter scene illustrations via Replicate.
- Writes all outputs to a markdown file in your chosen output directory.

## Project Layout

- `main.py` — thin CLI entry point: parse args, setup runtime, run pipeline, save output.
- `cli.py` — argument parsing (`build_arg_parser`, model/runtime/output flag groups).
- `models.py` — pipeline dataclasses (`GenerationParams`, `ImageArtifacts`, `StoryFoundation`, `StoryRunArtifacts`).
- `pipeline.py` — orchestration logic with `@observe()` tracing and user feedback loops.
- `ui.py` — Rich-based interactive UI layer (all `console.print` / `Prompt.ask` / `Confirm.ask`).
- `output.py` — file I/O helpers: incremental `update_artifact()` and final `save_story_output()`.
- `qa.py` — post-generation quality checks (similar-sentence detection).
- `story_modules.py` — core DSPy story generation modules/signatures.
- `world_bible_modules.py` — world bible question + generation modules.
- `world_bible.py` — structured `WorldBible` Pydantic model.
- `image_gen.py` — Replicate-based image generation helpers.
- `logging_config.py` — centralized logging configuration with token-usage callback.
- `dspy_runtime.py` — shared DSPy LM configuration helpers.
- `dspy_optimization.py` — optimized module loading.
- `_compat.py` — compatibility shims (Langfuse `@observe()` fallback).
- `scripts/` — utilities (Langfuse traces, text-pipeline optimization, word count).
- `test_story.py` — pytest coverage for the primary pipeline.

## Requirements

- Python 3.10+
- Access to an LLM provider supported by DSPy (default model is `openai/gpt-4o-mini`)
- Optional for images: Replicate account/token
- Optional for observability: Langfuse credentials

Install dependencies:

```bash
pip install -r requirements.txt
```

For development/test tooling:

```bash
pip install -r requirements-dev.txt
```

If you plan to use image generation, also install:

```bash
pip install replicate
```

## Environment Variables

Copy `.env.example` to `.env` and fill what you need.

Core model/runtime settings:

- `MODEL` (default: `openai/gpt-4o-mini`)
- `LLM_URL` (for local/custom providers, e.g. Ollama)
- `API_KEY`
- `DSPY_CACHE_DIR`
- `DSPY_USE_OPTIMIZED` (set `true`/`1` to load optimized text-module artifacts)
- `DSPY_OPTIMIZED_MANIFEST` (path to text-pipeline optimization manifest)

Optional image generation:

- `REPLICATE_API_TOKEN`

Optional logging:

- `LOG_LEVEL` (e.g. `DEBUG`, `INFO`)
- `LOG_FORMAT` (`text` or `json`)
- `LOG_FILE` (set to enable JSON file logging)

Optional Langfuse:

- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_HOST` (default in example: `https://cloud.langfuse.com`)

## Running the App

Basic run:

```bash
python main.py
```

Example with explicit model endpoint:

```bash
python main.py --model ollama_chat/llama3 --llm-url http://localhost:11434
```

Enable images:

```bash
python main.py --enable-images --replicate-api-token "$REPLICATE_API_TOKEN"
```

## Main CLI Flags (`main.py`)

- `--model`
- `--llm-url`
- `--api-key`
- `--max-tokens`
- `--output-dir` (default: `.tmp`)
- `--cache` / `--no-cache`
- `--memory-cache` / `--no-memory-cache`
- `--cache-dir` (default: `.cache/dspy`)
- `--use-optimized` / `--no-use-optimized`
- `--optimized-manifest` (default: `.tmp/dspy_optimized/text_pipeline_manifest.json`)
- `--enable-images`
- `--replicate-api-token`
- `--log-file`
- `-v`, `-vv`, `-vvv` (increasing verbosity)

## Text Pipeline Optimization

Compile/save text-module artifacts and a manifest:

```bash
python scripts/optimize_text_pipeline.py \
  --model openai/gpt-4o-mini \
  --manifest .tmp/dspy_optimized/text_pipeline_manifest.json
```

Run with optimized text modules enabled:

```bash
python main.py \
  --use-optimized \
  --optimized-manifest .tmp/dspy_optimized/text_pipeline_manifest.json
```

Optimize only a subset of text modules:

```bash
python scripts/optimize_text_pipeline.py \
  --modules QuestionGenerator,CorePremiseGenerator,StoryGenerator
```

## Output

By default, output is written to:

- `.tmp/story_output.md`

The markdown includes:

- Generation Parameters (model, max_tokens, cache settings)
- Core Premise
- Spine Template
- World Bible
- Chapter Plan
- Enhancers Guide
- Final Story
- Optional character portraits / scene image embeds when images are enabled

Sections are written incrementally as they are generated, so partial progress is preserved on interruption.

## Logging Behavior

- Console logging is always enabled.
- `--log-file` (or `LOG_FILE`) enables JSON file logging.
- Verbosity flags:
  - `-v`: INFO for app logs
  - `-vv`: includes LLM-related debug logs
  - `-vvv`: full HTTP + LLM debug firehose

## Langfuse Trace Utility

Fetch traces:

```bash
python scripts/fetch_langfuse_traces.py --mode fetch --limit 50 --hours 24 --output .tmp/langfuse_traces.json
```

Summarize traces:

```bash
python scripts/fetch_langfuse_traces.py --mode summarize --input .tmp/langfuse_traces.json --output .tmp/langfuse_summary.json --summary-hours 24
```

## Running Tests

```bash
pytest -q
```

## TODO

- Defer DSPy optimization for image-oriented modules until text-pipeline metrics are stable:
  - `CharacterVisualDescriber`
  - `SceneImagePromptGenerator`
