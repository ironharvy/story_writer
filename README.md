# Story Writer

Interactive DSPy-based story generation pipeline with optional image generation and Langfuse observability.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## What It Does

- Guides you through an interactive ideation flow (questions, premise refinement, spine, world bible).
- Generates structured story artifacts (arc outline, chapter plan, enhancers guide, full story).
- Optionally generates character portraits and per-chapter scene illustrations via Replicate.
- Writes all outputs to a markdown file in your chosen output directory.

## Project Layout

- `main.py` — primary interactive CLI for story generation.
- `story_modules.py` — core story generation modules/signatures.
- `world_bible_modules.py` — world bible question + generation modules.
- `image_gen.py` — Replicate-based image generation helpers.
- `logging_config.py` — centralized logging configuration.
- `alternate_story_modules.py` — alternate Architect→Director→Scripter→Writer pipeline modules.
- `scripts/fetch_langfuse_traces.py` — utility to fetch/summarize Langfuse traces.
- `test_story.py`, `test_alternate.py` — pytest coverage for main and alternate pipelines.

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
- `--cache-dir`
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

- Core Premise
- Spine Template
- World Bible
- Arc Outline
- Chapter Plan
- Enhancers Guide
- Final Story
- Optional character portraits / scene image embeds when images are enabled

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
