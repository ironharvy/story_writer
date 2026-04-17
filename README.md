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
- Optionally generates character portraits and per-chapter scene illustrations via Replicate, either inline during a run or after the fact via `scripts/generate_images.py`.
- Writes outputs as a human-readable markdown file plus a structured JSON sidecar.

## Project Layout

- `main.py` — primary interactive CLI for story generation.
- `story_modules.py` — core story generation modules/signatures.
- `world_bible_modules.py` — world bible question + generation modules.
- `image_gen.py` — Replicate-based image backend (portraits + scenes).
- `image_pipeline.py` — shared image pipeline (describe characters, generate portraits/scenes) consumed by both `main.py` and the standalone script.
- `story_artifacts.py` — `StoryArtifacts` dataclass + JSON sidecar I/O + markdown rendering.
- `logging_config.py` — centralized logging configuration.
- `alternate_story_modules.py` — alternate Architect→Director→Scripter→Writer pipeline modules.
- `scripts/generate_images.py` — standalone CLI that adds/refreshes images on an existing story JSON.
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

By default, every run writes two files to the output directory (default `.tmp`):

- `story_output.md` — human-readable markdown view.
- `story_output.json` — structured sidecar with all text artifacts plus any image paths. This is the input format consumed by `scripts/generate_images.py`.

The markdown includes:

- Core Premise
- Spine Template
- World Bible
- Arc Outline
- Chapter Plan
- Enhancers Guide
- Final Story
- Optional character portraits / scene image embeds when images are enabled

## Generating Images After the Fact

If you generated a story without `--enable-images` and want images later, feed the saved JSON to the standalone script:

```bash
python scripts/generate_images.py \
  --story-json .tmp/story_output.json \
  --output-markdown .tmp/story_output.with_images.md
```

By default it writes updated image paths back into the same JSON. Useful flags:

- `--skip-portraits`, `--skip-scenes` — do only one kind of image.
- `--regenerate-portraits`, `--regenerate-scenes` — replace images already recorded in the JSON.
- `--output-json`, `--output-markdown` — write to new files instead of mutating in place.
- `--images-dir` — where to save image files (default `images/`).
- `--model` / `--api-key` / `--llm-url` — the text LLM used to derive image prompts from the world bible and chapters.

You still need `REPLICATE_API_TOKEN` (or `--replicate-api-token`).

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
