# Architecture

> References: [DEC-0001], [DEC-0002], [DEC-0003], [DEC-0004], [DEC-0005], [DEC-0006], [DEC-0007], [DEC-0008], [DEC-0009], [DEC-0010]

## Stack

- Python 3.11+
- Typer for CLI commands
- Rich for interactive terminal prompts
- Pydantic v2 for artifact models
- DSPy for typed stage modules
- Ollama through DSPy/LiteLLM as the default local model provider
- Pytest and Ruff for validation

## Package Layout

```text
src/story_writer/
├── cli.py
├── config.py
├── interactive.py
├── orchestrator.py
├── providers.py
├── render.py
├── run_store.py
├── models/
│   ├── qa.py
│   ├── run.py
│   └── story.py
├── qa/
│   └── rules.py
└── stages/
    ├── base.py
    ├── clarify.py
    ├── premise.py
    ├── spine.py
    ├── world_bible.py
    ├── chapter_plan.py
    ├── enhancement.py
    ├── embellish.py
    └── prose.py
```

## Pipeline

```text
idea
  -> clarify
  -> premise [review]
  -> spine [review]
  -> world_bible [review]
  -> chapter_plan (includes chapter beats)
  -> enhancement
  -> embellish
  -> prose
  -> qa detection
  -> render command
```

## Stage Contract

Every DSPy generation stage exposes a `StageBase.run(context)` method and keeps its DSPy call behind `_execute(context)`. The orchestrator only sees the stage name and returned Pydantic model. It does not reach into stage internals. QA and rendering are orchestration services rather than DSPy generation stages.

Each stage file contains:

- one `dspy.Signature`
- one `StageBase` subclass
- a Pydantic return model from `story_writer.models.story`

## Persistence

`RunStore` owns all filesystem behavior under `runs/<slug>/`. Artifacts are JSON except `idea.txt` and rendered story files. The manifest records stage status, provider, model, timestamps, and errors.

## Provider Routing

`config.DEFAULT_ROUTING` maps stages to installed local Ollama models: `qwen3:latest` for structured planning, `gemma4:26b` for world bible and prose, and `gemma3:4b` for embellishment. `--model` can override all stages. DeepSeek or any other paid provider requires `--allow-paid`; MVP implementation rejects paid routing by default.

## QA Strategy

QA is detection-first. `detect_chapter()` returns `RuleResult` values for the canonical R1-R8 table in `docs/requirements.md`.

## Error Contract

Stage failures are recorded in the manifest as `failed` with the exception message. `--strict` additionally fails the command when hard QA violations are present.

## Deep Module Rationale

The orchestrator coordinates stages and persistence only. Prompt shape, structured parsing, provider details, and QA rule internals stay inside their modules so changes can be measured per stage without widening the public surface.
