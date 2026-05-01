# Story Writer

A local-first, DSPy-driven pipeline that turns a story idea into a coherent, artifact-free narrative. Runs against a local Ollama by default; optionally escalates to DeepSeek for harder steps.

This is the MVP rebuild: **CLI only, output quality is the only success metric.** Web UI, auth, image/audio generation, and story sharing are deferred to later phases.

## Quickstart

```bash
# 1. Make sure Ollama is running and you have models pulled
ollama pull llama3.1:8b
ollama pull qwen2.5:14b

# 2. Install
pip install -e .[dev]
cp .env.example .env

# 3. Run
story-writer new --idea "A girl looks for a brother no one remembers."
```

You'll be asked a few clarifying questions, then prompted to accept/edit/regenerate the premise, spine, and world bible. The remaining stages run unattended. Final story:

```bash
story-writer render <slug>     # writes runs/<slug>/story.md
```

## The pipeline

Each stage is a typed DSPy module. Each artifact is persisted as JSON under `runs/<slug>/`. Stages are skipped if their output already exists, so you can `Ctrl-C` and `story-writer resume <slug>` at any time.

| Stage           | What it produces                                                |
|-----------------|------------------------------------------------------------------|
| `clarify`       | 3–7 clarifying questions (user answers in interactive mode)      |
| `premise`       | Protagonist / want / obstacle / stakes + 2–4 sentence summary    |
| `spine`         | Pixar six-beat structure                                         |
| `world_bible`   | Rules, characters, locations, timeline (single source of truth)  |
| `chapter_plan`  | Ordered chapters sized by `--length`, each with 4–10 beats       |
| `enhancement`   | Four passes per chapter: tension, mystery, setup/payoff, theme   |
| `embellish`     | Random small detail injection (probability `--embellish-probability`) |
| `prose`         | The actual chapter text                                          |
| `qa` (always)   | Detection of LLM artifacts (R1–R6); reports under `runs/<slug>/qa/` |

## CLI

```bash
story-writer new --idea "..."  [--length short|novella|novel]
                               [--slug NAME]
                               [--embellish-probability 0.25]
                               [--non-interactive]
                               [--allow-paid]
                               [--strict]
                               [--skip-qa-embeddings]
                               [--model MODEL]      # override every stage with one Ollama model
                               [--profile PROFILE]  # named bundle: fast | quality | tiny

story-writer resume <slug>
story-writer render  <slug> [--fmt md|txt] [--out PATH]
story-writer inspect <slug> [--stage premise]
```

## Providers

- **Ollama** (default) — set `OLLAMA_HOST` if not on `localhost:11434`.
- **DeepSeek** (opt-in) — set `DEEPSEEK_API_KEY` and pass `--allow-paid`. Used only as a fallback when explicitly authorized.

### Default per-stage routing

| Stage(s) | Model | Why |
|---|---|---|
| `clarify`, `premise`, `spine`, `chapter_plan`, `enhancement` | `qwen3:latest` | installed local structured-output default |
| `world_bible`, `prose` | `gemma4:26b` | installed local quality default for fact and prose-heavy stages |
| `embellish` | `gemma3:4b` | installed local small model for short texture details |

Override the whole routing at runtime:

```bash
story-writer new --idea "..." --model qwen3:latest         # one model everywhere
story-writer new --idea "..." --profile fast               # qwen3:latest everywhere
story-writer new --idea "..." --profile quality            # gemma4:26b everywhere
story-writer new --idea "..." --profile tiny               # gemma3:4b everywhere
```

Per-stage routing lives in `src/story_writer/config.py` (`DEFAULT_ROUTING`). Edit there if you want different per-stage assignments.

### Known reasoning-model caveat

Reasoning models can occasionally mode-collapse on the internal `reasoning` field of a `dspy.ChainOfThought` Signature, especially under tight `max_tokens` budgets — the model gets stuck in a repetitive loop and never emits the structured output. If you see `AdapterParseError`, retry with a higher `max_tokens` (edit `DEFAULT_ROUTING`) or lower temperature. Tracked as SPIKE-002 in `docs/requirements.md`.

## QA

QA runs after every chapter. Eight rules:

| Rule | Severity | Catches                                              |
|------|----------|------------------------------------------------------|
| R1   | hard     | Empty or whitespace-only chapter prose               |
| R2   | hard     | Meta-framing openers ("In this chapter…")            |
| R3   | hard     | AI assistant leaks ("as an AI", "I cannot")          |
| R4   | hard     | Affirmation openers ("Certainly!", "Here is…")       |
| R5   | hard     | Proper nouns not in the world bible                  |
| R6   | soft     | Missing expected world-bible anchors                 |
| R7   | soft     | Within-chapter sentence repetition                   |
| R8   | soft     | Cross-chapter sentence repetition                    |

Pass `--strict` to fail the run on any hard violation. `--skip-qa-embeddings` is accepted for CLI compatibility; the MVP QA implementation does not download embedding models by default.

## Artifacts on disk (per run)

Every stage persists its output. After `story-writer new --slug demo`, you get:

```
runs/demo/
├── manifest.json          stage statuses, models used per stage, timestamps
├── idea.txt               your raw idea
├── clarify.json           list of {question, suggested_answer, answer, source}
├── premise.json           protagonist / want / obstacle / stakes / summary
├── spine.json             6 spine beats
├── world_bible.json       rules, characters, locations, timeline
├── chapter_plan.json      ordered chapters with their assigned spine beats
├── chapters/NN.json       per chapter: beats + enhancement_notes + embellishment + prose
├── qa/NN.json             per chapter: each rule's pass/fail/flag with offending spans
└── story.md               (after `story-writer render demo`) the final assembled story
```

Inspect any single artifact:

```bash
story-writer inspect demo                  # prints manifest
story-writer inspect demo --stage premise  # prints premise.json
```

## Layout

```
src/story_writer/
├── cli.py            # Typer app
├── orchestrator.py   # Stage walker + QA hand-off
├── run_store.py      # Filesystem persistence
├── providers.py      # Ollama / DeepSeek router
├── config.py         # Defaults, sizing, routing
├── interactive.py    # Rich prompts
├── render.py         # Final story export
├── stages/           # One DSPy module per stage
├── qa/               # Detection rules + embeddings
└── models/           # Pydantic story / run / qa models
```

## Development

```bash
pip install -e .[dev]
ruff check .
ruff format --check .
pytest                                # unit tests (no Ollama required)
pytest tests/integration/              # real-Ollama smoke; auto-skips if unreachable
```

## Documentation

- [`docs/project-brief.md`](docs/project-brief.md) — scope and success criteria
- [`docs/requirements.md`](docs/requirements.md) — FRs / NFRs / glossary
- [`docs/architecture.md`](docs/architecture.md) — module graph, signatures, data model
- [`docs/sprint-backlog.md`](docs/sprint-backlog.md) — epics → tasks
- [`docs/decisions.md`](docs/decisions.md) — every decision with rationale
