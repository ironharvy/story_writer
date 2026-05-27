# Story Writer

A local-first DSPy pipeline that turns a one-line idea into a multi-chapter
story. Runs against Ollama (or any DSPy-supported LLM), with optional Langfuse
tracing.

> The contributor guide is **[AGENTS.md](AGENTS.md)**. Pre-rewrite code lives
> under `archive/` and is not used.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env

# Run with the default drafting strategy (baseline)
python main.py --idea "your story idea here" --title "Working Title" --number-of-chapters 7

# Pick a different strategy
python main.py --strategy world_state --idea "..." --title "..."

# See what's available
python main.py --list-strategies
```

Outputs are written incrementally to `--output-file` (default `.tmp/story.md`),
so an interrupted run keeps everything produced so far.

## Chapter-drafting strategies

The foundation (idea → premise → spine → world bible → chapter plan) is shared
and runs once per story; the **strategy** picks how that plan turns into prose.
Strategies are first-class plug-ins under `generators/`, registered via
`@register(id, status, description)`. The CLI's `--strategy <id>` and
`--list-strategies` both read from the registry.

| Strategy | Drafting | Continuity carry |
|---|---|---|
| **`baseline`** *(default)* | per-chapter via `run_enhance_chapter` | rolling free-text "story so far" summary |
| **`world_state`** | per-chapter via `run_draft_chapter_with_state` | structured `WorldState` (story clock / character knowledge / locations / plot threads / key objects / recent events), advanced after each chapter |
| **`dspy_module`** | per-chapter via a `dspy.ChainOfThought(DraftChapter)` over the shared plan | none — each chapter drafted independently from its plan entry + act-sliced spine + world bible |

Variant candidates that have been prototyped but not yet promoted live in
**[docs/deferred-generators.md](docs/deferred-generators.md)** with re-land
checklists — multi-agent TRPG simulation (PR #79), recursive story-on-itself
(PR #100), editor-in-chief wrap (PR #86).

The goal function the registry's generators are scored against is
**[bench/eval-spec.md](bench/eval-spec.md)**; the deterministic Tier-1
checker is `bench/criteria.py`.

See **[docs/model-implementation-comparison-2026-05-10.md](docs/model-implementation-comparison-2026-05-10.md)**
for a head-to-head of the three strategies across models, and
**[docs/pipeline-known-issues.md](docs/pipeline-known-issues.md)** for
current limitations.

## Recommended local models

| Role | Model | When to use |
|---|---|---|
| **Quality** | `qwen3.6:27b` (~27.8B, 17 GB, 256K ctx) | Best output by a wide margin — names its protagonist, dramatizes the premise, lands real endings. ~6–7k words for a 7-chapter run. Cost: ~1.5 h wall-clock per story on one GPU, ~24 GB VRAM at `--num-ctx 24576`. Use with `--strategy baseline`. |
| **Fast / smoke / dev** | `qwen3:latest` (~8.2B, 5.2 GB, 41K ctx) | End-to-end pipeline runs in ~5 min; fine for "does it run" and iteration where prose quality doesn't matter. Writes short (~250-word) chapters. |

`gemma4:26b` is not recommended — it completes but drops/garbles words and
never names the protagonist; `qwen3:latest` is the better fast model and
`qwen3.6:27b` the better quality model.

> Pin `--max-tokens 8192` (the 4096 default truncates verbose models, which
> currently crashes the run — see the known-issues doc). At `--num-ctx 24576`,
> `qwen3.6:27b`'s chapter-draft inputs run ~13–18k tokens, so leave headroom;
> bump `--num-ctx` if the log shows context truncation (max 262144).

## Key CLI flags (`main.py`)

- `--strategy` (default `baseline`; `--list-strategies` lists them)
- `--model` (default `qwen3`), `--provider` (default `ollama`), `--api-key`
- `--idea`, `--title`, `--number-of-chapters` (default 7)
- `--max-tokens` (default 4096 — **set 8192+**), `--num-ctx` (default 16384)
- `--cache` / `--no-cache`, `--memory-cache` / `--no-memory-cache`, `--cache-dir`
- `--output-file` (default `.tmp/story.md`), `--log-file` (default `.tmp/main.log`)
- `-v` / `-vv` / `-vvv` (INFO / LLM debug / full HTTP+LLM firehose)

## Environment

Copy `.env.example` → `.env`. Relevant keys: `API_KEY`, `DEEPSEEK_API_KEY`
(paid fallback), `DSPY_CACHE_DIR`, and Langfuse (`LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`). Ollama is expected at
`http://localhost:11434`.

## Utilities

```bash
python scripts/run_qa.py path/to/story.md            # QA detection suite (name drift, character presence, phrase reuse)
python scripts/word_count.py path/to/story.md        # top repeated content words in the Final Story section
python scripts/render_story.py story.md [out.html]   # render the markdown artifact to a self-contained HTML page
python -m bench.criteria path/to/story.md            # apply bench/eval-spec.md Tier-1 gates; emits JSON scorecard
```

## Tests

```bash
pytest --ignore=archive -q
```
