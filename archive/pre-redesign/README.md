# Story Writer

A local-first DSPy pipeline that turns a one-line idea into a multi-chapter story.
Runs against Ollama (or any DSPy-supported LLM), with optional Langfuse tracing.

> The detailed contributor guide is **[AGENTS.md](AGENTS.md)**. A pre-rewrite
> implementation lives under `archive/` and is not used.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env

# non-interactive run (accepts the idea up front, still confirms each step)
python mymain.py --idea "your story idea here" --title "Working Title" --number-of-chapters 7

# fully interactive flow (clarifying questions, per-run folder under runs/)
python story.py
```

Outputs are written incrementally, so an interrupted run keeps everything produced so far.

## Chapter-drafting variants

The foundation (clarify → premise → spine → world bible → chapter plan) is the same;
the three entry points differ only in how chapters are drafted afterward:

| Variant | Entry point | Drafting strategy |
|---|---|---|
| **A · baseline** *(default)* | `python mymain.py` | per-chapter `enhance_chapter` + a rolling plain-text "story so far" summary |
| **B · world-state** | `python mymain_ws.py` | structured `WorldState` (story clock / characters / locations / plot threads / objects / recent events), updated after each chapter |
| **C · dspy.Module** | `python mymain_module.py` | each chapter drafted independently from its own outline beats + world bible + act-sliced spine |

See **[docs/model-implementation-comparison-2026-05-10.md](docs/model-implementation-comparison-2026-05-10.md)**
for a head-to-head of the three variants across models, and
**[docs/pipeline-known-issues.md](docs/pipeline-known-issues.md)** for current limitations.

## Recommended local models

| Role | Model | When to use |
|---|---|---|
| **Quality** | `qwen3.6:27b` (~27.8B, 17 GB, 256K ctx) | Best output by a wide margin — names its protagonist, dramatizes the premise, lands real endings. ~6–7k words for a 7-chapter run. Cost: ~1.5 h wall-clock per story on one GPU, ~24 GB VRAM at `--num-ctx 24576`. Use **variant A**. |
| **Fast / smoke / dev** | `qwen3:latest` (~8.2B, 5.2 GB, 41K ctx) | End-to-end pipeline runs in ~5 min; fine for "does it run" and iteration where prose quality doesn't matter. Writes short (~250-word) chapters. |

`gemma4:26b` is not recommended — it completes but drops/garbles words and never names the
protagonist; `qwen3:latest` is the better fast model and `qwen3.6:27b` the better quality model.

> Pin `--max-tokens 8192` (the 4096 default truncates verbose models, which currently
> crashes the run — see the known-issues doc). At `--num-ctx 24576`, `qwen3.6:27b`'s
> chapter-draft inputs run ~13–18k tokens, so leave headroom; bump `--num-ctx` if the log
> shows context truncation (max 262144).

## Key CLI flags (`mymain.py`, shared by `mymain_ws.py` / `mymain_module.py`)

- `--model` (default `qwen3`), `--provider` (default `ollama`), `--api-key`
- `--idea`, `--title`, `--number-of-chapters` (default 7)
- `--max-tokens` (default 4096 — **set 8192+**), `--num-ctx` (default 16384)
- `--cache` / `--no-cache`, `--memory-cache` / `--no-memory-cache`, `--cache-dir`
- `--output-file` (default `.tmp/story.md`), `--log-file` (default `.tmp/mymain.log`)
- `-v` / `-vv` / `-vvv` (INFO / LLM debug / full HTTP+LLM firehose)

## Environment

Copy `.env.example` → `.env`. Relevant keys: `API_KEY`, `DEEPSEEK_API_KEY` (paid fallback),
`DSPY_CACHE_DIR`, and Langfuse (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
`LANGFUSE_BASE_URL`). Ollama is expected at `http://localhost:11434`.

## Utilities

```bash
python scripts/run_qa.py path/to/story.md [...]      # QA detection suite (name drift, character presence, phrase reuse)
python scripts/word_count.py path/to/story.md        # top repeated content words in the Final Story section
python scripts/render_story.py story.md [out.html]   # render the markdown artifact to a self-contained HTML page
```

## Tests

```bash
pytest -q
```
