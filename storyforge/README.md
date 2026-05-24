# StoryForge

Turn a one-line story idea into a cohesive, complete, *shippable* manuscript —
and prove it with a runnable evaluation harness.

StoryForge is a from-scratch, **outline-first** story generator (premise → world
bible → spine → drafted chapters) with a bounded critic→revise pass, built around
a local Ollama model and an independent LLM judge. It implements
[`STORY_EVAL_SPEC.md`](../STORY_EVAL_SPEC.md) as its objective function: a draft
only "ships" when it clears every hard gate and the rubric threshold.

See [`DESIGN_NOTE.md`](DESIGN_NOTE.md) for the craft research, the competing
algorithm designs I weighed, and why I chose this one.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r storyforge/requirements.txt
# A local Ollama with the models named below, reachable at http://localhost:11434
ollama pull qwen3:latest        # fast, for building/iterating
ollama pull qwen3.6:27b         # quality + independent judge
```

API keys (Groq / DeepSeek / Langfuse) are read from a `.env` in the repo root and
are optional — everything runs fully local by default.

## Generate

```bash
# Interactive: asks a few clarifying questions, then writes the book
python -m storyforge generate --idea "a lighthouse keeper at the end of the world"

# Non-interactive, fully specified, then score it:
python -m storyforge generate \
  --idea "a memory-eating fog swallows a city; one cartographer still remembers" \
  --genre "literary speculative" --tone "haunting, hopeful" \
  --pov limited --length novelette --model fast \
  --eval --judge quality
```

### Story length

`--length` is named after the word-count categories writers use; each preset
keeps chapters inside the eval's 800-2500w band, so bigger stories scale by
*chapter count*, not chapter size:

| Preset | ≈ Total | Shape |
|---|---|---|
| `flash` | ~900w | 1 chapter — flash fiction |
| `short` | ~5,000w | 4 chapters — short story |
| `novelette` *(default)* | ~12,000w | 8 chapters |
| `novella` | ~24,000w | 14 chapters |
| `novel` | ~50,000w | 28 chapters — a multi-hour local run |
| `epic` | ~108,000w | 60 chapters — stress test, many hours |

Old names still work as aliases (`standard`→`novelette`, `long`→`novella`). For
ad-hoc sizes, `--chapters N` and `--words N` override the preset. Long runs write
incrementally and are resumable with `--resume --run-dir …`.

Output lands in `runs/<slug>-<timestamp>/`:

```
state.json        full run state (resumable)
chapters/chNN.md  each chapter, written the moment it's drafted
manuscript.md     the assembled book (STORY_EVAL_SPEC.md shape)
scorecard.json    the eval result (with --eval)
```

Because every artifact is written incrementally, an interrupted run loses nothing
and can be resumed:

```bash
python -m storyforge generate --resume --run-dir runs/<dir>
```

## Evaluate

```bash
python -m storyforge eval runs/<dir>/manuscript.md \
  --idea "the original idea" --judge quality
```

The harness emits a JSON scorecard — each check → severity + message, the rubric
scores, and a final `ship: true|false`. Exit code is `0` when it ships.

- **Tier 1** (deterministic, no LLM): chapter length, character presence, name
  drift, cross-chapter phrase reuse, content-word over-repetition.
- **Tier 2** (LLM judge): POV consistency, protagonist naming, continuity, premise
  fidelity.
- **Tier 3** (LLM judge): a 1–5 rubric over arc, ending, agency, scene-vs-summary,
  prose, cohesion. Ship threshold: average ≥ 4.0, no axis < 3.

Use `--tier1-only` for fast deterministic loops while iterating.

## Model strategy

Local Ollama first (per the brief). Build/iterate on `qwen3:latest`, validate prose
on `qwen3.6:27b`, and escalate to hosted `groq` / `deepseek` (via litellm) only when
the local models can't clear the bar. The eval judge is **independent** of the
drafter to avoid self-preference bias; the scorecard always records which models
were used.

### Choosing a model

`--model` and `--judge` each accept either form:

- **A preset**, when you don't care about the exact id: `fast` (qwen3:latest),
  `quality` (qwen3.6:27b), `groq`, `deepseek`.
- **An explicit `provider/model`**, e.g. `ollama/qwen3.6:27b`,
  `deepseek/deepseek-chat`, `groq/llama-3.3-70b-versatile`, or any other litellm
  id. The natural `ollama/…` prefix is normalized to litellm's chat endpoint
  (`ollama_chat/…`) automatically, so `--model ollama/qwen3.6:27b` just works.

A bare name with no provider (e.g. `qwen3.6:27b`) is passed through unchanged and
will error unless it's a preset — name the provider explicitly. The resolved
model is printed at startup (`• Drafting with …`) so you can confirm what was
picked.

## Layout

```
config.py      env + model roles + Langfuse (OTEL) wiring
llm.py         guarded litellm client (retries, <think> stripping, JSON extraction)
models.py      typed artifacts + resumable RunState
prompts.py     stage prompts (each encodes a defense against an eval failure mode)
pipeline.py    the generator: premise → bible → spine → draft → critic→revise
evaluate/      the harness: parse + tier1/2/3 + judge + scorecard/ship gate
tests/         deterministic + judge-gated regression tests & fixtures
```

## Test

```bash
python -m pytest storyforge/tests -q        # Tier-2 tests auto-skip if Ollama is down
```
