# Plan — Pre-Schema Preparation Phase

> **Status (handoff for a fresh session):**
> - ✅ **Step 1 — Evaluation Rubric** → `design/evaluation-rubric.md` (committed)
> - ✅ **Step 2 — Lessons from Prior** → `design/lessons-from-prior.md` (committed)
> - ✅ **Step 3 — Craft Vocabulary** → `design/craft-vocabulary.md` (committed)
> - ⬜ **Step 4 — Tracer Bullet** → remains. Build `src/story_writer/` and run locally against Ollama.
>
> Steps 1–3 needed no model and were completed in the design session. **Step 4 is the only work left in this phase** and is the reason for moving to an Ollama-equipped environment. Read `design/concept.md` first (the living design theory), then this plan's Step 4, then `design/evaluation-rubric.md` (what the output is judged against).
>
> Confirmed since planning: `deepseek-v4-pro` is real (`deepseek/deepseek-v4-pro` via litellm, released 2026-04-24, Thinking mode on by default, needs a large `--max-tokens` because reasoning tokens count against the output cap). It's the "max-quality, network-OK" cloud option, not the local default.

## Context

We've just reset the design conversation and established `design/concept.md` as the living theory: a service that generates novel-length cohesive stories via a pipeline (Idea → Premise → Bible → Outline → Manuscript), with cohesion enforced by constraining generation (Bible extract + Outline beat → Scene Generator) and verifying after (Consistency Checker → revision loop).

The obvious next move would be to design the Bible's schema, the Extract interface, and the Outline format in detail. But the books we're operating from (Ousterhout, Pragmatic Programmer, Brooks, Evans, Beck) all push toward a different next move: **before designing those interfaces, prepare four things that ground the design in evidence and craft.**

Without these preparations, schema design is guessing. With them, schema design responds to evidence.

Two key facts the Explore phase surfaced make this preparation cheap and high-leverage:
- `archive/pre-redesign/` is not a dead repo — it contains three working pipeline variants and four cohesion-enforcement modules (`qa.py`, `story_linter.py`, `pov_check.py`, `world_state.py`) that we can port or learn from.
- `archive/pre-redesign/docs/pipeline-known-issues.md` ranks seven concrete failure modes from prior runs — protagonist not named in ch.1, phrase-tic repetition, POV drift across chapters, etc. This is a rubric in negative form.

This plan covers a coherent **pre-schema preparation phase** with four sequential deliverables. Each is small enough to review and revise independently. After all four, we'll have an evaluation rubric, mined lessons, a craft vocabulary, and an actually-running tracer bullet — and *then* we design the Bible schema with grounded knowledge of what queries it needs to support and what cohesion failures it needs to prevent.

## Approach

Four steps. Each step produces one or more documents in `design/` or code under a new `src/story_writer/` package. Natural checkpoints between steps for user review.

### Step 1 — Evaluation Rubric (`design/evaluation-rubric.md`)

Operational criteria for "cohesive" and "interesting." Without this, every subsequent iteration is in the dark — we cannot tell if a change made the output better.

Composed from two sources:

- **Cohesion criteria** adapted from `archive/pre-redesign/docs/pipeline-known-issues.md` (the seven ranked problems) and the existing checks in `qa.py` / `story_linter.py` / `pov_check.py`. Includes: name consistency, POV consistency, character presence, chapter length, phrase-tic repetition, world-rule consistency, plot-thread payoff, no-contradiction-with-Bible.
- **Interestingness criteria** distilled from craft references in step 3. Includes: clear controlling idea, escalating stakes, character agency, scene-level dramatic value, satisfying turn/payoff per scene, voice distinctness.

Scoring format: severity-graded findings (`info`/`warn`/`fail`) for mechanical checks, plus a short human-judged section. Mechanical checks become tests; human-judged criteria stay rubric-only until we figure out how to automate them.

### Step 2 — Lessons from Prior (`design/lessons-from-prior.md`)

A concise writeup mining `archive/pre-redesign/`. Based on the Explore inventory we already have. Captures:

- The three pipeline variants attempted (baseline / world-state / module) and what each tried to solve.
- The four cohesion-enforcement layers that exist (`qa.py` heuristics, `story_linter.py` LLM find/replace, `pov_check.py` LLM POV classification, `world_state.py` structured state).
- The seven known issues from `docs/pipeline-known-issues.md`.
- A reusability matrix: **port directly** (qa, linter, pov_check, exceptions, artifact), **read for lessons** (world_state, pipeline_ws structure, story_module), **superseded** (variant A's rolling text summary, variant C's no-state drafting).

This is mostly an editorial pass on the Explore output, plus a short read of `docs/pipeline-known-issues.md`, `docs/model-implementation-comparison-2026-05-10.md`, and `docs/handoff-pipeline-fixes.md`.

### Step 3 — Craft Vocabulary (`design/craft-vocabulary.md`)

Evans's knowledge crunching applied to fiction. Embeds craft references into our ubiquitous language so the Bible and Outline can represent what makes stories work — not our invented reinventions of established concepts. Includes:

- Save the Cat beat sheet (Snyder) — opening image, theme stated, setup, catalyst, debate, break into two, B story, fun and games, midpoint, bad guys close in, all is lost, dark night of the soul, break into three, finale, final image.
- McKee *Story* — controlling idea, scene-driving values, story arc, story design (archplot / miniplot / antiplot).
- Three-act structure (and four-act / five-act variants).
- Hero's Journey (Campbell / Vogler).
- Scene-and-sequel (Bickham), motivation-reaction units (Swain).
- Freytag's pyramid (exposition / rising / climax / falling / dénouement).
- Spine (Pixar) — already in concept.md; cross-reference.
- POV taxonomy (first / second / third person; limited / omniscient; close / distant).
- Show-don't-tell, status changes, dramatic irony.

Each term gets a one-line definition in our voice plus a source attribution. This becomes the canonical reference for what the Bible and Outline must be able to represent.

### Step 4 — Tracer Bullet (`src/story_writer/` + run + findings)

The smallest end-to-end pipeline that takes an Idea and produces a short-story-length Manuscript (~2,000-3,000 words). Stupid implementations of every stage. The goal is to *feel* where cohesion breaks first, not to produce a good story.

Structure:

- `src/story_writer/__init__.py`
- `src/story_writer/cli.py` — entry point: `python -m story_writer --idea "..."`
- `src/story_writer/runtime.py` — DSPy LM configuration (provider, model, keys, caching). Adapted from `archive/pre-redesign/dspy_runtime.py`.
- `src/story_writer/pipeline.py` — orchestrator that runs the stages in order and persists each artifact.
- `src/story_writer/signatures.py` — DSPy `Signature` classes, one per stage (the typed input→output contracts). This is where prompts live, as signature instructions + field descriptions.
- `src/story_writer/stages/` — one DSPy module per stage:
  - `premise.py` — `dspy.Predict`/`ChainOfThought` over a Premise signature: Idea → Premise (central conflict, story question, genre, tone).
  - `bible.py` — Premise → Bible (free-form markdown text field; world rules, characters, plot skeleton, voice). Deliberately *not* a structured schema — we want to feel the pain of free-form Bible to motivate the real schema later.
  - `outline.py` — Bible → Outline (list of scenes; per-scene: who, where, beats, setups, payoffs).
  - `scene.py` — per scene: (Bible + Outline + immediately-preceding scene) → Scene prose.
- `src/story_writer/evaluate.py` — runs the ported cohesion checks against the produced Manuscript and prints findings against the rubric from step 1. Reuses logic from `archive/pre-redesign/qa.py`, `story_linter.py`, `pov_check.py` (all already DSPy-based).
- `stories/` — output directory, `.gitignore`'d. One subdirectory per Story containing `idea.md`, `premise.md`, `bible.md`, `outline.md`, `manuscript.md`, `evaluation.md`.

Tech stack:
- Python 3.12+ (`requires-python = ">=3.12"`).
- **DSPy** as the LLM programming layer — signatures + modules per stage. Aligns with the prior code, so ported checks (`qa.py`, `story_linter.py`, `pov_check.py`) drop in cleanly, and we get prompt optimization (teleprompters) available later without rework.
- **LLM provider: local Ollama first** (user's choice). DSPy talks to Ollama via LiteLLM using `ollama_chat/<model>` with `api_base=http://localhost:11434`. Verified-available models to try (checked against the live Ollama library, May 2026):
  - `ollama_chat/qwen3.6:27b` — 17GB, 256K ctx (quality; thinks by default — watch for stiff prose).
  - `ollama_chat/qwen3:14b` (9.3GB) or `ollama_chat/qwen3:8b` (5.2GB) — faster. *(Note: `qwen3:9b` does not exist; these are the closest.)*
  - `ollama_chat/nemotron3` — NVIDIA Nemotron 3 (agentic/reasoning-tuned); another data point.
- **Cloud is a config swap, not a rewrite.** The same `STORY_WRITER_LM` env var also accepts `groq/<model>` or `deepseek/deepseek-chat` (with `GROQ_API_KEY` / `DEEPSEEK_API_KEY`), so we can run cloud later without touching code — consistent with concept.md's "model/provider is a knob."
- Provider + model are **configurable via env var** (`STORY_WRITER_LM`, default `ollama_chat/qwen3.6:27b`). `runtime.py` may also expose `num_ctx` (the prior code controlled this) — for the short-story tracer the default context is fine, so this stays minimal.
- `pyproject.toml` for dependencies (`dspy`, plus whatever LiteLLM pulls in). No `requirements.txt`.

Run protocol (split, since generation runs on the user's machine):
- I write the code and 3-5 seed Ideas (a few I propose, the user can swap in their own).
- **The user runs** `python -m story_writer --idea "..."` locally against their Ollama models, for each seed Idea, and commits/shares the resulting `stories/<slug>/` outputs (or pastes them).
- We review the outputs together and score against the rubric (step 1). I write up observations in `design/tracer-bullet-findings.md`.
- The findings doc explicitly answers: *Where did cohesion break first? Which artifact's content was hardest to write a good prompt for? Where was structure missing and painful? Where was free prose actually fine?* Plus a quick read on which of the three models (qwen3.6:27b / qwen3:14b / nemotron3) produced the most usable output.

The findings doc is the input to the *real* Bible schema design — the work after this phase.

## Critical Files

**To create:**
- `design/evaluation-rubric.md`
- `design/lessons-from-prior.md`
- `design/craft-vocabulary.md`
- `design/tracer-bullet-findings.md` (after step 4 run)
- `src/story_writer/__init__.py`, `cli.py`, `runtime.py`, `pipeline.py`, `signatures.py`
- `src/story_writer/stages/{premise,bible,outline,scene}.py`
- `src/story_writer/evaluate.py`
- `pyproject.toml` (`requires-python = ">=3.12"`, deps: `dspy`)
- `.gitignore` additions for `stories/`, `.env`, and standard Python ignores

**To reference (read-only, source of patterns to port):**
- `archive/pre-redesign/qa.py:1-300` — six text-only checks (`check_name_drift`, `check_character_presence`, `cross_chapter_phrase_reuse`, `check_chapter_length`); proper-noun regex; `Finding` dataclass with severity. Port the check logic verbatim into `evaluate.py`, drop the file-aggregation harness.
- `archive/pre-redesign/story_linter.py` — LLM-driven find/replace for token-level fixes against a character list. Port the prompt structure + `Replacement` dataclass.
- `archive/pre-redesign/pov_check.py` — POV classification per chapter. Port directly as a check in `evaluate.py`.
- `archive/pre-redesign/world_state.py` — read for the `WorldState` dataclass pattern; do *not* port yet (Step 4 deliberately avoids structured state to feel the pain).
- `archive/pre-redesign/artifact.py` — markdown incremental append; inline if useful.
- `archive/pre-redesign/dspy_runtime.py` — DSPy LM configuration (it already has Ollama wiring + `num_ctx` handling); adapt into `src/story_writer/runtime.py`, default to `ollama_chat/qwen3.6:27b`, keep Groq/DeepSeek as env-selectable alternates.
- `archive/pre-redesign/docs/pipeline-known-issues.md` — read in full to inform the rubric (step 1).
- `archive/pre-redesign/docs/model-implementation-comparison-2026-05-10.md` — read for what the three variants actually produced.
- `archive/pre-redesign/docs/handoff-pipeline-fixes.md` — read for which known issues had work-in-progress fixes.

## Verification

- **Step 1**: `design/evaluation-rubric.md` exists. Includes both mechanical-check criteria (each maps to a check in `evaluate.py`) and human-judged criteria. Each criterion has a name, definition, severity, and at least one example failure.
- **Step 2**: `design/lessons-from-prior.md` exists. Includes the three-variant summary, four cohesion-enforcement layers, seven known issues, and reusability matrix. Cross-references specific files in `archive/pre-redesign/`.
- **Step 3**: `design/craft-vocabulary.md` exists. Each term has a one-line definition + source. Concept.md §4 gets a cross-reference appended pointing to this file.
- **Step 4 (what I verify in-container):** From a clean checkout on Python 3.12, `pip install -e .` succeeds; `python -m story_writer --help` works; the package imports; `evaluate.py` runs against a hand-written sample manuscript fixture (no live LM needed) and emits findings. I cannot run a 27B Ollama model here, so end-to-end generation is verified by the user.
- **Step 4 (what the user verifies locally):** With Ollama running, `python -m story_writer --idea "a knight discovers a dragon is the kingdom's last librarian"` produces a `stories/<slug>/` directory containing all five artifacts; `python -m story_writer evaluate stories/<slug>/` prints cohesion findings. Repeat for ≥3 ideas across the candidate models. User shares the outputs.
- **Findings:** `design/tracer-bullet-findings.md` exists and explicitly addresses the four run-protocol questions, written from the shared outputs.

## Out of scope (intentionally)

- Structured Bible schema, Extract interface, Outline format definition — these come *after* this phase, informed by it.
- Consistency Checker integrated into the pipeline (in-loop revision) — tracer's cohesion enforcement is post-hoc audit only.
- Constrain-on-generation hard constraints beyond a prompt — no Pydantic-schema-validated outputs, no retry-on-violation.
- DSPy optimization/compilation — we use plain `Predict`/`ChainOfThought` modules only. No teleprompters, no metric-driven compilation yet (that's a later lever, once we have an evaluation metric worth optimizing against).
- Async / jobs / database / web / auth / accounts — all of Phase 2.
- Novel-length scale — tracer is deliberately short-story (~2-3k words). Lessons inform novel design.
- Forking one of the three prior variants — we start fresh in `src/story_writer/`. We borrow modules, not architecture.
- Updating `design/concept.md` — the concept doc reflects target design; this phase's outputs are evidence for refining it later.

## Dependencies & risks

- **Generation runs are local (user's machine).** Since we target Ollama, Step 4's actual generation happens on the user's machine, not in this container. I deliver the code; the user runs it and shares outputs. This means a handoff round-trip before the findings doc can be written.
- **No in-container end-to-end verification of generation.** I can verify install/import/CLI/eval-on-fixture here, but not a real generation run (no Ollama / GPU). The user's run is the real test.
- **Ollama must be running locally** with the chosen model pulled (`ollama pull qwen3.6:27b`, etc.). The user already has these per their note.
- **Cloud later is low-risk** — `STORY_WRITER_LM=groq/...` or `deepseek/deepseek-chat` plus the matching key in `.env` switches providers with no code change. `.env.example` will document all three. (DeepSeek V4's exact model string post-dates my knowledge cutoff; confirm from live docs if/when we go cloud.)

## Pause points for user review

After each step, we pause and the user reviews the produced artifact. Natural places to revise scope, add criteria, change craft references, etc. The tracer bullet (Step 4) is the largest step and we may want a sub-checkpoint after the pipeline runs end-to-end on the first seed Idea, before running it on the full set.
