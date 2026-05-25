# 01 · Vision / Product Brief

## Problem

Turning a one-line story idea into a coherent, multi-chapter story by hand is
slow, and naive "write me a story" LLM prompts produce shapeless prose with no
through-line, drifting names, and forgotten characters. There is no
local-first, inspectable pipeline that takes an idea through the same structured
steps a human author would (premise → arc → world → outline → draft) and that
checks its own output.

## What we are building

**Story Writer** — a local-first [DSPy](https://dspy.ai) pipeline that turns a
one-line idea into a structured multi-chapter story, running against Ollama (or
any DSPy-supported LLM), with optional image generation and Langfuse tracing.

It works in two phases:

1. **Foundation** — clarify the idea, derive a core premise, a story spine
   (Pixar 7-step), a world bible (rules / locations / timeline / characters),
   and a per-chapter plan.
2. **Drafting + QA** — draft each chapter on top of that foundation, then run
   post-generation quality checks (phrase reuse, name drift, character
   presence, chapter length, POV consistency).

## Target users

- **Hobbyist / indie writers** who want a structured first draft from an idea,
  on their own hardware, with no per-token cost.
- **Pipeline tinkerers / researchers** experimenting with multi-step LLM
  generation, prompt structure, and quality evaluation (the repo ships three
  drafting variants and a model bake-off for exactly this).

## Goals

- One idea in → one readable, structured, multi-chapter story out.
- **Local-first**: usable fully offline against Ollama; no API key required.
- **Inspectable**: every intermediate artifact (premise, spine, bible, plan) is
  written to the output incrementally, so an interrupted run keeps its work.
- **Self-checking**: automated QA surfaces the common failure modes.
- **Model-portable**: the same pipeline runs across local and hosted models.

## Non-goals

- Not a polished, publication-ready editor or a GUI app (CLI + Rich prompts).
- Not a chat assistant or general-purpose writing tool.
- Not cloud-first or multi-tenant; no accounts, no hosting.
- Not aiming for a single "best" algorithm — the three drafting variants are
  kept in parallel for comparison, not collapsed.

## Success criteria

- A 7-chapter run completes without crashing on the recommended config and
  produces 7 non-empty chapters with a named protagonist and a real ending.
- The QA suite passes (no spurious fails) on a good run and flags the known
  failure modes on a bad one.
- A new contributor can read these docs + `AGENTS.md` and run the pipeline.

## Guiding principles

Local-first · incremental, inspectable artifacts · human-in-the-loop review at
each step · structure before prose · check what you generate.
