# 02 · Requirements (PRD + Functional + Non-Functional)

Scope-setting companion to [01-vision.md](01-vision.md). IDs are for reference,
not formal traceability.

## Personas & primary use cases

- **Author** — has an idea, wants a structured draft. Runs interactively
  (`story.py`) answering clarifying questions, or non-interactively
  (`mymain.py --idea ... --title ...`).
- **Tinkerer** — compares drafting strategies / models. Runs the three variants
  (`mymain.py`, `mymain_ws.py`, `mymain_module.py`) and the QA scripts.

## User stories

- As an author, I give a one-line idea and get a multi-chapter story file.
- As an author, I review and correct each foundation step before it locks in.
- As an author, an interrupted run keeps everything produced so far.
- As a tinkerer, I run the same idea through different models/variants and get a
  comparable QA report for each.

## Functional requirements

**Input & config**
- FR-1 Accept a story idea, optional title, and chapter count (default 7).
- FR-2 Accept model / provider / token / context / cache config via CLI flags
  and `.env`; default to local Ollama.

**Foundation**
- FR-3 Clarify the idea via generated questions + proposed answers, then fold
  answers back into an updated idea; generate a title if none given.
- FR-4 Generate a core premise.
- FR-5 Generate a story spine using the Pixar 7-step formula.
- FR-6 Generate a world bible: rules of the world, locations (+ enhanced),
  timeline, characters (+ enhanced).
- FR-7 Run a consistency sanity check across idea / spine / world bible.
- FR-8 Generate a per-chapter plan (title + beats) for N chapters.

**Drafting (three variants, shared foundation)**
- FR-9 **Variant A (baseline)** — draft each chapter from its beats + a rolling
  "story so far" text summary; map each chapter to a 3-act slice of the spine.
- FR-10 **Variant B (world-state)** — maintain a structured `WorldState`
  (clock / characters / locations / threads / objects / recent events), updated
  after each chapter and fed into the next draft.
- FR-11 **Variant C (module)** — draft each chapter independently from its own
  outline beats + world bible + act-sliced spine.

**Human-in-the-loop**
- FR-12 In interactive mode, present each generated step for review; allow
  free-text feedback that regenerates the step, or acceptance.

**Output**
- FR-13 Write all artifacts (params, idea, premise, spine, bible, plan,
  chapters) incrementally to a markdown file in a defined section layout.
- FR-14 Optionally generate images.
- FR-15 Render the markdown artifact to standalone HTML.

**Quality**
- FR-16 Provide a QA suite over the output: cross-chapter phrase reuse, name
  drift, character presence, chapter length.
- FR-17 Provide an LLM-backed POV-consistency check.
- FR-18 Provide a post-generation linter that replaces placeholder/misspelled
  character references in chapter prose only.

## Non-functional requirements

- NFR-1 **Local-first / offline** — must run fully against local Ollama with no
  API key.
- NFR-2 **Model-portable** — work across DSPy-supported LLMs (local + hosted);
  no hard dependency on one model's quirks.
- NFR-3 **Inspectable & incremental** — intermediate artifacts persisted as they
  are produced; an interrupted run retains prior output.
- NFR-4 **Observability** — structured logging with verbosity levels; optional
  Langfuse tracing via `@observe()`.
- NFR-5 **Resource budget** — runnable on a single consumer GPU; context window
  configurable (`--num-ctx`); recommended quality config documented.
- NFR-6 **Robustness** *(target; partially unmet)* — a single bad/truncated LM
  response should not crash a long run. See risk R-1 in
  [06-decisions-and-risks.md](06-decisions-and-risks.md).
- NFR-7 **Code quality** — conventions enforced per `AGENTS.md` (≤50-line
  functions, type hints, separation of UI/business logic, tests with `MockLM`).

## Out of scope

GUI/editor · accounts/multi-tenancy · cloud hosting · publication-grade
formatting · automatic acceptance of foundation steps without review (in
interactive mode).
