# 06 · Decisions, Constraints & Risks

## Architecture decisions (ADRs, condensed)

| # | Decision | Why | Trade-off |
|---|---|---|---|
| D-1 | **DSPy** as the LLM framework | Typed signatures + structured outputs + provider-agnostic (litellm) + `@observe()` tracing | Couples the codebase to DSPy's parsing; truncated output → `None` is a sharp edge (R-1). |
| D-2 | **Local-first, Ollama default** | No per-token cost, offline, privacy; fits hobbyist users | Quality ceiling/latency bound by local hardware; big models are slow (~90 min/story). |
| D-3 | **Pixar 7-step spine** | Cheap, well-known arc skeleton that maps cleanly onto 3 acts | Imposes one story shape; not ideal for every genre. |
| D-4 | **3-act spine slicing** (`act_hint_for_chapter`) | Stops later-act beats from leaking into early chapters | Heuristic 25/50/25 split; coarse for very short/long books. |
| D-5 | **Human-in-the-loop review loops** | Author corrects each foundation step before it locks in | Interactive runs need attention; non-interactive runs auto-accept. |
| D-6 | **Incremental artifact writes** | Interrupted runs keep their work; everything is inspectable | The single markdown file doubles as the data format; QA parsers are coupled to its heading layout. |
| D-7 | **Three drafting variants kept in parallel** | Lets us compare continuity strategies empirically (the bake-off) | Triplicated drafting code/entry points to maintain. |
| D-8 | **Separate post-generation QA** (text + LLM checks) | Catch the common failure modes without changing the generator | QA only flags; nothing auto-acts on findings yet. |

## Constraints

- Must run fully offline against local Ollama (`http://localhost:11434`).
- Output is a single markdown file in the layout in
  [03-architecture.md](03-architecture.md); tools depend on it.
- Spine is exactly 7 newline-separated beats in Pixar order (the slicer indexes
  positions 3 and 5).
- Code conventions in `AGENTS.md` are mandatory (≤50-line functions, typed,
  UI/business-logic separation).

## Assumptions

- The configured model can follow structured-output instructions reasonably.
- The chosen `--max-tokens` / `--num-ctx` leave headroom for the model's
  verbosity (the recommended quality config is `qwen3.6:27b`, `--num-ctx 24576`,
  `--max-tokens 8192`).
- The protagonist's name reliably appears by chapter 2 (used by the linter fix).

## Risk register

Full detail and priority order: [pipeline-known-issues.md](pipeline-known-issues.md);
action plan: [handoff-pipeline-fixes.md](handoff-pipeline-fixes.md).

| # | Risk | Impact | Mitigation / status |
|---|---|---|---|
| R-1 | Truncated/`None` LM output crashes the whole run (no retry/guard) | A 90-min run dies at chapter 6 | **Open, highest priority.** Workaround: `--max-tokens 8192`. Real fix: raise default + retry + `None`-guard + checkpoint/resume. |
| R-2 | Chapter-1 cold open has no protagonist name (variant A/C) | Literal "the protagonist"/"the child" in prose | Linter (`story_linter.py`) patches it; cleaner fix: name in foundation / feed into ch1 (as B does). |
| R-3 | Model-tic phrase repetition leaks across chapters | Lower prose quality | QA flags it; no anti-repetition feedback into the drafter yet. |
| R-4 | QA false positives (`character_presence` bold-bullet parse, `name_drift` paraphrases) | Spurious fails erode trust | Partly addressed (bold-only bullets skipped, drift is `warn`). |
| R-5 | POV / pronoun drift across chapters (esp. variant C) | Inconsistent narration | `pov_check.py` added; not folded into the main QA run. |
| R-6 | No character-name uniqueness check in the bible | Two characters share a name (seen with `deepseek-v4-pro`) | **Open.** Add a foundation-stage uniqueness check. |
| R-7 | Variant A can duplicate/loop late beats (summary-only continuity) | Repeated climax across ch5–7 | **Open.** Feed prior chapter's closing beats as "already happened"; or adjacent-chapter overlap check. |
| R-8 | QA parsers coupled to the markdown heading layout | A layout change silently breaks checks | Keep artifact format stable; covered by tests. |
