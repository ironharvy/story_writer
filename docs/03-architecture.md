# 03 · Architecture / High-Level Design

How the pieces fit. Algorithms are in [04-algorithms.md](04-algorithms.md).

## Data flow

```
idea ─▶ clarify ─▶ premise ─▶ spine (Pixar 7-step)
                                  │
                                  ▼
                      world bible: rules · locations(+enhance)
                                   · timeline · characters(+enhance)
                                  │
                                  ▼
                         sanity check ─▶ chapter plan (N × {title, beats})
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                    ▼
     A: enhance_chapter    B: WorldState        C: independent
     + story-so-far        init/advance/draft   per-chapter draft
              └───────────────────┼────────────────────┘
                                  ▼
                     markdown artifact (incremental)
                                  ▼
              QA suite · POV check · linter · HTML render
```

Foundation is identical across variants; the variants differ only in how
chapters are drafted (inter-chapter continuity strategy).

## Components

| Layer | Module(s) | Responsibility |
|---|---|---|
| Entry points | `mymain.py` (A), `mymain_ws.py` (B), `mymain_module.py` (C), `story.py` (interactive) | Parse args, configure runtime/logging, kick off a run |
| Orchestration | `pipeline.py` (A), `pipeline_ws.py` (B), `pipeline_module.py` (C) | Sequence the stages, log steps, write artifacts |
| Story modules | `story.py` | DSPy signatures + `run_*` functions for every foundation/draft step; `act_hint_for_chapter` |
| World state | `world_state.py`, `story_module.py` | Variant B/C drafting state + signatures |
| Quality | `qa.py`, `pov_check.py`, `story_linter.py` | Post-generation checks + prose lint |
| Runtime | `dspy_runtime.py`, `dspy_optimization.py` | `DSPyConfig` / `configure_dspy`; module loading/optimization |
| Output | `artifact.py` | `initialize_artifact`, `update_artifact` (incremental markdown) |
| UI | `ui.py` | Rich prompts/review (kept out of pipeline logic) |
| Cross-cutting | `logging_config.py`, `_compat.py`, `exceptions.py`, `image_gen.py` | Logging, Langfuse `@observe()` fallback, errors, images |
| Scripts | `scripts/` | `run_qa.py`, `check_pov.py`, `lint_story.py`, `render_story.py`, `word_count.py`, `optimize_text_pipeline.py` |

## The three drafting variants

| Variant | Entry / pipeline | Inter-chapter continuity |
|---|---|---|
| **A · baseline** *(default)* | `mymain.py` / `pipeline.py` | per-chapter `enhance_chapter` + rolling text "story so far" |
| **B · world-state** | `mymain_ws.py` / `pipeline_ws.py` + `world_state.py` | structured `WorldState`, updated each chapter |
| **C · module** | `mymain_module.py` / `pipeline_module.py` + `story_module.py` | none — each chapter drafted independently from its own outline |

## External dependencies & runtime

- **LLM via DSPy** — default provider `ollama` (expected at
  `http://localhost:11434`), model `qwen3`; any DSPy/litellm model works
  (e.g. hosted `deepseek-v4-pro`).
- **Pydantic / dataclasses** — typed pipeline state and structured LM I/O.
- **Rich** — interactive review UI.
- **Langfuse** *(optional)* — tracing via `@observe()`; no-op shim if absent.
- **Replicate** *(optional)* — image generation.
- **Config** — CLI flags + `.env` (`API_KEY`, `DEEPSEEK_API_KEY`,
  `DSPY_CACHE_DIR`, `LANGFUSE_*`). DSPy disk/memory cache on by default.

## Artifact format (output markdown)

`# Story` → `## Generation Parameters / Story Title / Core Premise / Spine` →
`## World bible` (`### Rules / Locations / Timeline / Characters`, with
`#### Location N` / `#### Character N`) → `## Chapters Plan` → `## Final Story`
(`### Chapter N: <title>`). The QA/linter parsers key off exactly this layout.
