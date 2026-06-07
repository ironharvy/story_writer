# 07 · Acceptance Criteria & Quality Strategy

How we decide a run — and the system — is good enough.

## Definition of done (one story run)

A run on the recommended config is acceptable when:

1. It **completes without crashing** and writes all artifact sections.
2. It produces **N non-empty chapters** (none below the hard floor of 80 words;
   thin `<300` / bloated `>2500` chapters warn but don't block).
3. The **protagonist is named** in chapter 1 (no "the protagonist"/"the child"
   placeholders left in prose).
4. The story has a **real ending** (the spine's final beats are dramatized, not
   gestured at). *Not automatable yet — a `manual` check pending the Tier-3
   judge.*
5. The **QA suite reports no real fails** (parser-artifact fails don't count).
6. **Narration POV is consistent** across chapters (no unexplained drift).

## The unified scorecard (executable Definition of Done)

One command turns the list above into a machine-readable verdict:

```bash
python scripts/evaluate.py path/to/story.md             # Tier-1 only (no model)
python scripts/evaluate.py path/to/story.md --with-llm  # + Tier-2 POV & prose lint
python -m bench.evaluate   path/to/story.md             # Tier-1 only, no dspy needed (JSON)
```

It runs the deterministic Tier-1 gates plus (with `--with-llm`) the LLM-backed
Tier-2 gates, and writes a `<story>.scorecard.json` sidecar. Three top-level
verdicts:

- **`tier1_clean`** — no deterministic FAIL and budgets respected.
- **`complete`** — every required gate actually ran (Tier-2 not skipped).
- **`ship`** — `tier1_clean` **and** `complete`. A deterministic-only run is
  honest: `ship=false` because POV couldn't be checked.

Each Definition-of-Done item maps to a gate:

| DoD item | Gate | Tier | Severity |
|---|---|---|---|
| All required sections present | `structure` | 1 | fail |
| N non-empty chapters (≥ 80 words) | `chapter_length` | 1 | fail |
| Chapters in target band (300–2500) | `chapter_band` | 1 | warn |
| Protagonist named (no placeholders) | `placeholder_protagonist` | 1 | fail |
| Every cast member appears | `character_presence` | 1 | fail |
| No misspelled-name drift | `name_drift` | 1 | warn (budget ≤ 2) |
| Low cross-chapter phrase reuse | `cross_chapter_phrase_reuse` | 1 | warn (budget ≤ 5) |
| POV consistent across chapters | `pov_consistency` | 2 | fail |
| Placeholders / canonical-name lint | `prose_linter` | 2 | warn (advisory) |
| **Real ending dramatizes the spine** | — | 3 | **manual** |

Length thresholds live in `qa.py` (`CHAPTER_MIN_WORDS` = 80 hard floor;
`CHAPTER_TARGET_BAND` = 300–2500) — the single source of truth shared by
`bench/criteria.py`, `bench/evaluate.py`, and `bench/eval-spec.md`.

### Lower-level tools (building blocks)

```bash
python scripts/run_qa.py path/to/story.md      # the original 4 deterministic checks
python scripts/check_pov.py path/to/story.md    # LLM POV consistency only
python scripts/lint_story.py path/to/story.md   # placeholder/misspelling fixes (writes + applies)
python scripts/word_count.py path/to/story.md   # top repeated content words
```

Severity meaning: `fail` = must fix, `warn` = inspect, `info` = context.

## Read-through rubric (human, 1–5 each)

From the bake-off methodology — score every run on:
**continuity · coherence · structure · prose · plot-thread tracking.**
Used to compare models/variants beyond what automated QA can see (see
[model-implementation-comparison-2026-05-10.md](model-implementation-comparison-2026-05-10.md)).

## Test strategy (code)

- **Framework:** `pytest` (`pytest -q`). CI runs in `.github/workflows/ci.yml`.
- **LLM isolation:** unit tests configure a `MockLM` and assert on parsed
  outputs / Pydantic validation rather than calling a real model.
- **Coverage today:** `test_qa.py`, `test_pov_check.py`, `test_story_linter.py`,
  `test_story_module.py`, `test_world_state.py`. New checks/algorithms should
  ship with tests mirroring this pattern.
- **Static quality:** `ruff` (lint + format), and optionally `pylint` / `radon`
  / `vulture` per `AGENTS.md`.

## Model bake-off (release-level validation)

Before changing the default model/variant, re-run the matrix (models × variants)
on a fixed idea/title/chapter count and compare: completion/robustness, prose
word counts, repeated-5-gram counts, real QA fails, and the read-through rubric.
Driver and raw outputs live under `.tmp/cmp/` (gitignored); methodology and the
last results are in the comparison doc above.
