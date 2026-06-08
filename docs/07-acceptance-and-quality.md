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
   gestured at). Judged by the Tier-3 `rubric` (`ending` axis) when it runs;
   reported as a `manual` check otherwise.
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
| No factual contradictions | `continuity` | 2 | fail (hard) / warn (minor) |
| Story dramatizes the premise | `premise_fidelity` | 2 | fail (needs `--idea`) |
| **Real ending dramatizes the spine** | `rubric` | 3 | fail (rubric-judged; `manual` if not run) |
| Six-axis quality bar (avg ≥ 4, no axis < 3) | `rubric` | 3 | fail |

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

## Read-through rubric (1–5 each)

The six-axis rubric (arc · ending · agency · scene-vs-summary · prose ·
cohesion, defined in [bench/rubric.md](../bench/rubric.md)) is now scored by
the Tier-3 LLM judge (`bench/judge.py`, via `scripts/evaluate.py --with-llm`),
not only by hand. For variance control, pass `--rubric-samples N` to score the
rubric N times and take the per-axis median; prefer an independent judge model
(different from the generator). A human read-through over the same axes is still
the gold standard for comparing models/variants (see
[model-implementation-comparison-2026-05-10.md](model-implementation-comparison-2026-05-10.md)).

## Test strategy (code)

- **Framework:** `pytest` (`pytest -q`). CI runs in `.github/workflows/ci.yml`.
- **LLM isolation:** unit tests configure a `MockLM` and assert on parsed
  outputs / Pydantic validation rather than calling a real model.
- **Coverage today:** `test_qa.py`, `test_pov_check.py`, `test_story_linter.py`,
  `test_story_module.py`, `test_world_state.py`, `tests/test_evaluate.py`
  (scorecard wiring), `tests/test_judge.py` (judge logic), `tests/test_bench.py`.
  The judge tests inject precomputed verdicts so the gate logic is covered
  without a model. New checks/algorithms should ship with tests mirroring this.
- **Static quality:** `ruff` (lint + format), and optionally `pylint` / `radon`
  / `vulture` per `AGENTS.md`.

## Model bake-off (release-level validation)

Before changing the default model/variant, re-run the matrix (models × variants)
on a fixed idea/title/chapter count and compare: completion/robustness, prose
word counts, repeated-5-gram counts, real QA fails, and the read-through rubric.
Driver and raw outputs live under `.tmp/cmp/` (gitignored); methodology and the
last results are in the comparison doc above.
