# 07 · Acceptance Criteria & Quality Strategy

How we decide a run — and the system — is good enough.

## Definition of done (one story run)

A run on the recommended config is acceptable when:

1. It **completes without crashing** and writes all artifact sections.
2. It produces **N non-empty chapters** (none below the length floor, 80 words).
3. The **protagonist is named** in chapter 1 (no "the protagonist"/"the child"
   placeholders left in prose).
4. The story has a **real ending** (the spine's final beats are dramatized, not
   gestured at).
5. The **QA suite reports no real fails** (parser-artifact fails don't count).
6. **Narration POV is consistent** across chapters (no unexplained drift).

## QA gates (automated)

Run after generation:

```bash
python scripts/run_qa.py path/to/story.md      # phrase reuse, name drift, presence, length
python scripts/check_pov.py path/to/story.md    # LLM POV consistency
python scripts/lint_story.py path/to/story.md   # placeholder/misspelling fixes (writes sidecar)
python scripts/word_count.py path/to/story.md   # top repeated content words
```

| Gate | Source | Pass condition |
|---|---|---|
| Chapter length | `qa.check_chapter_length` | every chapter ≥ 80 words |
| Character presence | `qa.check_character_presence` | every canonical character appears in prose |
| Name drift | `qa.check_name_drift` | no real misspelled-name variants (warn-level) |
| Phrase reuse | `qa.check_cross_chapter_phrase_reuse` | repeated-5-gram count low (warn-level) |
| POV consistency | `pov_check` | no `mixed` chapter; none disagreeing with the dominant POV |

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
