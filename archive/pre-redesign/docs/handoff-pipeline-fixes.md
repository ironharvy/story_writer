# Handoff — pipeline fixes after the model bake-off

For the next session. Context lives in:
- `docs/model-implementation-comparison-2026-05-10.md` — the 3-models × 3-variants bake-off + recommendation.
- `docs/pipeline-known-issues.md` — the full prioritized issue list (this handoff is the action plan for it).
- `.tmp/cmp/` — raw bake-off outputs (gitignored): `<model>/{A_baseline,B_worldstate,C_module}.{md,log,console}`,
  `NOTES.md`, `REPORT.md`, the per-model `*.driver.log`, and `run_model.sh` (the driver — re-usable for re-running the matrix).
- PR **#101** (`docs/model-roles-and-known-issues` branch) — README rewrite + the two docs above + a `mymain.py` help fix. Docs-only; can merge independently.

Recommended config for any test run: **`--model qwen3 --num-ctx 24576 --max-tokens 8192`** (the small 8B; full A run ≈ 5 min).
For a real quality check, `--model qwen3.6:27b` (≈90 min for variant A). Drive runs unattended with `yes '' | python mymain.py ...`.
After a run: `python scripts/run_qa.py <story.md>` and eyeball the prose.

## Work items, in suggested order

### 1. `--max-tokens` truncation → `None` → hard crash  *(highest impact)*
- Bump the `--max-tokens` default in `mymain.py` from `4096` to `8192` (and check `story.py` / any other entry point for the same default).
- Wrap the LM-producing steps so a single bad/truncated/`None` response doesn't kill the run:
  - guard sites that currently assume non-`None`: `pipeline.py` (`len(enhanced)` around line ~159), `world_state.py` (`render_world_state(state.story_clock)` etc.), `story_module.py`, `pipeline_ws.py`, `pipeline_module.py`.
  - add a small bounded retry (2–3 tries) around each `dspy` call that returns a structured field; on persistent failure, fail *that chapter/step* with a clear logged error rather than crashing the whole pipeline (and write a placeholder so the artifact still renders).
- Stretch: a checkpoint/resume so a run that dies at chapter N can restart from N (today only the DSPy disk cache softens a same-process re-run).
- Verify: ran `qwen3.6:27b` at `--max-tokens 4096` and all three variants crashed exactly here; at `8192` all three completed. Repro the crash by forcing `--max-tokens 4096` with `qwen3.6:27b` (or any verbose model).

### 2. Chapter-1 cold-open has no protagonist name  *(user thinks this is the easiest)*
The drafter writes literal "the protagonist" / "the child" in ch1 because the name isn't established yet (variant B avoids it — its `WorldState` carries the name into ch1; variant C is worst).
Two ways, pick one:
- **Cheap, no LLM:** once the protagonist's name is known (it reliably appears by ch2 — "Cinder" in the bake-off, and/or is derivable from the world-bible Characters block), post-process chapter 1: literal string-replace the placeholder ("the protagonist", "the child", maybe "the Hunter") with the name. This is basically search-and-replace. Watch for: the placeholder appears mid-sentence, sometimes capitalized ("The protagonist"); only do it in ch1 prose, not the world-bible/plan sections.
- **Cleaner:** add a `protagonist_name` field to the foundation (have the premise/world-bible step name them) and feed it into chapter-1 drafting (the way `pipeline_ws.py` already does via `WorldState`). Then merge that behaviour into variant A.
- Verify: `grep -n "the protagonist\|the child" <ch1 of the rendered story>` should come back empty after the fix; re-read ch1.

### 3. QA: fail empty / very-short prose  *(small, contained)*
`qa.py` — add a hard rule: chapter prose below ~N words (start with N≈80) fails. Pre-existing gap: `runs/demo-hollow` ch3 was empty and passed R1–R6.

### 4. QA: fix `character_presence` parser  *(small, contained)*
`qa.py` — it currently extracts "character names" by grabbing bolded spans, so it treats world-bible list items ("**Reclaim Identity**", "**Protect the Nursery**", …) as characters → spurious FAIL on every `qwen3.6:27b` run. Key off the actual `#### Character N` blocks and their declared names instead. Also relax `name_drift` so paraphrases ("High Cardinal Vane" ↔ "High Clergy") and WorldState-block parse artifacts ("Renn\n\nThe Sanctum", "Soren the Ash" ↔ "Soren Vael") don't FAIL.

### 5. QA: add a POV / protagonist-naming consistency check across chapters  *(new, moderate)*
Variant C's real flaw at 27B is narration drift (third-person "Cinder" → full first-person in ch5 → "the protagonist" in ch6) and `name_drift` can't see it. Add a check that flags: (a) chapters whose dominant narrator-person differs from the rest, (b) chapters that use a placeholder ("the protagonist"/"the child") instead of the established name.

### 6. Per-chapter anti-repetition  *(moderate–large)*
Model tics leak across chapters in every variant ("silver suppression runes etched into…" — `qwen3.6:27b`; "kaleidoscopic nightmare of overlapping…" — `gemma4:26b`). Feed prior chapters' sentences (or the top repeated n-grams from `scripts/word_count.py` / the QA reuse list) into the drafter as avoid/negative context.

### 7. Variant B: stop leaking `#### World State after Chapter N` blocks into the artifact  *(small–moderate; only if B is kept/merged)*
`pipeline_ws.py` renders the state into the story markdown after every chapter. Render it to a sidecar file, or strip it before the "Final Story" section. (It also doubles B's file size and floods the QA repetition metric.)

### 8. Fast-model chapter length  *(small)*
`qwen3:latest` writes ~250-word chapters regardless of `--max-tokens`. If it's ever used for real output, give the drafter an explicit per-chapter word target.

## Cross-cutting note for whoever picks this up
The `qwen3.6:27b` + variant A path is the recommended quality config (README). Items 1, 2, 3, 4 are the ones that most directly improve *that* path; 5–8 are quality polish. Items 3 and 4 are nearly free. Item 2 is the one the user flagged as probably a simple search-and-replace.
