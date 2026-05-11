# Pipeline known issues

Last reviewed: 2026-05-10, against the **`qwen3.6:27b` + variant A (baseline)** path —
the recommended quality config (see `README.md` and
`docs/model-implementation-comparison-2026-05-10.md`).

Ordered roughly by how much they hurt that path.

## 1. Truncated / `None` LM output is a hard crash (no retry, no guard) — **highest priority**

`qwen3.6:27b` is verbose. With the default `--max-tokens 4096`, chapter-draft and
world-state calls hit the token ceiling exactly, the response is truncated, DSPy fails to
parse the structured field and returns `None`, and the pipeline dies on the next use of it:

- `pipeline.py` — `len(enhanced)` on `enhanced is None` → `TypeError`
- `world_state.py` — `render_world_state(None)` → `AttributeError: 'NoneType' has no 'story_clock'`
- `story_module.py` — same `None` → `TypeError`

This took down the entire `qwen3.6:27b` run on the first pass. Workaround today: pass
`--max-tokens 8192`. Real fix: (a) raise the default, **and** (b) wrap LM-producing steps in
a bounded retry plus a `None`/empty guard so one bad response doesn't kill a ~90-minute run.
Bonus: a checkpoint/resume so a run that dies at chapter 6 doesn't need the exact same
process to restart and rely on the DSPy cache.

## 2. Chapter 1 (the cold open) has no protagonist name yet

Variant A drafts chapter 1 before the protagonist is named anywhere, so the prose says
"the protagonist" / "the child" literally. (Variant B avoids this because its `WorldState`
hand-off carries the name into chapter-1 drafting; variant C is worse — "the child" ×16 in
ch1.) Fix: establish the protagonist's name in the foundation (premise/world bible) and feed
it into chapter-1 drafting, the way B does.

## 3. Model-tic repetition leaks across chapters

`qwen3.6:27b` reuses signature phrases — e.g. "silver suppression runes etched into …",
"the heat in their veins", an italicised *Ash-Burn* on every occurrence — in every variant.
QA's `cross_chapter_phrase_reuse` flags it but nothing acts on it. Fix: feed prior chapters'
sentences into the drafter as avoid/negative context.

## 4. QA `character_presence` mis-parses the world bible

It treats bold list items in the world-bible Characters section ("**Reclaim Identity**",
"**Protect the Nursery**", "**Subvert the Tithe**", …) as character names, then reports them
as "never appears in chapters" → spurious `FAIL` on every `qwen3.6:27b` run. Tighten the
extractor to the actual `#### Character N` blocks (and their names), not every bolded span.

## 5. QA can't see POV / pronoun drift

Variant C's real defect at 27B is that the narrative voice drifts chapter-to-chapter
(third-person "Cinder" → full first-person in ch5 → "the protagonist" in ch6). `name_drift`
only tracks proper nouns, so this goes unflagged — while `name_drift` *does* fire on harmless
paraphrases ("High Cardinal Vane" ↔ "High Clergy") and on `WorldState`-block parsing
artifacts ("Renn\n\nThe Sanctum", "Soren the Ash" ↔ "Soren Vael"). Add a
narrator-person / protagonist-naming consistency check; relax the proper-noun matcher so
paraphrase variants don't FAIL.

## 6. QA still doesn't flag empty / very-short prose

Pre-existing: an empty chapter passes R1–R6 (`runs/demo-hollow` chapter 3 was empty and
passed). Add a hard rule: chapter prose below N words fails.

## 7. Variant B leaks pipeline scaffolding into the artifact

`pipeline_ws.py` renders a `#### World State after Chapter N` block into the story markdown
after every chapter — a human reader has to skip seven of them, and it inflates word-count /
repetition metrics (~2× the file size, hundreds of bogus repeated n-grams from the block
headers). Render the state to a sidecar file, or strip it before the "Final Story" section.
(Not on the recommended path, but relevant if B is kept or merged into A.)

## 8. Chapters are thin on the fast model

`qwen3:latest` writes ~250-word chapters regardless of `--max-tokens`, so a 7-chapter "story"
is ~2k words. Fine for smoke tests; if it's ever used for real output, the drafter needs an
explicit length target.

## Not an issue (clearing the record)

- **`gemma4:26b` "fails miserably"** — that was issue #1 (the 4096-token crash), not the
  model. At `--max-tokens 8192` on the current pipeline it completes all 7 chapters; it's
  just lower quality (garbled/dropped words, unnamed protagonist) than `qwen3.6:27b`.
- **Variant C "deterministically crashes" on `qwen3:latest`** — also issue #1: `build_outline`
  JSON was truncated at a deterministic point. At `--max-tokens 8192` it runs clean.
