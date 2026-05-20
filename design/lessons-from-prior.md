# Lessons from the Prior Implementation

What `archive/pre-redesign/` tried, what it taught us, and what to carry forward. Sources: the code in `archive/pre-redesign/`, plus `docs/pipeline-known-issues.md`, `docs/handoff-pipeline-fixes.md`, and `docs/model-implementation-comparison-2026-05-10.md` (a 4-model × 3-variant bake-off on the idea *"a nameless child raised by the Church to hunt demons discovers the Church breeds the demons it sells protection from"*).

This is evidence, not gospel. The prior code is a thrown-away first system (Brooks). We mine it; we don't inherit its architecture.

## The pipeline it built

All variants shared one foundation, then differed only in how they drafted chapters:

**Foundation (shared):** idea → clarify idea → core premise → spine (three-act beats) → world bible (rules, locations, characters, timeline) → chapter plan (titles + beats).

This is strikingly close to our target pipeline (Idea → Premise → Bible → Outline → Manuscript). The vocabulary already overlaps — premise, spine, world bible, characters, locations, timeline. The main differences in our design: a *structured* Bible (theirs is markdown), a sealed *Extract* interface (theirs re-reads markdown), and explicit setups/payoffs in the Outline.

## The three drafting variants

| Variant | Inter-chapter continuity | Result |
|---|---|---|
| **A · baseline** (`pipeline.py`) | rolling plain-text "story so far" summary | Best all-rounder at 27B. **But** cold-open ch1 unnamed; with only a text summary, late beats can duplicate/loop. |
| **B · world-state** (`pipeline_ws.py` + `world_state.py`) | structured `WorldState` (clock, characters, locations, threads, objects, recent events), updated each chapter | **Best narration consistency, lowest repetition.** Names ch1 correctly (state carries the name in). Wart: rendered the state blocks *into* the artifact (UX bug). |
| **C · dspy.Module** (`story_module.py`) | none — each chapter drafted independently from outline + bible + act-sliced spine | Most prose, coherent arc, **but voice drifts chapter-to-chapter** (third person → first person in ch5 → "the protagonist" in ch6). |

**The single most important lesson:** state representation is the crux of cohesion. Plain-text summary (A) drifts and loops; no state (C) drifts worst; **structured state (B) wins** — most consistent narration, lowest phrase repetition, correct naming. This is direct empirical support for our central bet: a *structured* Bible, not prose. The better the structured memory, the better the cohesion.

## Model bake-off takeaways

- **`qwen3.6:27b`** — best local model. 3–4× the prose of the 8B models, dramatizes the premise, lands real endings, the only model that reliably *names* its protagonist. ~86 min for a 7-chapter variant-A run, ~24GB VRAM at `num_ctx 24576`. **Our Step-4 default.**
- **`deepseek-v4-pro`** (hosted, Thinking mode, released 2026-04-24, `deepseek/deepseek-v4-pro` via litellm) — best sentence-level prose and most surprising ending, ~2× faster, well under $1/run. But hosted (against local-first), drifts pronoun in ch3, and its back third stutters (ch5/ch6 beat duplication). Needs `--max-tokens 32768` because reasoning tokens eat the output budget. The "max-quality, network-OK" option, not the default.
- **`qwen3:latest` (~8B)** — coherent but thin (~260 words/chapter) and stiff endings; good fast smoke-test model.
- **`gemma4:26b`** — drops/garbles words ("shredly", "curdended"), never names the protagonist; same-speed sanity model, not a quality one.

Robustness is a *pipeline* problem, not a model one (see issue #1 below): every 27B variant crashed at `--max-tokens 4096` and completed at `8192`.

## The four cohesion-enforcement layers it built

All are **post-generation audits** — they flag, but nothing acts on the findings in-loop. This is the prior system's defining limitation, and exactly what our constrain → verify → **revise** loop is meant to fix.

1. **`qa.py` — four text-only heuristic checks**, each a pure function returning `Finding(check, severity, message)`:
   - `cross_chapter_phrase_reuse` — n-gram (n=5) phrases repeated across ≥2 chapters (model tics), excluding bible n-grams.
   - `name_drift` — chapter proper-nouns sharing a canonical character's first token but differing in full form (catches misspellings; also fires on harmless paraphrases — a known weakness).
   - `character_presence` — every canonical character must appear in chapter prose.
   - `chapter_length` — fail chapters under 80 words.
2. **`story_linter.py` — LLM find/replace** (DSPy `ChainOfThought`). Reads the Characters section + chapter prose, returns verbatim `Replacement(find, replace, reason)` pairs for placeholders ("the protagonist") and name misspellings. Applied to chapter prose only, with word-boundary-aware substitution and a `validate()` pass that drops edits whose `find` isn't present.
3. **`pov_check.py` — LLM POV classifier** (DSPy `ChainOfThought`). Classifies each chapter's dominant narration (first/third/mixed/other), then flags chapters that disagree with the dominant POV or shift within themselves. Built specifically because `name_drift` is blind to voice drift.
4. **`world_state.py` — structured state** (variant B). Not a checker per se, but the `WorldState` dataclass + LLM-driven advance step is the prior art for structured story memory — the seed of our Bible.

## The known issues (the rubric in negative form)

The prior runs surfaced a ranked failure catalogue. These ARE our cohesion criteria, stated as what goes wrong:

1. **Truncated/`None` LM output → hard crash** (highest impact). Verbose model hits the token ceiling → DSPy returns `None` → unguarded `len(None)` kills a 2-hour run. Fix: raise default max-tokens, bounded retry, `None`/empty guard, checkpoint/resume.
2. **Chapter-1 cold open has no protagonist name** → literal "the protagonist"/"the child". Fix: establish the name in the foundation and feed it into ch1.
3. **Model-tic phrase repetition across chapters.** Fix: feed prior chapters' sentences as negative/avoid context.
4. **`character_presence` mis-parses the bible** — treats bold list items ("**Reclaim Identity**") as character names → spurious FAIL. Fix: key off real `#### Character N` blocks. *(In our design: structured Bible eliminates this whole class of markdown-scraping bug.)*
5. **QA can't see POV/pronoun drift** while `name_drift` over-fires on paraphrases. Fix: dedicated narrator-person check (built as `pov_check.py`); relax the proper-noun matcher.
6. **Empty/very-short prose passes.** Fix: hard minimum word count.
7. **Variant B leaks `#### World State after Chapter N` into the artifact.** Fix: sidecar the state. *(In our design: the Bible is never rendered into the Manuscript.)*
8. **Character-name uniqueness unenforced** — `deepseek-v4-pro` produced *two* characters named Elara; the prose lampshaded it. `name_drift` can't see a collision. Fix: foundation-stage uniqueness check.
9. **Per-chapter drafting duplicates/loops late beats** — with only a text summary, ch6's climax bled into ch5, then ch6 re-ran it, then ch7 replayed ch6's cliffhanger. Fix: feed the prior chapter's *actual closing beats* as "already happened, don't repeat"; adjacent-chapter overlap check; structured state would catch "the relic is already in the protagonist's possession."

## Reusability matrix

**Port directly** (with the fixes above):
- `qa.py` — the four checks. Drop the file-aggregation harness; apply fix #4 (parse structured characters), #5 (relax matcher), and keep #6 (length).
- `story_linter.py` — the find/replace prompt + `Replacement` model + word-boundary `apply()`.
- `pov_check.py` — the POV classifier + consistency check.
- `exceptions.py` — the recoverable-exception tuples.
- `artifact.py` — markdown incremental append (inline if useful).

**Read for lessons** (don't copy wholesale):
- `world_state.py` — the `WorldState` dataclass is the conceptual seed of the structured Bible; study how it's advanced per chapter.
- `pipeline_ws.py` — the "build foundation once, then loop chapters carrying structured state" shape is the right backbone.
- `story_module.py` — the outline-then-draft separation (plot decided before prose) aligns with our Bible/Outline/Scene decision split.
- `dspy_runtime.py` — already has Ollama wiring, `num_ctx` control, an Ollama runtime probe, and secret redaction. Adapt into `runtime.py`.

**Superseded:**
- Variant A's rolling text-summary continuity (drifts, loops late beats).
- Variant C's no-inter-chapter-state drafting (worst voice/name drift).
- Variant B's rendering of state into the artifact (UX bug).
- Markdown-scraping QA parsers — replaced in our design by querying the structured Bible.

## Strategic conclusions for the redesign

1. **Structured memory beats prose memory.** The bake-off proves it. The structured Bible is the right central bet.
2. **Auditing without acting is not enough.** Every check flagged problems nothing fixed. Our constrain → verify → **revise** loop is the missing third step.
3. **Robustness is architectural.** A single bad response must not kill a long run. Checkpoint/resume per stage is non-negotiable at novel length.
4. **Decide before drafting.** Names, uniqueness, plot beats must be settled in the Bible/Outline so the Scene stage only renders. Drift comes from deciding during drafting.
5. **Never scrape what you can query.** The fragile markdown parsers are an argument for structured artifacts with a typed query interface (Extract).
