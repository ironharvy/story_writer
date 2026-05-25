# 04 · Detailed Design & Algorithms

The "how" of each stage. Components are in [03-architecture.md](03-architecture.md).

## Human-in-the-loop review loop (foundation steps)

Most `run_*` functions in `story.py` share one pattern: generate with a DSPy
`ChainOfThought`, show the result via `ui.review_answer`, and loop on free-text
feedback until the user accepts. Feedback and the previous result are fed back
into the signature as inputs, so regeneration is correction-aware.

```
while True:
    result = predict(..., previous_result=prev, feedback=feedback)
    feedback, ok = ui.review_answer(label, result)
    if ok: return result
```

## Foundation

- **Clarify idea** (`run_clarify_idea`) — generate clarifying questions +
  proposed answers; user reviews each; fold Q&A into an `updated_idea`; generate
  a title if none.
- **Core premise** (`run_generate_core_premise`) — single field, review loop.
- **Spine** (`run_generate_spine`) — **Pixar 7-step formula**, one output field
  per beat: `once_upon_a_time`, `every_day`, `until_one_day`, `because_of_that`,
  `and_because_of_that`, `until_finally`, `ever_since_that_day`. Stored as the 7
  beats joined by newlines (this newline order matters — the act slicer indexes
  into it).
- **World bible** (`build_world_bible`) — rules → locations (each enhanced) →
  timeline → characters (each enhanced). Result is a `WorldBible`.
- **Sanity check** (`sanity_check`) — LM returns a boolean: are idea / spine /
  bible consistent? Pipeline A aborts the run if false.
- **Chapter plan** (`run_generate_chapters_plan`) — N × `PlanEntry`
  (`chapter_title`, `chapter_beats`).

## Act assignment & spine slicing (`act_hint_for_chapter`)

Maps 1-indexed chapter `i` of `n` onto a 3-act structure so a chapter only ever
sees spine beats up to and including its own act (prevents later-act
foreshadowing leaking into early chapters).

1. **Split chapters** ≈ 25 / 50 / 25: `a1 = a3 = ceil(n/4)`, `a2 = n − a1 − a3`,
   with at least one chapter per act for `n ≥ 3`; special-cased for `n = 2`
   (1/0/1) and `n = 1` (all act 1).
2. **Assign act**: `i ≤ a1` → act 1; `i ≤ a1+a2` → act 2; else act 3.
3. **Slice spine** by beat index — act 1 → beats[:3], act 2 → beats[:5],
   act 3 → beats[:7] — using the Pixar ordering above. Returns
   `{act, label, spine_through_act}`.

## Variant A — baseline (`pipeline.py` + `run_enhance_chapter`)

For each chapter: compute the act hint, optionally (≈33% chance) generate a
"random detail" to enrich the scene, then draft prose from the act-sliced spine
+ world bible + the rolling **story-so-far** summary. After each chapter,
`run_generate_story_so_far` summarizes the plan-so-far + prior summary + new
chapter into an updated summary that feeds the next draft. The draft signature
instructs: treat the bible as reference (don't reuse its phrasing verbatim),
include a concrete friction beat, end on action not aphorism, vary rhythm,
respect the current act and don't foreshadow later ones.

## Variant B — world-state (`world_state.py`)

- `WorldState` = clock + characters + locations + plot threads + key objects +
  recent events.
- `run_init_world_state` — translate the static bible into the opening snapshot
  (recent_events empty).
- `run_advance_world_state` — fold a freshly written chapter into a new state
  (positions, knowledge, threads, clock, objects; append + condense recent
  events).
- `run_draft_chapter_with_state` — draft from bible + current state + act slice;
  must honour the state and not contradict it. Because the state carries the
  protagonist's name into chapter 1, B avoids variant A's "unnamed cold open".

## Variant C — module (`story_module.py`)

Builds its own outline and drafts each chapter independently from its outline
beats + world bible + act-sliced spine. No inter-chapter state → most prose, but
voice/POV can drift chapter-to-chapter.

## QA suite (`qa.py`) — pure text checks

Parses the artifact by its heading layout. Each check returns `Finding(check,
severity, message)` with severity `info | warn | fail`.

| Check | Algorithm | Severity |
|---|---|---|
| `cross_chapter_phrase_reuse` | 5-gram sets per chapter; report grams appearing in ≥2 chapters (excluding bible grams), top 10 | warn (+info) |
| `name_drift` | Proper-noun regex over prose; flag forms sharing a canonical character's first token but differing in full form; skips strict prefixes and generic `The/A/An` | warn (+info) |
| `character_presence` | Each canonical character (from the `### Characters` numbered list, skipping bold-only motivation bullets) must appear by full name or first token | fail (+info) |
| `chapter_length` | Chapters under `CHAPTER_MIN_WORDS` (80) fail — catches empty/truncated chapters | fail (+info) |

## POV consistency (`pov_check.py`) — LLM-backed

Sends all chapters to the LM, which classifies each chapter's *narration only*
(ignoring dialogue/letters) as `first_person | third_person | mixed | other`.
The dominant decisive POV is the mode; chapters that are `mixed`, or decisive
but disagree with the dominant, are flagged `fail`. Catches drift `name_drift`
cannot (e.g. ch5 first-person in an otherwise third-person book).

## Prose linter (`story_linter.py`) — LLM-backed find/replace

Given the `### Characters` section + chapter prose, the LM returns verbatim
find/replace pairs for placeholder references ("the protagonist") and
misspelled/inconsistent canonical names. `validate` drops pairs whose `find`
isn't in the prose, are trivial, or duplicate. `apply` rewrites **chapter prose
only** — the region under `## Final Story` — using word-boundary-aware patterns,
leaving the bible/plan/spine untouched. Counts are written to a
`.replacements.json` sidecar.
