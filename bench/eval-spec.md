# Story Manuscript — Evaluation Spec (objective function)

A self-contained spec for judging whether a generated manuscript is *cohesive,
complete, and worth reading*. It is the **objective function** for the build:
a draft "ships" only when it clears the hard gates and meets the rubric
threshold. Re-run after every change so quality is measured, not guessed.

These checks were distilled from real failure modes observed across local and
hosted models (qwen3, gemma, deepseek-v4): truncated chapters, unnamed
protagonists, point-of-view that lurches mid-book, duplicate character names,
and signature phrases repeated in every chapter. Each one exists because a
model actually shipped that defect.

This file is the source of truth that `bench/criteria.py` (deterministic
Tier-1 checks) and the (still-stubbed) LLM judge for Tier-2/3 implement.

---

## Expected manuscript shape

The checks assume the program emits the finished manuscript as Markdown with
at least these sections (names can differ, but the harness needs to locate
the prose and the cast):

```
# <Title>
## Premise            # one paragraph
## World / Characters  # canonical cast list: "1. <Name>, <description>"
## Manuscript          # the actual story
### Chapter 1: <title>
<prose>
### Chapter 2: <title>
...
```

Today's `update_artifact` writes `## Final Story` then `### Chapter N: <title>`
— the harness must accept that section name too. The "canonical name" of a
cast entry is the text before the first `, : ; — – -` delimiter, with
surrounding `*`/`_` emphasis stripped (`**Cinder**, a runaway` → `Cinder`).

---

## Tier 1 — Deterministic checks (no LLM, run on every draft)

Fast, free, reproducible. These are gates: a `FAIL` here blocks shipping.

### T1.1 Chapter length — `FAIL`
Every chapter's prose must be at least a hard floor and should fall in a
target band.
- **Hard floor (gate): 300 words.** Below this the chapter is empty/truncated.
- **Target band (warn outside): 800–2500 words.** Thin ~250-word chapters
  are the classic fast-model failure; bloated >3000-word chapters usually
  mean the outline collapsed into one chapter.

### T1.2 Character presence — `FAIL`
Every canonical character must appear in the manuscript prose at least once.
Match the full name **or** the first token (so `Kaelen` alone counts for
`Kaelen Vey`). A character defined in the cast list but never present in the
story is a `FAIL`. Skip cast bullets whose entire content is bold/italic with
no trailing description — some models put motivations in the cast slot.

### T1.3 Name drift — `WARN`
Detect chapter prose that uses a variant spelling of a canonical name.
Collect proper nouns from the prose (1–3 capitalized tokens). For each,
if its first token matches a canonical name's first token but the full form
differs **and** isn't a clean prefix of the canonical (`Kaelen` for
`Kaelen Vey` is fine), flag it (`Cinder` vs `Cindar`). Skip generic leads
(`The`, `A`, `An`).

### T1.4 Cross-chapter phrase reuse — `WARN` (with a budget)
For each chapter, take the set of 5-grams (lowercased word tuples); count
how many chapters share each 5-gram; report any 5-gram present in ≥2
chapters. Exclude 5-grams that also occur in the world/premise sections.
- **Budget:** ≤ 5 distinct cross-chapter 5-grams for a 7-chapter story.

### T1.5 Content-word over-repetition — `WARN`
Count word frequency in the manuscript prose (lowercase, strip possessives,
drop stopwords, ignore words shorter than 4 chars). After dropping the
~20 most-frequent words, flag any non-name content word whose count
exceeds `max(8, word_count / 400)`.

---

## Tier 2 — LLM-judge checks (need a model; run before shipping)

Heuristics can't read. Use the local Ollama model as the judge first;
only escalate to the hosted fallback if the local judge gives
unstable/garbled verdicts. Always feed the judge the *narration only*
instruction so quoted dialogue doesn't confound it.

### T2.1 POV / narrator consistency — `FAIL` on within-chapter shift
Ask the judge to classify each chapter's **dominant narration POV** —
`first_person`, `third_person`, `mixed`, or `other`. Ignore POV inside
dialogue, letters, journals, reported speech.
- **`FAIL`** any chapter classified `mixed`.
- **`FAIL`** if decisive chapters disagree with the book's dominant POV.

### T2.2 Protagonist is named and named consistently — `FAIL`
The protagonist must be referred to by a proper name in the prose. **`FAIL`**
if chapters lean on `the protagonist` / `the child` / `the boy` / `the woman`
as the primary referent (the classic cold-open defect). **`FAIL`** if the
protagonist is called by two different names across chapters.

### T2.3 Continuity / no contradictions — `FAIL` on hard contradiction
Ask the judge to list factual contradictions across the manuscript: a
character dead in ch3 acting in ch5, eye/hair color or relationships that
flip, a timeline that doesn't add up, two distinct characters sharing one
name, locations that teleport. Hard contradictions are `FAIL`; minor wobble
is `WARN`.

### T2.4 Premise fidelity — `WARN→FAIL` if it drifts off-idea
Give the judge the user's original idea + the manuscript: does the story
actually dramatize *this* premise, or did it drift into a generic tale?
A manuscript that ignores the user's core idea is a `FAIL` regardless of
prose quality.

---

## Tier 3 — Qualitative rubric (LLM judge, 1–5 each)

| Axis | 1 | 5 |
|---|---|---|
| **Arc & structure** | episodic, no shape | clear setup → rising action → climax → resolution |
| **Ending** | stops / fizzles / deus ex machina | earned, lands the premise's promise |
| **Character agency** | events happen *to* passive figures | choices drive the plot, characters want things |
| **Scene vs summary** | tells/summarizes throughout | dramatizes key beats in scene |
| **Prose quality** | flat, repetitive, garbled | varied, controlled, few tics |
| **Cohesion** | threads dropped, setups unpaid | setups pay off, threads resolve |

**Threshold to ship: average ≥ 4.0 with no axis below 3.**

---

## Ship gate

A draft is shippable iff **all** hold:

1. **Zero Tier-1 `FAIL`** (T1.1, T1.2).
2. **Zero Tier-2 `FAIL`** (T2.1, T2.2, T2.3, T2.4).
3. **Tier-3 rubric average ≥ 4.0, no axis < 3.**
4. **Soft budgets respected:** T1.3 ≤ 2, T1.4 ≤ 5, T1.5 nothing flagged.

Emit one machine-readable scorecard per draft (JSON: each check → severity
+ message, plus rubric scores and a final `ship: true|false`).

---

## Regression list — defects this spec exists to prevent

- Empty/truncated chapter → **T1.1**.
- Cold-open chapter 1 says "the protagonist"/"the child" → **T2.2**.
- Signature phrase in every chapter → **T1.4**.
- POV lurch in the back third → **T2.1**.
- Two characters named the same → **T2.3**.
- Thin ~250-word chapters on the fast model → **T1.1** band.

---

## Provenance

The text of this spec was extracted from PR #112 / #113 (the duplicate eval-spec
PRs). Those PRs also shipped a 41-file `storyforge/` parallel implementation
that we deliberately did not land, since Plan A consolidates competing variants
behind a single `generators/` registry instead of forking the tree. Both PRs
were closed as superseded by this file and the upcoming bench harness.
