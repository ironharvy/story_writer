# Evaluation Rubric

You cannot improve what you cannot measure (Beck). This rubric defines, operationally, what "cohesive" and "interesting" mean for a generated Story. It is the feedback loop: every change to the pipeline is judged against this. Without it, iteration is guesswork.

Two halves:
- **Cohesion** — mostly mechanical; the story doesn't contradict itself. Drawn from the prior implementation's nine known failure modes (`lessons-from-prior.md`).
- **Interestingness** — human-judged; the story is worth reading. Drawn from `craft-vocabulary.md`.

Each criterion has: a **name**, a **definition**, a **severity** (`fail` blocks; `warn` is a quality concern; `info` is observational), how it's **detected** (automated check vs. human/LLM judgment), and an **example failure** (most are real, from the prior bake-off).

A story's verdict is the worst severity across cohesion plus the human interestingness read. A `fail` on any cohesion criterion means the pipeline is not yet working, regardless of prose quality.

---

## Part A — Cohesion (mostly mechanical)

### A1. Name consistency — `fail`
Canonical character names are spelled consistently; no drift or misspelling. **Detected:** automated (`name_drift` + linter). **Example:** "Cindar" used for the canonical "Cinder."

### A2. Name uniqueness — `fail`
No two distinct characters share a name (or a colliding first name). **Detected:** automated (Bible-stage check). **Example:** `deepseek-v4-pro` produced two characters named Elara; the prose lampshaded it ("Her name is Elara," said the other Elara).

### A3. Protagonist naming — `fail`
The protagonist is named from their first appearance; no "the protagonist"/"the child" placeholders in prose. **Detected:** automated (placeholder scan) + linter. **Example:** variant C's chapter 1 used "the child" ×16.

### A4. POV / narration consistency — `fail`
The dominant narrative person and tense are stable across the manuscript; no chapter shifts within itself. **Detected:** automated (`pov_check` LLM classifier). **Example:** variant C narrated ch1–4 in third person, ch5 in full first person, ch6 in "the protagonist."

### A5. Character presence — `warn`
Every character defined in the Bible appears in the manuscript (or is deliberately cut). **Detected:** automated (`character_presence`, parsing the structured Bible — not scraped markdown). **Example:** qwen3:8b's world-state run dropped Eira and Kael entirely.

### A6. Scene/chapter length — `fail`
No empty or stub scenes. **Detected:** automated (min word count). **Example:** `runs/demo-hollow` ch3 was empty and silently passed every prior check.

### A7. Phrase-tic repetition — `warn`
No model-tic phrases reused verbatim across scenes. **Detected:** automated (`cross_chapter_phrase_reuse`, n=5). **Example:** "silver suppression runes etched into…" recurring in every chapter.

### A8. No adjacent-beat duplication — `warn`
Consecutive scenes/chapters don't re-run the same events. **Detected:** automated (high adjacent overlap) + human. **Example:** `deepseek-v4-pro` ch5 stormed the chamber and seized the relic, ch6 did it again, ch7 replayed ch6's cliffhanger.

### A9. World-rule consistency — `fail`
Nothing in the prose contradicts an established Bible world rule ("no moon landing in a world that just invented the wheel"). **Detected:** human / LLM-judged against the Bible (the future Consistency Checker). **Example:** a magic system stated as blood-cost in ch2 working for free in ch9.

### A10. Plot-thread payoff — `warn` (→ `fail` if a central thread)
Setups are paid off; threads don't dangle or get contradicted. **Detected:** human / LLM-judged against the Outline's setup/payoff tracking. **Example:** the "unburning candle" set up vividly in ch2, referenced in ch4, never explained.

### A11. Timeline / continuity — `fail`
Events, ages, locations, and possessions stay consistent over time. **Detected:** human / LLM-judged. **Example:** a character holding an object in ch6 that they handed away in ch4.

### A12. No Bible/scaffolding leakage — `fail`
Pipeline internals (Bible dumps, state blocks, planning notes) never appear in the manuscript a Reader sees. **Detected:** automated (structural). **Example:** variant B rendered seven "#### World State after Chapter N" blocks into the story.

---

## Part B — Interestingness (human-judged)

Scored 1–5 (1 absent, 3 competent, 5 excellent). These are not yet automatable; they're the human read that tells us whether the engine produces something worth reading.

### B1. Clear controlling idea
The story is *about* something; a single sentence of meaning holds it together. *(craft: controlling idea.)*

### B2. Escalating stakes & causal momentum
Scenes connect by consequence ("but/therefore", not "and then"); stakes rise. *(craft: the gap, try/fail cycles, stakes.)*

### B3. Protagonist agency
The protagonist drives events by choice rather than being passively swept along. *(craft: agency.)*

### B4. Scene-level dramatic value
Each scene turns — shifts a value charge — rather than sitting inert. *(craft: scene value/turn.)*

### B5. Earned, distinctive ending
Setups land; the climax and resolution feel inevitable yet surprising, not whiffed or abrupt. *(craft: payoff.)* **Counter-example:** qwen3:8b closed ch5, ch6, *and* ch7 on the protagonist's hand "hovering over the choice" — no decision, no climax.

### B6. Voice & prose quality
The prose has a discernible voice; characters sound distinct; figurative language serves rather than decorates. *(craft: narrative distance, free indirect discourse.)*

### B7. "Keep reading" gut check
Would a reader turn to the next chapter? The honest holistic verdict.

---

## Scoring format

Per story, produce:

```
COHESION (mechanical)        severity   detail
  A1 name consistency        pass
  A3 protagonist naming      FAIL       "the child" ×16 in ch1
  A6 scene length            pass
  ... (all automated criteria)
COHESION (judged)            severity
  A9 world-rule consistency  pass
  A10 plot-thread payoff     warn       unburning candle never explained
  ...
INTERESTINGNESS (1–5)
  B1 controlling idea        4
  B5 earned ending           2          climax whiffs; no decision
  ...
VERDICT: FAIL (A3) — protagonist unnamed in cold open
```

## What's automated now vs. later

- **Automated in the Step-4 tracer's `evaluate.py`** (ported from the prior code): A1, A3 (placeholder scan), A4 (POV), A5, A6, A7. Plus A12 (structural) and A2, A8 as cheap heuristics.
- **Human/LLM-judged for now** (candidates for the future Consistency Checker): A9, A10, A11, and all of Part B.

The mechanical criteria become regression tests. The judged criteria stay rubric-only until we have a Consistency Checker good enough to automate them — at which point each graduates from Part-B-judgment to an automated check.

## How this rubric is used

1. Run the pipeline on a seed Idea.
2. Run `evaluate.py` → mechanical findings (Part A automated).
3. Read the manuscript → judged findings (A9–A11) + interestingness (Part B).
4. Record the verdict. A change to the pipeline is an improvement only if it moves these numbers in the right direction without regressing others.

This rubric is itself a living artifact — as we learn what actually predicts a good Story, criteria get added, sharpened, or retired.
