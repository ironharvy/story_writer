# StoryForge — Design Note

How a story idea becomes a cohesive, complete manuscript here, what craft research
informed it, the competing algorithms I weighed, and why I chose the one I did.

This is a **from-scratch** implementation (the prior code in the repo root is
ignored, per the goal brief). Everything new lives under `storyforge/`.

---

## 1. What I learned from researching the craft

A handful of durable ideas from the writing-craft literature shaped the pipeline.
Sources are linked at the bottom.

- **Premise / logline first.** A tight one-sentence promise — *protagonist + goal +
  conflict + stakes* — is the anchor that everything else must serve. Randy
  Ingermanson's **Snowflake Method** starts here and grows the book *fractally*:
  one sentence → one paragraph → character sheets → scene list. The crucial
  property is that each level is **derived from the level above**, which is exactly
  the mechanism that keeps a long text globally coherent.
- **Structure scaffolds.** The **three-act structure** is the backbone (setup →
  rising action through two disasters and a midpoint → climax → resolution).
  **Save the Cat!** (Blake Snyder) refines it into 15 concrete beats; **Dan
  Harmon's 8-step Story Circle** (You → Need → Go → Search → Find → Take → Return
  → Change) is a compact, character-arc-centric skeleton that maps cleanly onto a
  short manuscript and forces a *transformation* (a real ending, not a fizzle).
- **Scene vs. sequel.** The unit of fiction is a **scene** (proactive:
  goal → conflict → outcome) followed by a **sequel** (reactive: reaction →
  dilemma → decision). Alternating them creates rhythm and, more importantly,
  *agency*: the protagonist's decisions drive the next scene rather than events
  happening *to* them. This directly targets two rubric axes (character agency,
  scene-vs-summary).
- **Character arc.** Want vs. need, the lie the character believes, the ghost in
  their past — a **positive change arc** is what makes the ending feel *earned*.
- **Outlining vs. discovery.** "Plotters" outline; "pantsers" discover. For an LLM
  on a small local model, **outline-first wins decisively** for coherence —
  discovery writing drifts and contradicts itself over a long context. But
  over-outlining yields "told not shown" prose, so we outline *structure* and let
  the drafter dramatize within each beat.
- **Revision in passes.** Professional revision is layered: **developmental**
  (structure/arc) → **line** (prose) → **copy** (consistency). Our self-eval +
  targeted-revise loop is a compressed version of this, driven by the eval
  scorecard.

---

## 2. Competing algorithm designs

I deliberately sketched four, rather than committing to the first obvious pipeline.

### A — Snowflake / fractal outline-first (waterfall + continuity bible)
idea → logline → one-paragraph (3-act) → **world bible** (cast + locations + rules)
→ beat sheet (N beats → chapters) → per-chapter scene plan → draft chapters
sequentially with rolling context (prior-chapter summary + bible + avoid-list) →
self-eval → targeted revise.

### B — Beat-driven scene/sequel with an agency loop
Like A, but the atomic unit is the **scene/sequel pair**, each tagged with POV
character, goal, conflict, outcome, and the character's want/need; scenes are then
packed into chapters.

### C — Iterative "spiral" / progressive refinement (discovery-flavored)
Generate a fast full zero-draft, then run whole-manuscript revision passes
(developmental → line), each conditioned on the eval scorecard. Fewer upfront
constraints, more emergent.

### D — Multi-agent roles (architect + drafter + continuity-editor + line-editor)
Separate role-prompts negotiate: planner proposes, critic critiques, drafter
writes, continuity-checker validates against the bible, editor polishes.

### Tradeoff comparison

| Design | Coherence | Surprise | Cost/speed (local) | Robustness |
|---|:---:|:---:|:---:|:---:|
| **A** Snowflake outline-first | ★★★★ | ★★ | ★★★★ | ★★★★ |
| **B** Scene/sequel | ★★★★ | ★★★ | ★★★ | ★★★ |
| **C** Spiral refinement | ★★ | ★★★★ | ★★ | ★★ |
| **D** Multi-agent | ★★★★ | ★★★ | ★ | ★★ |

---

## 3. The choice: A as the spine, hybridized with the best of B and D

**Core = Design A (Snowflake / fractal outline-first).** On a small local model it
is the best blend of coherence, robustness, and cost, and its top-down derivation
is the single strongest lever against drift. It is also, not by coincidence, the
shape the evaluation spec rewards.

I fold in **bounded** enhancements rather than committing to a heavier design:

- **From B:** each chapter plan is expressed as **scene/sequel beats** with an
  explicit POV character, goal, conflict, outcome, and the character's want/need.
  This buys agency and scene-vs-summary *without* a scene-level call explosion —
  we still draft one chapter per call.
- **From D:** a single bounded **critic → revise** pass per chapter (deterministic
  Tier-1 checks + one focused LLM critique → at most one targeted rewrite), not a
  full agent swarm. This is our compressed developmental+line pass and it is
  capped to control cost.

**Runners-up are kept explicitly in reserve** (and the code is structured so each
is a small change, not a rewrite):
- If **agency / scene-vs-summary** rubric stays low → drop to **B** (scene-level
  drafting) for the weak chapters.
- If **prose quality / surprise** stays low → add a **C**-style whole-manuscript
  line-polish pass as a final stage.
- If **continuity** fails persistently → promote the critic to a dedicated **D**
  continuity agent with the full bible in context.

The eval scorecard is the fitness signal that decides whether any reserve is
activated — measured, not guessed.

---

## 4. Engineering the pipeline *for the eval gates*

Every failure mode in `STORY_EVAL_SPEC.md` is designed against up front, so we
clear gates by construction rather than by luck:

| Gate | Failure mode | How the pipeline prevents it |
|---|---|---|
| T1.1 | thin/truncated chapters | per-chapter word **budget** in the plan; `max_tokens ≥ 8192`; auto-retry "expand" if a draft is under the floor |
| T1.2 | cast member never appears | planner assigns every canonical character a role in ≥1 chapter |
| T1.3 | name drift (`Cindar`/`Cinder`) | canonical name list injected into every draft prompt; post-check → revise |
| T1.4 | signature phrase in every chapter | rolling **avoid-list** of prior chapters' distinctive n-grams fed forward |
| T1.5 | content-word over-repetition | varied-diction instruction; post-check → revise |
| T2.1 | POV lurch | single POV **locked in the bible**, restated every chapter, "narration only" |
| T2.2 | "the protagonist"/"the child" | name fixed in premise+bible *before* ch.1; ch.1 prompt bans placeholders |
| T2.3 | contradictions / duplicate names | world bible + rolling chapter summaries as canon; continuity critic |
| T2.4 | drifts off the idea | original idea + logline carried into every chapter prompt; final fidelity check |

**Robustness requirements (build, not quality):** every LM call is guarded (retry
on `None`/empty/parse-failure), and **every artifact is written to disk the moment
it is produced** (premise, bible, each chapter), so a crash at chapter 6 never
loses chapters 1–5 and a run is **resumable** from its run directory.

---

## 5. Model strategy

Per the brief and the chosen session config: **local Ollama first.** Build and
smoke-test the workflow on the fast `qwen3:latest`, validate real prose quality on
`qwen3.6:27b`, and escalate to hosted `deepseek` / `groq` (via litellm) only if the
local models repeatedly cannot clear the bar — kept minimal because it costs money.
The eval harness uses an **independent judge** (a different model from the drafter)
to avoid self-preference bias. Every scorecard records which model produced the
result.

---

## 6. Empirical findings (what the evidence changed)

Measured on this WSL2 box against the live backends — the scorecard, not
assumptions, drove these:

- **qwen3 is a *thinking* model.** With reasoning on (the default), Ollama puts the
  chain-of-thought in a separate `thinking` stream and a short `num_predict`
  budget is consumed before any `content` is emitted — every call came back
  **empty**. Fix: pass `think=false` to Ollama (litellm forwards it) and strip any
  stray `<think>…</think>`. This was the first real bug and would silently break
  any naive integration.
- **`num_ctx` must be pinned.** Ollama defaults context to 4096, which silently
  truncates the rolling synopsis + an 8k-token chapter. Set to 24576.
- **`qwen3:latest` (8B):** ~10 s/chapter (think off); coherent structure, named
  protagonist, real dialogue — but **parrots its own signature phrasings** across
  chapters (T1.4 = 25–30 raw). Great for "does it run"; needs the de-repetition
  pass to approach the ship budget.
- **The de-repetition pass works:** on the 8B it cut cross-chapter 5-gram reuse
  25 → 9 in two rounds by rewriting only the later-occurring duplicates.
- **`qwen3.6:27b` (27B):** ~4 min one-time model load, then ~15–40 s/call; used to
  validate real prose quality and as the quality drafter.
- **Independent judge:** draft and judge must differ (no self-grading). With a 27B
  draft the local independent judge is `qwen3:latest`; the 8B judge reliably caught
  every blatant Tier-2 defect in the fixtures.
- **Judge strength matters for the *ship* decision.** On the real 7-chapter draft
  the 8B judge was both lenient and wrong: it scored the rubric **4.5**, mislabeled
  a clean third-person book as first person (fooled by first-person *dialogue*), and
  raised a **false** "hard contradiction" (it read a character *losing her memories*
  — the story's whole theme — as her death). The hosted **DeepSeek** judge got POV
  right, found no contradiction, and scored a stricter, more credible **3.5**. This
  is exactly why the spec wants a stronger independent second opinion at the ship
  gate — and why the final scorecard here is DeepSeek's, not the 8B's.
- **Hosted backends (litellm):** **Groq** (`llama-3.3-70b`, `gpt-oss-120b`) is
  sub-second but the free tier caps at **8000 TPM** — fine for small checks, too
  small to judge a whole 7-chapter manuscript in one call. **DeepSeek**
  (`deepseek-chat`, `deepseek-v4-pro`, `deepseek-reasoner`) handles full context and
  is the reserved hosted second-opinion / escalation judge.
- **Observability:** Langfuse v4 is OTEL-based, so litellm's `langfuse_otel`
  callback is required (the legacy `langfuse` callback crashes on v4). With that,
  every stage traces to Langfuse under its step name (`draft-ch3`, `critique-ch2`,
  `t3-rubric`, …).

> The scored sample manuscript and its committed scorecard live in
> [`samples/`](samples/); the scorecard records exactly which models produced and
> judged it.

### Sources
- Snowflake Method — Randy Ingermanson: https://www.advancedfictionwriting.com/articles/snowflake-method/ ·
  Reedsy summary: https://reedsy.com/blog/snowflake-method/
- Save the Cat! beat sheet (Blake Snyder), overview via MasterClass:
  https://www.masterclass.com/articles/how-to-use-the-snowflake-method-to-outline-your-novel
- Dan Harmon's Story Circle: https://www.studiobinder.com/blog/dan-harmon-story-circle/
- Scene & sequel / story structure — Helping Writers Become Authors:
  https://www.helpingwritersbecomeauthors.com/scene-structure/ ·
  Writers Helping Writers: https://writershelpingwriters.net/2015/01/writing-patterns-fiction-scene-sequel/
- Outlining vs. discovery writing & writing glossary:
  https://www.helpingwritersbecomeauthors.com/writing-glossary/
