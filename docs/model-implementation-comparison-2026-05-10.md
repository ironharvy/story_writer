# Story-pipeline comparison — models × drafting variants

Run date: 2026-05-10. Idea: *"A nameless child, raised and trained by the Church to hunt
demons, discovers the Church secretly breeds the very demons it sells protection from — and
must decide whether to destroy the one institution standing between the world and the dark."*
Title: **"The Tithe of Ash"**. 7 chapters. Settings: `--num-ctx 24576 --max-tokens 8192`,
Ollama, DSPy disk cache **on** (wiped per model; within a model, variant A warms the
foundation and B/C reuse it). Raw outputs: `.tmp/cmp/<model>/{A_baseline,B_worldstate,C_module}.{md,log,console}`.
First pass used `--max-tokens 4096` — see "Cross-cutting issues" #1; archived in
`.tmp/cmp/*_mt4096*`.

> **Addendum 2026-05-12 — `deepseek-v4-pro`.** Added a fourth model: DeepSeek V4 (released
> 2026-04-24), the **Pro** size, Thinking mode (on by default). **Variant A only**, same
> idea/title/chapters, but run against the **hosted API** (`deepseek/deepseek-v4-pro` via
> litellm, key in `.env`) — not Ollama — at `--max-tokens 32768`. The big token budget is
> deliberate: DeepSeek bills reasoning tokens against the *output* cap, so a smaller
> `--max-tokens` risks the truncation-→-`None` crash (Cross-cutting issue #1) on every step.
> Raw output: `.tmp/cmp/deepseek_v4_pro/A_baseline.{md,log,console}`; driver:
> `.tmp/cmp/run_deepseek.sh`. Rows tagged *deepseek-v4-pro* below; dedicated read-through and
> recommendation note added. 45 LLM calls, ≈207k input / ≈73k output tokens (output incl.
> reasoning); litellm has no price card for the new model yet so it logged `$0` — at
> DeepSeek's published rates this run is well under $1. Wall-clock 43 min (vs 86 min for
> `qwen3.6:27b` A — the hosted API has no local-VRAM bottleneck).

## The variants

| | Entry point | Inter-chapter continuity |
|---|---|---|
| **A · baseline** | `mymain.py` / `pipeline.py` | per-chapter `run_enhance_chapter` + rolling text summary |
| **B · world-state** | `mymain_ws.py` / `pipeline_ws.py` + `world_state.py` | structured `WorldState` (story clock / characters / locations / threads / objects / recent events), updated each chapter, **also rendered into the artifact** |
| **C · dspy.Module** | `mymain_module.py` / `pipeline_module.py` + `story_module.py` | none — each chapter drafted independently from its own outline beats + world bible + act-sliced spine. Builds its own outline (different chapter titles). |

## The models

| Tag | Size | Notes |
|---|---|---|
| `qwen3:latest` | ~8B (5.2 GB) | small local baseline; the model used in earlier `runs/` |
| `qwen3.6:27b` | ~27B (17 GB) | the newly-pulled candidate |
| `gemma4:26b` | (17 GB) | the model that "failed miserably" before |
| `deepseek-v4-pro` | ~1.6T MoE / 49B active — **hosted API, not local** | added 2026-05-12; DeepSeek V4 (2026-04-24), Thinking mode on by default; `deepseek/deepseek-v4-pro` via litellm, `DEEPSEEK_API_KEY` in `.env` |

## Completion / robustness

All 9 runs completed and produced 7 non-empty chapters **at `--max-tokens 8192`**. At the
old default `--max-tokens 4096`: `qwen3:latest` A & B completed but **C crashed
deterministically** (`build_outline` JSON truncated → `ValidationError` on `list[PlanEntry]`),
and **all three `qwen3.6:27b` variants crashed** (verbose model hits the 4096 ceiling →
truncated response → DSPy returns `None` → unguarded `len(None)` / `None.story_clock`). So
robustness is a *pipeline* problem, not a model one — see issue #1. *(2026-05-12: the
`deepseek-v4-pro` variant-A run completed cleanly — 7 non-empty chapters, no crash — at
`--max-tokens 32768`; that headroom is needed because Thinking-mode reasoning tokens count
against the output cap, so a smaller budget would re-trigger issue #1 on every step.)*

## Metrics (prose only; variant B's rendered WorldState blocks excluded)

| Model | Variant | Wall-clock | Prose words | Per-chapter words | Repeated 5-grams (≥2 ch) | QA fails (real) |
|---|---|---|---|---|---|---|
| qwen3:latest | A | 297 s | 1,815 | 206–307 | 27 | char_presence (3 cast absent) |
| qwen3:latest | B | 196 s | 1,932 | 186–386 | 27 | (name_drift fail is a parse artifact) |
| qwen3:latest | C | 91 s | 2,042 | 224–430 | 21 | char_presence (2 cast absent) |
| **qwen3.6:27b** | **A** | **86 min** | **6,362** | 596–1,634 | 26 | — (char_presence fail is a parser artifact) |
| **qwen3.6:27b** | **B** | 39 min | 5,821 | 558–1,196 | **12** | — (Soren the Ash↔Soren Vael, minor) |
| **qwen3.6:27b** | **C** | 21 min | **7,154** | 816–1,536 | 21 | — |
| gemma4:26b | A | 552 s | 3,642 | 393–690 | 22 | name_drift (Vane↔High Clergy, paraphrase); protagonist unnamed |
| gemma4:26b | B | 402 s | 3,711 | 440–583 | 16 | char_presence ('Elara' unused) |
| gemma4:26b | C | 196 s | 3,618 | 256–619 | 24 | name_drift (Vane↔High Inquisitor) |
| **deepseek-v4-pro** *(hosted, Thinking)* | **A** | **43 min** | **10,761** | 898–2,161 | 28 | — QA clean (char_presence ✓, name_drift ✓); *but* QA misses two real ones — ch3 lurches to "he/him", and the cast has **two characters named Elara** (the text even lampshades it) |

(B's *raw* file word counts are ~2× higher because the per-chapter "World State after Chapter
N" blocks are rendered into the artifact. Wall-clock for B/C is short because they reuse the
cached foundation; the 27B's foundation alone is ~45–50 min.)

## Read-through (1–5: continuity / coherence / structure / prose / plot-thread tracking)

### qwen3.6:27b — clearly the best model
- **A (baseline)** — 4.5 / 4.5 / 4.5 / 4 / 4. Crisp declarative prose, concrete sensory
  detail; named protagonist **Cinder**; coherent cast (Veyra, Soren, Archivist Malora,
  Cardinal Valerius, Matron Oros, Kaelen — who sacrifices himself in ch6). The
  protection-racket premise is actually *dramatized* (the Ash-Furnace Atrium rendering
  hunters into ash bricks, demons kept as "stock", an "Asset Alpha — Terminal Yield: Maximum"
  ledger). Real resolution in ch7 (the "third path" — weaving human/demon resonance instead
  of consuming either; the furnaces fall silent), with a controlled cold-tea/mug image set up
  in ch7 and paid off in the final beat. **Flaw:** chapter 1 (the cold open) calls the
  character "the protagonist" ×8 — the name isn't established yet. Minor repetition crutches
  ("silver suppression runes etched into…", "the heat in their veins", italic *Ash-Burn*).
- **B (world-state)** — 4.5 / 4 / 4.5 / 4 / 4. Same world; **the most consistent narration**
  — "Cinder", third person, in *every* chapter including ch1 (the WorldState handoff feeds
  the name into ch1 drafting). Lowest prose repetition of any run (12). Slightly thinner
  prose (5.8k). Downside: the artifact is cluttered with seven "#### World State after Chapter
  N" scaffolding dumps a human reader has to skip; one minor name wobble (Soren the
  Ash ↔ Soren Vael).
- **C (dspy.Module)** — 4 / 3 / 4.5 / 4 / 3.5. *Most* prose (7.2k) and the same coherent arc
  (the one-shot outline carries the through-line even with no inter-chapter state). **But the
  narrative voice isn't stitched:** ch1 = "the child" ×16, ch2–4 = "Cinder", **ch5 lurches to
  full first person** ("I stepped through the fracture… my collarbone…", "Cinder" ×0), ch6
  reverts to third person but says "the protagonist" ×12, ch7 = "Cinder". Each chapter drafted
  independently → POV/name drift the `name_drift` QA can't detect.

### deepseek-v4-pro (hosted, Thinking) — best prose tested; the back third stutters *(added 2026-05-12)*
- **A (baseline)** — 3.5 / 4 / 3.5 / **4.5** / 4. The strongest sentence-level prose of any
  run here — figurative without being purple, dense with concrete sensory detail ("the road
  to Thornwood was a tongue of packed dirt licking through blackthorn and mud", a headman whose
  "back [is] a question mark", "coins that clinked soft as regret"), and it carries real
  motifs: the seamstress's button-eyed doll (glimpsed in a hut in ch1 → clutched on the altar
  in ch5 → carried by the freed demon in ch7 → tucked into the orphan's belt, "a relic of
  their own"), the candle in the skull-cup that "burned without consuming", the falchion
  *Silence* whose hum shifts from "hymn" to "howl" as the protagonist de-conditions. The
  premise is dramatized hard (the "extraordinary tithe" of children, the Bleeding Chamber, the
  heart-relic, the cells behind the Tithe Vault, Isidor's 34-year ledger of vanished names),
  and the ending is genuinely distinctive: Elara insists in ch3 that there is "no third path"
  — destroy the Church and unleash the feral surge, or be hunted — and ch7 *is* the third
  path: the orphan founds an order of "binders" who leash feral demons without blood sacrifice,
  at the cost of channelling the demon's terror back into themselves, ending on a knife-edge
  ("the moors waited, screaming, to see if the new order would hold"). Distinctive choice:
  **the protagonist is never named** — "the nameless orphan" / "the orphan" throughout — and
  here that reads as thematic intent (a husked weapon reclaiming a self), not the unnamed-cold-
  open *bug* that A has on `qwen3.6:27b`. **Real flaws**, none of which QA catches:
  - **POV drift in ch3** — chapters 1–2 and 4–7 narrate the orphan as "they/them"; **ch3
    abruptly switches to "he/him"** ("The steeple swallowed him whole…") and back. Same family
    as the `qwen3.6:27b` variant-C voice drift, in variant A this time — exactly Cross-cutting
    issue #3.
  - **Two characters named Elara** — "Sister Elara", the scarred Remnant exorcist, *and* the
    mute Thornwood seamstress (the world bible names her Elara too). Ch5 has the exorcist say
    of the seamstress "Her name is Elara," which only highlights the collision. A world-bible
    name-uniqueness check would catch this; `name_drift` can't (it's a collision, not drift).
  - **The back third duplicates beats.** Ch5 ("The Tithe of Bodies") already storms the
    Bleeding Chamber, kills the ritualists, severs the main tether, the orphan seizes the
    heart-relic, the demons go feral, they flee up the bone stair. Ch6 ("The Heart of Ash")
    then does *the same things again* (kill ritualists incl. Marcellus, seize the heart-relic
    "from its nest of desiccated flesh", tethers snap, ferals erupt, "Run", up the bone stair)
    plus the cathedral reveal — and ch7 *opens by replaying ch6's cliffhanger* (the relic
    sizzling on the altar, the first spider-limbed feral crawling out). The per-chapter draft +
    rolling-text-summary continuity (variant A) let the model pull ch6's climax into ch5 and
    then loop. Ch6 is also the runt at ~900 words and reads more like a breathless recap than a
    chapter; Marcellus, built up over five chapters, dies almost in passing.
  - Minor: the unburning candle is set up vividly in ch2, referenced in ch4, and never
    actually explained — an evocative loose thread. A few tics ("with the economy of long
    practice" ×2, "Silence's hum dropped to a…" ×2, "rain-soaked slate" / "rain-slate" eyes
    used for *two* characters).
- **Vs `qwen3.6:27b` A** (4.5 / 4.5 / 4.5 / 4 / 4): roughly a wash on overall quality, with
  opposite strengths. `qwen3.6:27b` is the more *disciplined* draft — even chapter lengths, no
  pronoun drift, no beat duplication, and it names its protagonist (Cinder); its prose is crisp
  but plainer. `deepseek-v4-pro` writes markedly better prose and a more surprising ending but
  has a sloppier spine in the back third. Also: ~2× faster (43 vs 86 min) and cheap — but it's
  a hosted API, so it loses on the project's local-first axis.

### gemma4:26b — *not* a miserable failure anymore; ~8B-tier
- **A** — 3.5 / 3.5 / 3.5 / 2.5 / 3. Coherent: "the Hunter" (never named — "Elara" sits unused
  in the bible), "The Sight" power, the Aegis as a parasitic shield that consumes the
  "Unworthy", Father Malachi (ex-mentor), High Cardinal Vane, Sister Valerica the relic-flail
  Inquisitor, the Great Conduit. Recurring bell motif (ch2 dead-thud → ch7 found in the ash).
  Ending lands but is **bleak and abrupt** — the Hunter shatters the Aegis, the world drowns
  in grey/Ash-Wastes, story ends mid-stalk; picks "destroy", no third path. **Real prose
  defects:** dropped/garbled words — "shredly the reinforced leather", "wreatened in a dim
  light", "curdended into oily", "curdified" (×2), "Lower Wands" (= Wards). qwen3.6:27b never
  did this. Heavy tics: "kaleidoscopic nightmare of overlapping {realities, truths}", "the
  Hunter gasped, teeth grinding against…".
- **B** — ~3.5 overall. Same story; the WorldState updates barely evolve between chapters
  ("industrial gothic processing area for the 'unworthy'" appears in 5 of 7 state blocks,
  "demon incursion at the iron border outpost" in 4) — the world-state mechanism adds little
  signal with this model.
- **C** — ~3.5; ch7 collapses to 262 prose words. Coherent, own outline; uses "the hunter"
  throughout so it doesn't show the name/POV drift the 27B's C did.

### qwen3:latest (8B) — coherent but thin and stiff
- **A** — 4 / 3.5 / 3 / 2.5 / 2.5. ~1.8k words total (≈260 wpc). Coherent (Kael, Veyra, the
  High Inquisitor, demons-as-trapped-souls, the relic-is-a-forge reveal, reformist Brother
  Elias), but **the ending whiffs** — chapters 5, 6 and 7 all close on Kael's hand "hovering
  over the choice"; no decision, no climax. Crutch-phrased ("their scar burned in time with…"
  nearly every chapter; "[place] seemed to hold its breath" ×3). The protection-racket
  economics are stated, never shown.
- **B** — similar quality; QA caught two world-bible characters (Eira, Kael) never appearing
  in the chapters — the world-state summarization dropped them.
- **C** — runs clean at 8192 (the 4096 crash was a truncation artifact); ~2k words; same tier.

## Recommendation

1. **Adopt `qwen3.6:27b` as the default local model.** It is a clear, large step up from both
   `qwen3:latest` and `gemma4:26b` — 3–4× the prose volume, dramatizes the premise instead of
   gesturing at it, lands real endings, and is the only model that reliably names its
   protagonist. Cost: ~2.5 h for a 7-chapter story (vs ~10–20 min for the smaller models), and
   ~24 GB VRAM at `num-ctx 24576`. Worth it for quality runs; keep `qwen3:latest` as a fast
   smoke-test model.
2. **`gemma4:26b`: keep around as a same-speed sanity model, not a quality model.** It no
   longer "fails miserably" (that was the 4096-token crash), but it drops/garbles words and
   never names the protagonist. If you want a fast model, `qwen3:latest` writes cleaner prose;
   `gemma4:26b` is a touch more coherent on plot. Roughly a wash.
3. **Variant: keep A as the default, but adopt B's two good ideas; drop C as a standalone.**
   - **A (baseline)** is the best all-rounder at 27B. Its one real wart is the unnamed cold
     open — fix by establishing the protagonist's name in the foundation (or feeding it into
     ch1 drafting, the way B does).
   - **B (world-state)** gives the best narration consistency and the lowest prose repetition,
     but rendering the WorldState blocks into the artifact is a UX bug — render them to a
     sidecar or strip them before "Final Story". With that fixed, B ≈ A; the structured state
     is a genuine asset. Worth merging the WorldState-into-chapter-1 behaviour into A even if
     B itself isn't kept.
   - **C (dspy.Module)** produces the most prose and a coherent arc (the one-shot outline does
     the heavy lifting), but drafting chapters in isolation lets the *voice* drift
     chapter-to-chapter (first person in ch5, "the protagonist" in ch6). Not worth keeping as
     a user-facing mode unless it also threads POV/style state between chapters — at which
     point it converges on B.
4. **`deepseek-v4-pro` (added 2026-05-12): the "I want the best prose and don't mind a hosted
   API" option — not the local default.** On variant A it matched `qwen3.6:27b` on overall
   quality (≈ a wash) while writing distinctly better prose and a more surprising ending, in
   half the wall-clock (43 vs 86 min) for well under $1. But it's a hosted API (against the
   project's local-first grain), it drifts the protagonist's pronoun in ch3, and its back
   third stutters (ch5/ch6 beat duplication, ch6→ch7 cliffhanger replay) — the same
   per-chapter-drafting weakness, just dressed in better sentences. So: keep **`qwen3.6:27b`**
   as the default *local* model; reach for **`deepseek-v4-pro`** when you want a max-quality
   pass and a network call is acceptable. The pipeline fixes below (esp. #1's `None`-guard,
   #3's POV check, and a new world-bible name-uniqueness check) matter for *both* — and run
   `deepseek` with a generous `--max-tokens` (32768 here) since its reasoning tokens eat the
   output budget.

## Cross-cutting issues (independent of model & variant — worth fixing before the next bake-off)

1. **`--max-tokens` default (4096) is too low and the failure mode is a hard crash.** A
   verbose model truncates at the ceiling → DSPy returns `None` for the field → `pipeline.py`
   does `len(enhanced)` / `world_state.py` does `state.story_clock` on `None` and the whole
   run dies. Raise the default (8192 worked here) **and** wrap LM-producing steps with a
   retry + a `None`/empty guard so one bad response doesn't kill a 2-hour run.
2. **QA `character_presence` mis-parses the world bible.** It treats bold list items
   ("**Reclaim Identity**", "**Protect the Nursery**", …) as character names → spurious FAILs
   on every `qwen3.6:27b` run. Tighten the extractor to the actual `#### Character N` blocks.
3. **QA can't see POV / pronoun drift** — variant C's real weakness (first-person ch5,
   "the protagonist" ch6) goes unflagged while `name_drift` fires on harmless paraphrases
   ("High Cardinal Vane" ↔ "High Clergy") and on WorldState-block parsing artifacts
   ("Renn\n\nThe Sanctum"). Add a check for narrator-person consistency and protagonist-naming
   consistency across chapters. *(Reconfirmed 2026-05-12: `deepseek-v4-pro` variant A narrates
   the protagonist as "they/them" in every chapter except ch3, which switches to "he/him" —
   same blind spot, different model and variant.)*
4. **QA still doesn't flag empty / very-short prose** (pre-existing — `runs/demo-hollow` ch3
   was empty and passed R1–R6). Add a hard rule: chapter prose below N words fails.
5. **Variant B leaks pipeline scaffolding into the artifact** — the "#### World State after
   Chapter N" blocks. Render to a sidecar file or strip before the "Final Story" section.
6. **Chapter-1 cold-open has no protagonist name** in A and (worse) C → literal "the
   protagonist" / "the child" placeholder text. B avoids it. Fix in the foundation or in ch1
   drafting.
7. **Per-chapter anti-repetition would help every model.** "silver suppression runes etched
   into…" (qwen3.6:27b) and "kaleidoscopic nightmare of overlapping…" (gemma4:26b) recur
   across all three variants. Feeding prior chapters' sentences into the drafter as
   negative/avoid context would suppress the worst tics.
8. **Nothing enforces character-name uniqueness in the world bible** *(new, found 2026-05-12)*.
   `deepseek-v4-pro` produced a bible/draft with **two characters named Elara** — the Remnant
   exorcist and the Thornwood seamstress — and the prose ended up lampshading it ("Her name is
   Elara," said the other Elara). Add a foundation-stage check that the `#### Character N`
   blocks have distinct names (warn on near-duplicates / first-name collisions). `name_drift`
   can't see this — it's a collision, not drift.
9. **Variant A's per-chapter drafting can duplicate/loop late beats** *(new, found 2026-05-12)*.
   With only a rolling text summary for continuity, `deepseek-v4-pro` pulled ch6's climax
   (storm the chamber → kill ritualists → seize the heart-relic → tethers snap → ferals surge →
   flee up the stair) into ch5, then ch6 re-ran it, and ch7 opened by replaying ch6's
   cliffhanger. Worth: (a) feeding the *prior chapter's actual closing beats* (not just a
   summary) into the next draft as "already happened, do not repeat", and/or (b) a QA check for
   high n-gram/event overlap between adjacent chapters. (Variant B's structured `WorldState`
   would also have caught "the heart-relic is already in the protagonist's possession".)
