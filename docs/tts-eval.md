# TTS Engine Evaluation (audiobook narration)

Goal: pick a text-to-speech engine for turning generated stories into audiobooks.
Priority is **expressive, emotional delivery** (not flat narration), with acceptable
long-form stability and speed on a single RTX 4090.

## Why not just audiblez/Kokoro

[audiblez](https://github.com/santinic/audiblez) is a clean EPUB to .m4b packager but is
hard-wired to **Kokoro-82M**, which is fast and reliable but emotionally flat. It stays as
the speed/reliability baseline, not the target.

## Test harness

Install **Ultimate TTS Studio** via Pinokio (one-click). It bundles Kokoro, Chatterbox,
Higgs Audio, Fish-Speech, F5 and IndexTTS2 in one Gradio UI with an eBook-to-audiobook
mode, so every candidate is available without per-model setup. Models load/unload on
demand, so a 24 GB 4090 handles all of them (one at a time).

- Ultimate TTS Studio: https://github.com/SUP3RMASS1VE/Ultimate-TTS-Studio-SUP3R-Edition
- Alt (more engines, fiddlier): https://github.com/rsxdalv/TTS-WebUI

## Candidates

| Engine | Why it's in the running |
|---|---|
| Chatterbox (Resemble) | Expressive/storytelling, emotion-exaggeration dial. Top pick for "performs". |
| Higgs Audio v2 (BosonAI) | Most natural emotion + voice cloning. ~18-20 GB with cloning. |
| Kokoro-82M | Control: fast, rock-solid stable, but flat. |
| IndexTTS2 / F5 | Backups if the top two disappoint. |

## Method: two tiers

Each engine exposes a **different, incompatible** way to control expressiveness, so test
both ends:

- **Tier 1 - inference (production-realistic).** Feed the *plain* passage, no markup, no
  per-line tuning. This is how the pipeline will actually run -- we feed generated prose at
  scale and can't hand-annotate thousands of lines. Measures how well the model *infers*
  emotion from context + punctuation alone. **This is the primary score.**
- **Tier 2 - directed (the ceiling).** Apply each engine's *native* control surface to the
  same passage. Measures the best the model can do when told what to feel. Informs whether
  it's worth having the story LLM (or a post-pass) emit annotations.

Run the same passage through every engine with a comparable voice; listen blind; run each
twice to catch drift.

## Test passage (Tier 1, plain)

> The lighthouse keeper, Eamon O Briain, hadn't spoken to another soul since the 17th of
> November, 2019 -- six hundred and forty-three days.
>
> "You came back," he whispered. He didn't dare move. "After everything... you actually
> came back."
>
> She set the lantern down. "Did you think I wouldn't?"
>
> "Run!" he screamed suddenly, grabbing her wrist. "The tide -- it's wrong, it's coming in
> wrong!"
>
> And then, softer than the wind: "I never stopped waiting. Not for one single night."

It deliberately stresses: a quiet whisper, a tender beat, a sudden shout, two-speaker
dialogue, plus stability traps (an unusual proper noun, a date, spelled-out numbers, an
em-dash, a trailing ellipsis, a question).

Listen for: does the shout have real volume/urgency? Does the whisper actually drop? Does
the sad line land, or is it read like a grocery list? Are the name and date correct?

## Control surfaces (Tier 2)

The markup is **per-engine and not portable** -- Kokoro stress marks are not Orpheus tags
are not a Chatterbox slider. Apply each to the same passage:

| Engine | Control surface | How |
|---|---|---|
| Kokoro | In-text prosody (no emotion tags) | stress marks `ˈ ˌ`, punctuation for intonation, stress levels `[word](-1)` / `[word](+2)`, pronunciation `[word](/phonemes/)` |
| Chatterbox | Global parameters (not in-text) | `exaggeration` (emotion intensity), lower `cfg_weight` / temperature for more drama |
| Orpheus | In-text emotion tags | `<laugh> <sigh> <gasp> <groan> <chuckle> <sniffle>` inline |
| Dia | In-text tags | speaker tags `[S1] [S2]`, nonverbals `(gasps) (sighs) (whispers)` |
| Higgs v2 | Reference clip + scene prompt | emotion carried by the cloned reference audio plus a scene/system description |

Example, Kokoro-marked (illustrative -- syntax is Kokoro-only):

> [Eamon O Briain](/ˈeɪmən oʊ ˈbɹiːən/) hadn't spoken to another soul since the 17th of
> November, 2019 -- six hundred and forty-three days.
>
> "You came back," he whispered... "After everything... you actually came back."
>
> "Run! The tide -- it's wrong, it's coming in [wrong](+2)!"

Note the key strategic point: because **we generate the prose**, the cheapest path to
reliable expressiveness may be to have the story LLM (or a light post-pass) emit the target
engine's annotations inline -- the model that wrote the scene knows the intended emotion.
That favors a **tag-based engine (Orpheus / Dia)**, or **Kokoro + a stress post-processor**,
over Chatterbox/Higgs whose control is *global* (one slider / one reference clip per render)
rather than per-line. Weigh this when picking.

## Scoring (1-5 each)

| Engine | Emo (T1 plain) | Emo (T2 directed) | Naturalness | Dialogue | Stability | Speed (s) | Notes |
|---|---|---|---|---|---|---|---|
| Chatterbox | | | | | | | |
| Higgs v2 | | | | | | | |
| Kokoro | | | | | | | |
| | | | | | | | |

- Emo (T1 plain): emotional range inferred from raw prose -- the production-realistic score.
- Emo (T2 directed): emotional ceiling with the engine's native control applied.
- A big T2-minus-T1 gap means the engine needs annotations to shine (factor in pipeline cost).
- Emotional range: shout vs whisper vs tender -- real dynamics, not monotone.
- Naturalness: breaths, pacing, no robotic seams.
- Dialogue: the two characters sound distinct / intentful.
- Stability: no skipped/garbled words; name + date correct across two runs.
- Speed: wall-clock to render the passage on the 4090.

## Decision

Pick the highest combined score, weighting emotional range + stability. Once chosen, wire
it into the pipeline as an optional `--audiobook` step (markdown -> chunked synthesis ->
m4b), with a Whisper pass to verify chunks for the expressive (drift-prone) engines.
