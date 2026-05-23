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

## Method

Run the **same fixed passage** through every engine with a comparable voice. Differences
should be the model, not the input. Listen blind, score, run each twice to catch drift.

## Test passage

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

## Scoring (1-5 each)

| Engine | Emotional range | Naturalness | Dialogue | Stability | Speed (s) | Notes |
|---|---|---|---|---|---|---|
| Chatterbox | | | | | | |
| Higgs v2 | | | | | | |
| Kokoro | | | | | | |
| | | | | | | |

- Emotional range: shout vs whisper vs tender -- real dynamics, not monotone.
- Naturalness: breaths, pacing, no robotic seams.
- Dialogue: the two characters sound distinct / intentful.
- Stability: no skipped/garbled words; name + date correct across two runs.
- Speed: wall-clock to render the passage on the 4090.

## Decision

Pick the highest combined score, weighting emotional range + stability. Once chosen, wire
it into the pipeline as an optional `--audiobook` step (markdown -> chunked synthesis ->
m4b), with a Whisper pass to verify chunks for the expressive (drift-prone) engines.
