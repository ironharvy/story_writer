#!/usr/bin/env python3
"""Render the same passage through a TTS engine for the audiobook evaluation.

Companion to docs/tts-eval.md. Runs one engine at a time so you can A/B the
candidates under identical input. Each engine is an optional dependency,
imported only when selected.

Install (pick the engine you want to test):

    # Kokoro (fast baseline) -- also needs the espeak-ng system package
    pip install kokoro soundfile

    # Chatterbox (expressive)
    pip install chatterbox-tts soundfile

    # Higgs v2 (quality; needs a >=24GB GPU)
    git clone https://github.com/boson-ai/higgs-audio.git
    cd higgs-audio && pip install -r requirements.txt && pip install -e .
    pip install soundfile

Run:

    python scripts/tts_eval.py --engine kokoro --speed 0.9
    # Chatterbox: lower cfg-weight slows/steadies it; chunking kills repetition
    python scripts/tts_eval.py --engine chatterbox --cfg-weight 0.3 --speed 0.92
    # Higgs: low temperature is the fix for uncanny/"creepy" output
    python scripts/tts_eval.py --engine higgs --temperature 0.1 --speed 0.95

Writes out_<engine>.wav and prints render time + real-time factor (RTF).
A reference clip enables voice cloning on Chatterbox (--ref voice.wav).

Anti-artifact notes (see docs/tts-eval.md):
  - Chatterbox repeats/garbles on long text -- it is chunked by sentence here.
  - --speed slows playback without changing pitch (Kokoro native, others via
    ffmpeg atempo) for engines that read too fast.
  - For stubborn Chatterbox artifacts, the Chatterbox-TTS-Extended fork adds a
    denoise + auto-editor cleanup pass.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Tier 1 (plain) passage from docs/tts-eval.md. Kept in sync by hand.
DEFAULT_TEXT = (
    "The lighthouse keeper, Eamon O Briain, hadn't spoken to another soul since "
    "the 17th of November, 2019 -- six hundred and forty-three days.\n\n"
    '"You came back," he whispered. He didn\'t dare move. "After everything... '
    'you actually came back."\n\n'
    'She set the lantern down. "Did you think I wouldn\'t?"\n\n'
    '"Run!" he screamed suddenly, grabbing her wrist. "The tide -- it\'s wrong, '
    "it's coming in wrong!\"\n\n"
    "And then, softer than the wind: \"I never stopped waiting. Not for one "
    'single night."'
)


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _as_mono_float(samples) -> np.ndarray:
    """Coerce a torch tensor or numpy array to a 1-D float32 numpy array."""
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float32)
    return samples.squeeze()


def _split_sentences(text: str) -> list[str]:
    """Split into sentence-ish chunks; Chatterbox repeats/garbles on long input."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _apply_speed_ffmpeg(path: Path, speed: float) -> None:
    """Time-stretch a wav in place without changing pitch (ffmpeg atempo)."""
    if abs(speed - 1.0) < 1e-3:
        return
    if shutil.which("ffmpeg") is None:
        print("warning: ffmpeg not found; --speed ignored", file=sys.stderr)
        return
    tmp = path.with_suffix(".tmp.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-filter:a", f"atempo={speed}", str(tmp)],
        check=True,
    )
    os.replace(tmp, path)


def synth_kokoro(text: str, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise SystemExit("kokoro not installed -- pip install kokoro soundfile") from exc

    pipeline = KPipeline(lang_code=args.lang)
    chunks = [
        _as_mono_float(audio)
        for _gs, _ps, audio in pipeline(
            text, voice=args.voice, speed=args.speed, split_pattern=r"\n+"
        )
    ]
    if not chunks:
        raise SystemExit("Kokoro produced no audio.")
    return np.concatenate(chunks), 24000


def synth_chatterbox(text: str, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as exc:
        raise SystemExit(
            "chatterbox not installed -- pip install chatterbox-tts soundfile"
        ) from exc

    model = ChatterboxTTS.from_pretrained(device=_pick_device(args.device))
    temperature = 0.8 if args.temperature is None else args.temperature
    kwargs = {
        "exaggeration": args.exaggeration,
        "cfg_weight": args.cfg_weight,
        "temperature": temperature,
        "repetition_penalty": args.repetition_penalty,
        "min_p": args.min_p,
    }
    if args.ref:
        kwargs["audio_prompt_path"] = args.ref

    pieces = [text] if args.no_chunk else _split_sentences(text)
    sample_rate = int(model.sr)
    gap = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
    audio: list[np.ndarray] = []
    for index, piece in enumerate(pieces):
        audio.append(_as_mono_float(model.generate(piece, **kwargs)))
        if index != len(pieces) - 1:
            audio.append(gap)
    return np.concatenate(audio), sample_rate


def synth_higgs(text: str, args: argparse.Namespace) -> tuple[np.ndarray, int]:
    try:
        from boson_multimodal.data_types import ChatMLSample, Message
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
    except ImportError as exc:
        raise SystemExit(
            "higgs not installed -- see install notes at the top of this file"
        ) from exc

    system_prompt = (
        "Generate audio following instruction.\n\n"
        f"<|scene_desc_start|>\n{args.scene}\n<|scene_desc_end|>"
    )
    engine = HiggsAudioServeEngine(
        "bosonai/higgs-audio-v2-generation-3B-base",
        "bosonai/higgs-audio-v2-tokenizer",
        device=_pick_device(args.device),
    )
    output = engine.generate(
        chat_ml_sample=ChatMLSample(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=text),
            ]
        ),
        max_new_tokens=args.max_new_tokens,
        temperature=0.3 if args.temperature is None else args.temperature,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    return _as_mono_float(output.audio), int(output.sampling_rate)


ENGINES = {
    "kokoro": synth_kokoro,
    "chatterbox": synth_chatterbox,
    "higgs": synth_higgs,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", required=True, choices=sorted(ENGINES))
    parser.add_argument("--text", help="Inline text to speak (overrides default passage).")
    parser.add_argument("--text-file", type=Path, help="Read text to speak from a file.")
    parser.add_argument("--out", type=Path, help="Output wav path (default out_<engine>.wav).")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (<1 slower). Kokoro native; others via ffmpeg atempo.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: chatterbox 0.8, higgs 0.3). "
        "Lower Higgs to ~0.1 to reduce uncanny output.",
    )

    # Kokoro
    parser.add_argument("--voice", default="af_heart", help="Kokoro voice id.")
    parser.add_argument("--lang", default="a", help="Kokoro lang code ('a' = US English).")

    # Chatterbox
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Chatterbox emotion.")
    parser.add_argument("--cfg-weight", type=float, default=0.5,
                        help="Chatterbox cfg weight; lower (~0.3) slows/steadies pacing.")
    parser.add_argument("--repetition-penalty", type=float, default=1.2,
                        help="Chatterbox repetition penalty; raise to ~1.4 if it repeats.")
    parser.add_argument("--min-p", type=float, default=0.05, help="Chatterbox min_p.")
    parser.add_argument("--no-chunk", action="store_true",
                        help="Disable Chatterbox sentence chunking (chunking reduces artifacts).")
    parser.add_argument("--ref", help="Reference wav for voice cloning (Chatterbox).")

    # Higgs
    parser.add_argument(
        "--scene",
        default="Audio is recorded from a quiet room.",
        help="Higgs scene description (its in-text expressiveness control).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Higgs token cap.")

    args = parser.parse_args(argv)

    if args.text and args.text_file:
        parser.error("pass only one of --text / --text-file")
    if args.text_file:
        text = args.text_file.read_text(encoding="utf-8")
    else:
        text = args.text or DEFAULT_TEXT

    try:
        import soundfile as sf
    except ImportError:
        raise SystemExit("soundfile not installed -- pip install soundfile")

    out_path = args.out or Path(f"out_{args.engine}.wav")

    start = time.perf_counter()
    samples, sample_rate = ENGINES[args.engine](text, args)
    elapsed = time.perf_counter() - start

    sf.write(str(out_path), samples, sample_rate)
    if args.engine != "kokoro":
        _apply_speed_ffmpeg(out_path, args.speed)

    duration = len(samples) / sample_rate
    rtf = elapsed / duration if duration else float("nan")
    print(f"engine={args.engine} device={_pick_device(args.device)}")
    print(f"wrote {out_path}  ({duration:.1f}s audio @ {sample_rate} Hz)")
    print(f"render {elapsed:.1f}s  RTF {rtf:.2f} (lower is faster than real time)")


if __name__ == "__main__":
    main()
