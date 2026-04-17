#!/usr/bin/env python3
"""Standalone audiobook synthesizer for story_writer markdown output.

Reads a story markdown file (default `.tmp/story_output.md`), extracts chapters
from the `## Final Story` section, and renders one audio file per chapter via a
selectable TTS provider. Local and remote providers share a thin adapter
interface; pick via `--provider`.

Supported providers:
  Local   : kokoro, piper
  Remote  : nvidia (Magpie TTS NIM; self-hosted or build.nvidia.com),
            openai, elevenlabs

Each provider imports its SDK lazily so you only need the dependency you use.

Examples:
  python audiobook.py --provider kokoro
  python audiobook.py --provider piper --voice-model en_US-lessac-medium.onnx
  python audiobook.py --provider nvidia --nvidia-url http://localhost:9000 \\
      --voice Magpie-Multilingual.EN-US.Aria
  python audiobook.py --provider openai --voice nova
  python audiobook.py --provider elevenlabs --voice Rachel
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


# ---------- Markdown parsing ----------

FINAL_STORY_HEADER = re.compile(r"^##\s+Final Story\s*$", re.MULTILINE)
NEXT_H2 = re.compile(r"^##\s+\S", re.MULTILINE)
CHAPTER_SPLIT = re.compile(r"^###\s+Chapter\s+", re.MULTILINE)

IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
INLINE_FMT_RE = re.compile(r"(\*\*|\*|__|_|`)")
HEADING_HASHES_RE = re.compile(r"^#+\s*", re.MULTILINE)


@dataclass
class Chapter:
    index: int
    title: str
    text: str


def extract_final_story(md: str) -> str:
    m = FINAL_STORY_HEADER.search(md)
    if not m:
        raise SystemExit("No '## Final Story' section found in input markdown.")
    start = m.end()
    nxt = NEXT_H2.search(md, pos=start)
    return md[start : nxt.start() if nxt else len(md)]


def parse_chapters(md: str) -> list[Chapter]:
    body = extract_final_story(md)
    pieces = CHAPTER_SPLIT.split(body)
    pieces = [p for p in pieces if p.strip()]
    chapters: list[Chapter] = []
    for i, piece in enumerate(pieces, start=1):
        first_line, _, rest = piece.partition("\n")
        title = first_line.strip().rstrip(":") or f"Chapter {i}"
        chapters.append(Chapter(index=i, title=f"Chapter {title}", text=rest.strip()))
    if not chapters:
        raise SystemExit("Found '## Final Story' but no '### Chapter ...' entries.")
    return chapters


def clean_for_tts(text: str) -> str:
    text = IMG_RE.sub("", text)
    text = LINK_RE.sub(r"\1", text)
    text = HEADING_HASHES_RE.sub("", text)
    text = INLINE_FMT_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- Provider interface ----------

class TTSProvider(ABC):
    """One audio file per synth() call. Output format is provider-dependent
    (wav/mp3); callers use the returned path as-is."""

    name: str = "base"
    default_ext: str = "wav"

    @abstractmethod
    def synth(self, text: str, out_path: Path, voice: str | None) -> Path: ...


# ---------- Local: Kokoro ----------

class KokoroProvider(TTSProvider):
    name = "kokoro"
    default_ext = "wav"

    def __init__(self, voice_default: str = "af_heart", lang: str = "a"):
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError as e:
            raise SystemExit(
                "kokoro not installed. Try: pip install kokoro soundfile"
            ) from e
        self._KPipeline = KPipeline
        self._pipe = KPipeline(lang_code=lang)
        self._voice_default = voice_default

    def synth(self, text: str, out_path: Path, voice: str | None) -> Path:
        import numpy as np
        import soundfile as sf
        v = voice or self._voice_default
        chunks = []
        for _, _, audio in self._pipe(text, voice=v):
            chunks.append(audio)
        if not chunks:
            raise RuntimeError("Kokoro produced no audio.")
        audio = np.concatenate(chunks)
        sf.write(str(out_path), audio, 24000)
        return out_path


# ---------- Local: Piper ----------

class PiperProvider(TTSProvider):
    name = "piper"
    default_ext = "wav"

    def __init__(self, voice_model: str):
        if not voice_model:
            raise SystemExit(
                "--voice-model is required for piper (path to a .onnx voice)."
            )
        try:
            from piper.voice import PiperVoice  # type: ignore
        except ImportError as e:
            raise SystemExit("piper-tts not installed. Try: pip install piper-tts") from e
        self._voice = PiperVoice.load(voice_model)

    def synth(self, text: str, out_path: Path, voice: str | None) -> Path:
        import wave
        with wave.open(str(out_path), "wb") as wf:
            self._voice.synthesize(text, wf)
        return out_path


# ---------- Remote: NVIDIA Magpie TTS NIM ----------

class NvidiaMagpieProvider(TTSProvider):
    """Works against a self-hosted Magpie NIM container or a build.nvidia.com
    hosted endpoint. Uses the NIM REST surface at /v1/audio/synthesize."""

    name = "nvidia"
    default_ext = "wav"

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        language: str = "en-US",
        voice_default: str = "Magpie-Multilingual.EN-US.Aria",
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._language = language
        self._voice_default = voice_default

    def synth(self, text: str, out_path: Path, voice: str | None) -> Path:
        try:
            import requests
        except ImportError as e:
            raise SystemExit("requests not installed. Try: pip install requests") from e
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self._base_url}/v1/audio/synthesize"
        r = requests.post(
            url,
            headers=headers,
            data={
                "language": self._language,
                "text": text,
                "voice": voice or self._voice_default,
            },
            timeout=600,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"NVIDIA TTS error {r.status_code}: {r.text[:300]}")
        out_path.write_bytes(r.content)
        return out_path


# ---------- Remote: OpenAI ----------

class OpenAIProvider(TTSProvider):
    name = "openai"
    default_ext = "mp3"

    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini-tts"):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise SystemExit("openai not installed. Try: pip install openai") from e
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model

    def synth(self, text: str, out_path: Path, voice: str | None) -> Path:
        with self._client.audio.speech.with_streaming_response.create(
            model=self._model,
            voice=voice or "nova",
            input=text,
            response_format="mp3",
        ) as resp:
            resp.stream_to_file(str(out_path))
        return out_path


# ---------- Remote: ElevenLabs ----------

class ElevenLabsProvider(TTSProvider):
    name = "elevenlabs"
    default_ext = "mp3"

    def __init__(self, api_key: str | None, model: str = "eleven_multilingual_v2"):
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore
        except ImportError as e:
            raise SystemExit("elevenlabs not installed. Try: pip install elevenlabs") from e
        self._client = ElevenLabs(api_key=api_key) if api_key else ElevenLabs()
        self._model = model

    def synth(self, text: str, out_path: Path, voice: str | None) -> Path:
        audio = self._client.text_to_speech.convert(
            voice_id=voice or "JBFqnCBsd6RMkjVDRZzb",
            model_id=self._model,
            text=text,
            output_format="mp3_44100_128",
        )
        with open(out_path, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)
        return out_path


# ---------- Orchestration ----------

def build_provider(args: argparse.Namespace) -> TTSProvider:
    p = args.provider
    if p == "kokoro":
        return KokoroProvider(voice_default=args.voice or "af_heart", lang=args.kokoro_lang)
    if p == "piper":
        return PiperProvider(voice_model=args.voice_model)
    if p == "nvidia":
        key = args.api_key or os.getenv("NVIDIA_API_KEY")
        return NvidiaMagpieProvider(
            base_url=args.nvidia_url,
            api_key=key,
            language=args.nvidia_language,
            voice_default=args.voice or "Magpie-Multilingual.EN-US.Aria",
        )
    if p == "openai":
        key = args.api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIProvider(api_key=key, model=args.openai_model)
    if p == "elevenlabs":
        key = args.api_key or os.getenv("ELEVENLABS_API_KEY")
        return ElevenLabsProvider(api_key=key, model=args.elevenlabs_model)
    raise SystemExit(f"Unknown provider: {p}")


def safe_slug(s: str, limit: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    return s[:limit] or "chapter"


def render_chapters(
    chapters: list[Chapter],
    provider: TTSProvider,
    voice: str | None,
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ch in chapters:
        text = clean_for_tts(ch.text)
        if not text:
            print(f"[skip] Chapter {ch.index}: empty after cleaning", file=sys.stderr)
            continue
        fname = f"ch{ch.index:02d}_{safe_slug(ch.title)}.{provider.default_ext}"
        out_path = out_dir / fname
        print(f"[{provider.name}] Chapter {ch.index}: {out_path}")
        provider.synth(text, out_path, voice)
        paths.append(out_path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a story markdown into per-chapter audio.")
    ap.add_argument("--input", default=".tmp/story_output.md",
                    help="Path to story markdown (default: .tmp/story_output.md)")
    ap.add_argument("--output-dir", default=".tmp/audiobook",
                    help="Where to write per-chapter audio files.")
    ap.add_argument("--provider", required=True,
                    choices=["kokoro", "piper", "nvidia", "openai", "elevenlabs"])
    ap.add_argument("--voice", default=None,
                    help="Provider-specific voice identifier (falls back to a sensible default).")
    ap.add_argument("--api-key", default=None, help="Remote-provider API key (overrides env).")
    ap.add_argument("--only", type=int, default=None,
                    help="Render only this chapter index (1-based).")

    # Kokoro
    ap.add_argument("--kokoro-lang", default="a",
                    help="Kokoro lang_code: a=American, b=British, etc.")
    # Piper
    ap.add_argument("--voice-model", default=None,
                    help="Piper: path to .onnx voice model.")
    # NVIDIA
    ap.add_argument("--nvidia-url", default="http://localhost:9000",
                    help="Base URL for Magpie TTS NIM (local container or hosted endpoint).")
    ap.add_argument("--nvidia-language", default="en-US")
    # OpenAI
    ap.add_argument("--openai-model", default="gpt-4o-mini-tts")
    # ElevenLabs
    ap.add_argument("--elevenlabs-model", default="eleven_multilingual_v2")

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    md_path = Path(args.input)
    if not md_path.exists():
        raise SystemExit(f"Input not found: {md_path}")
    chapters = parse_chapters(md_path.read_text(encoding="utf-8"))
    if args.only is not None:
        chapters = [c for c in chapters if c.index == args.only]
        if not chapters:
            raise SystemExit(f"--only {args.only} did not match any chapter.")
    provider = build_provider(args)
    paths = render_chapters(chapters, provider, args.voice, Path(args.output_dir))
    print(f"\nWrote {len(paths)} file(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
