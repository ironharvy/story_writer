#!/usr/bin/env python3
"""Emit copy-pasteable Suno prompts (Style + Lyrics) from a story_writer
markdown.

Suno is a music engine first, narrator second. This script doesn't call any
API: it produces per-chunk `.txt` files you paste into Suno's Custom mode
(Style of Music + Lyrics fields). Each chunk:

  * Stays under 2500 characters of prose so total lyrics (prose + tags) fit
    comfortably under Suno's 5000-character lyrics cap with a safe buffer.
  * Packs greedily on paragraph boundaries — never splits a paragraph unless
    that paragraph alone exceeds the cap, in which case it splits on sentence
    boundaries.
  * Wraps each paragraph in a fresh `[Narrator]` tag so Suno re-evaluates
    tone per paragraph (helps when emotion shifts between paragraphs).
  * Prepends `[Spoken Word]` / `[no singing]` / `[no melody]` to bias the
    model away from melodic vocals.

Usage:
  python suno_prompt.py
  python suno_prompt.py --input .tmp/story_output.md --output-dir .tmp/suno
  python suno_prompt.py --style "noir audiobook narration, jazz ambient bed"
  python suno_prompt.py --only 3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from audiobook import Chapter, parse_chapters, clean_for_tts, safe_slug


DEFAULT_STYLE = (
    "spoken word audiobook narration, warm baritone narrator, "
    "gentle cinematic ambient underscore, slow calm pacing, "
    "no singing, no melodic vocals, no chorus, no rap"
)

LYRICS_HEADER = "[Spoken Word]\n[no singing]\n[no melody]\n"
PARA_TAG = "[Narrator]"

PROSE_CAP = 2500            # per-chunk prose budget
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def split_long_paragraph(p: str, cap: int) -> list[str]:
    """Sentence-pack a too-long paragraph into <=cap pieces."""
    sentences = SENTENCE_RE.split(p)
    out: list[str] = []
    buf = ""
    for s in sentences:
        candidate = (buf + " " + s).strip() if buf else s
        if len(candidate) <= cap:
            buf = candidate
            continue
        if buf:
            out.append(buf)
        if len(s) <= cap:
            buf = s
        else:
            # Sentence itself exceeds cap; hard-wrap on whitespace.
            for i in range(0, len(s), cap):
                out.append(s[i : i + cap])
            buf = ""
    if buf:
        out.append(buf)
    return out


def chunk_paragraphs(paragraphs: list[str], cap: int = PROSE_CAP) -> list[list[str]]:
    """Greedy pack paragraphs into chunks up to `cap` chars of prose."""
    chunks: list[list[str]] = []
    current: list[str] = []
    used = 0
    for p in paragraphs:
        pieces = [p] if len(p) <= cap else split_long_paragraph(p, cap)
        for piece in pieces:
            cost = len(piece) + (2 if current else 0)  # +2 for paragraph break
            if used + cost > cap and current:
                chunks.append(current)
                current = [piece]
                used = len(piece)
            else:
                current.append(piece)
                used += cost
    if current:
        chunks.append(current)
    return chunks


def render_lyrics_block(paragraphs: list[str]) -> str:
    body = "\n\n".join(f"{PARA_TAG}\n{p}" for p in paragraphs)
    return f"{LYRICS_HEADER}\n{body}\n"


def render_chunk_file(
    chapter: Chapter,
    chunk_idx: int,
    chunk_total: int,
    paragraphs: list[str],
    style: str,
) -> str:
    lyrics = render_lyrics_block(paragraphs)
    prose_chars = sum(len(p) for p in paragraphs)
    tips = (
        "- Use Suno Custom mode; paste Style above into 'Style of Music' and Lyrics into 'Lyrics'.\n"
        "- Leave 'Instrumental' UNCHECKED — you want vocals, just not melodic ones.\n"
        "- If it starts singing, regenerate or strengthen the negative cues "
        "(e.g. add 'monotone, deadpan' to style).\n"
        "- For consistent voice across chunks, reuse the same Persona/Style for every paste."
    )
    return (
        f"=== STYLE (paste into 'Style of Music') ===\n{style}\n\n"
        f"=== LYRICS (paste into 'Lyrics') ===\n{lyrics}\n"
        f"=== TIPS ===\n{tips}\n\n"
        f"--- meta ---\n"
        f"chapter: {chapter.title}\n"
        f"chunk: {chunk_idx}/{chunk_total}\n"
        f"prose_chars: {prose_chars}\n"
        f"lyrics_chars: {len(lyrics)}\n"
    )


def emit_for_chapter(chapter: Chapter, out_dir: Path, style: str) -> list[Path]:
    text = clean_for_tts(chapter.text)
    if not text:
        return []
    paragraphs = split_paragraphs(text)
    chunks = chunk_paragraphs(paragraphs, cap=PROSE_CAP)
    paths: list[Path] = []
    for i, chunk in enumerate(chunks, start=1):
        fname = f"ch{chapter.index:02d}_part{i:02d}_{safe_slug(chapter.title)}.txt"
        path = out_dir / fname
        path.write_text(
            render_chunk_file(chapter, i, len(chunks), chunk, style),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=".tmp/story_output.md")
    ap.add_argument("--output-dir", default=".tmp/suno_prompts")
    ap.add_argument("--style", default=DEFAULT_STYLE,
                    help="Override the Suno 'Style of Music' string.")
    ap.add_argument("--only", type=int, default=None,
                    help="Emit prompts only for this chapter index (1-based).")
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for ch in chapters:
        paths = emit_for_chapter(ch, out_dir, args.style)
        total += len(paths)
        print(f"{ch.title}: {len(paths)} chunk(s)")
    print(f"\nWrote {total} prompt file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
