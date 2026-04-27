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

LLM-segmented mode (`--llm-segment`) asks a DSPy-configured model to split
each chapter into emotionally coherent segments and author a fresh Suno
"Style of Music" string for each one. Falls back to greedy chunking if the
LLM output fails validation.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional

from audiobook import Chapter, parse_chapters, clean_for_tts, safe_slug


logger = logging.getLogger("suno_prompt")


DEFAULT_STYLE = (
    "spoken word audiobook narration, warm baritone narrator, "
    "gentle cinematic ambient underscore, slow calm pacing, "
    "no singing, no melodic vocals, no chorus, no rap"
)

LYRICS_HEADER = "[Spoken Word]\n[no singing]\n[no melody]\n"
PARA_TAG = "[Narrator]"

PROSE_CAP = 2500            # per-chunk prose budget
PROSE_FLOOR = 500           # min chars per LLM segment (single-paragraph segments may be smaller)
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


# ---------- LLM-based segmentation (DSPy) ----------

def _build_segment_signature():
    """Lazy-build the DSPy signature/module so we don't import dspy unless used."""
    import dspy  # type: ignore
    from pydantic import BaseModel, Field

    class ChapterSegment(BaseModel):
        paragraph_start: int = Field(description="0-indexed first paragraph (inclusive).")
        paragraph_end: int = Field(description="0-indexed last paragraph (inclusive).")
        style: str = Field(description=(
            "Suno 'Style of Music' string for this segment. Describe genre, "
            "mood, instrumentation, tempo. Discourage melodic singing — this "
            "is spoken-word audiobook narration with a music underscore."
        ))
        rationale: str = Field(description="One sentence explaining why these paragraphs form one beat.")

    class SegmentChapterSig(dspy.Signature):
        """Split a chapter into contiguous segments for music-backed
        spoken-word narration via Suno. Each segment becomes a single Suno
        generation and gets its own Style string matched to the segment's
        emotional/narrative tone.

        Hard rules:
        - Segments are contiguous and ordered: segment[0].paragraph_start = 0,
          segment[i].paragraph_start = segment[i-1].paragraph_end + 1, and
          the last segment ends at len(paragraphs) - 1.
        - Total prose chars in each segment must be <= max_chars.
        - Multi-paragraph segments must have prose chars >= min_chars.
          A single-paragraph segment may be shorter than min_chars only if
          that paragraph's tone clearly differs from its neighbors.
        - Prefer fewer, longer segments over many tiny ones; only split when
          there's a real shift in tone, pace, location, or stakes.
        - Style strings must be specific (instruments, mood, tempo) and
          must include negative cues against melodic vocals (e.g.
          "no singing, no melody on vocals, narration only").
        """
        chapter_title: str = dspy.InputField()
        paragraphs: list[str] = dspy.InputField(desc="Chapter paragraphs in order, 0-indexed.")
        min_chars: int = dspy.InputField()
        max_chars: int = dspy.InputField()
        segments: list[ChapterSegment] = dspy.OutputField()

    class ChapterSegmenter(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = dspy.ChainOfThought(SegmentChapterSig)

        def forward(self, chapter_title, paragraphs, min_chars, max_chars):
            return self.predict(
                chapter_title=chapter_title,
                paragraphs=paragraphs,
                min_chars=min_chars,
                max_chars=max_chars,
            )

    return ChapterSegmenter, ChapterSegment


def configure_dspy(model: str, llm_url: Optional[str], api_key: Optional[str]) -> None:
    import dspy  # type: ignore
    kwargs = {}
    if llm_url:
        kwargs["api_base"] = llm_url
    if api_key:
        kwargs["api_key"] = api_key
    elif "openai" in model.lower() and os.environ.get("OPENAI_API_KEY"):
        kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
    lm = dspy.LM(model, max_tokens=4000, **kwargs)
    dspy.configure(lm=lm)


def _segments_valid(
    segments,
    paragraphs: list[str],
    min_chars: int,
    max_chars: int,
) -> bool:
    if not segments:
        return False
    n = len(paragraphs)
    if segments[0].paragraph_start != 0 or segments[-1].paragraph_end != n - 1:
        return False
    for i, seg in enumerate(segments):
        if seg.paragraph_start > seg.paragraph_end or seg.paragraph_end >= n:
            return False
        if i > 0 and seg.paragraph_start != segments[i - 1].paragraph_end + 1:
            return False
        if not seg.style or not seg.style.strip():
            return False
        span = paragraphs[seg.paragraph_start : seg.paragraph_end + 1]
        prose_chars = sum(len(p) for p in span) + 2 * max(0, len(span) - 1)
        if prose_chars > max_chars:
            return False
        if len(span) > 1 and prose_chars < min_chars:
            return False
    return True


def llm_segment_chapter(
    chapter: Chapter,
    paragraphs: list[str],
    min_chars: int,
    max_chars: int,
):
    try:
        ChapterSegmenter, _ = _build_segment_signature()
        segmenter = ChapterSegmenter()
        result = segmenter(
            chapter_title=chapter.title,
            paragraphs=paragraphs,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        segments = list(result.segments or [])
    except Exception as e:
        logger.warning("LLM segmentation failed for %s: %s", chapter.title, e)
        return None
    if not _segments_valid(segments, paragraphs, min_chars, max_chars):
        logger.warning(
            "LLM segments for %s failed validation; falling back to greedy chunker.",
            chapter.title,
        )
        return None
    return segments


# ---------- Emission ----------

def _emit_chunk(
    chapter: Chapter,
    chunk_idx: int,
    chunk_total: int,
    paragraphs: list[str],
    style: str,
    out_dir: Path,
) -> Path:
    fname = f"ch{chapter.index:02d}_part{chunk_idx:02d}_{safe_slug(chapter.title)}.txt"
    path = out_dir / fname
    path.write_text(
        render_chunk_file(chapter, chunk_idx, chunk_total, paragraphs, style),
        encoding="utf-8",
    )
    return path


def emit_for_chapter(
    chapter: Chapter,
    out_dir: Path,
    style: str,
    *,
    use_llm: bool = False,
    min_chars: int = PROSE_FLOOR,
) -> list[Path]:
    text = clean_for_tts(chapter.text)
    if not text:
        return []
    paragraphs = split_paragraphs(text)

    segments = (
        llm_segment_chapter(chapter, paragraphs, min_chars, PROSE_CAP)
        if use_llm
        else None
    )

    paths: list[Path] = []
    if segments is not None:
        # One file per LLM segment, using the LLM-authored style. Sentence-split
        # any single-paragraph segment that exceeds the cap.
        emitted_chunks: list[tuple[list[str], str]] = []
        for seg in segments:
            span = paragraphs[seg.paragraph_start : seg.paragraph_end + 1]
            prose_chars = sum(len(p) for p in span) + 2 * max(0, len(span) - 1)
            if prose_chars <= PROSE_CAP:
                emitted_chunks.append((span, seg.style))
                continue
            # Single-paragraph oversize: sentence-split, reuse same style.
            for piece in split_long_paragraph(span[0], PROSE_CAP):
                emitted_chunks.append(([piece], seg.style))
        for i, (chunk, chunk_style) in enumerate(emitted_chunks, start=1):
            paths.append(_emit_chunk(chapter, i, len(emitted_chunks), chunk, chunk_style, out_dir))
        return paths

    # Greedy fallback (also the default when --llm-segment is off).
    chunks = chunk_paragraphs(paragraphs, cap=PROSE_CAP)
    for i, chunk in enumerate(chunks, start=1):
        paths.append(_emit_chunk(chapter, i, len(chunks), chunk, style, out_dir))
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", default=".tmp/story_output.md")
    ap.add_argument("--output-dir", default=".tmp/suno_prompts")
    ap.add_argument("--style", default=DEFAULT_STYLE,
                    help="Default Suno 'Style of Music' string (used unless --llm-segment overrides per segment).")
    ap.add_argument("--only", type=int, default=None,
                    help="Emit prompts only for this chapter index (1-based).")
    ap.add_argument("--llm-segment", action="store_true",
                    help="Use a DSPy LM to split chapters into emotionally coherent segments and author a Suno style per segment.")
    ap.add_argument("--min-chars", type=int, default=PROSE_FLOOR,
                    help=f"Minimum prose chars per multi-paragraph LLM segment (default {PROSE_FLOOR}).")
    ap.add_argument("--model", default=os.environ.get("MODEL", "openai/gpt-4o-mini"),
                    help="DSPy model id (used only with --llm-segment). Defaults to MODEL env var.")
    ap.add_argument("--llm-url", default=os.environ.get("LLM_URL"),
                    help="Custom API base URL (e.g. Ollama).")
    ap.add_argument("--api-key", default=os.environ.get("API_KEY"),
                    help="API key for the LLM (overrides env).")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    md_path = Path(args.input)
    if not md_path.exists():
        raise SystemExit(f"Input not found: {md_path}")
    chapters = parse_chapters(md_path.read_text(encoding="utf-8"))
    if args.only is not None:
        chapters = [c for c in chapters if c.index == args.only]
        if not chapters:
            raise SystemExit(f"--only {args.only} did not match any chapter.")
    if args.llm_segment:
        configure_dspy(args.model, args.llm_url, args.api_key)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for ch in chapters:
        paths = emit_for_chapter(
            ch, out_dir, args.style,
            use_llm=args.llm_segment,
            min_chars=args.min_chars,
        )
        total += len(paths)
        print(f"{ch.title}: {len(paths)} chunk(s)")
    print(f"\nWrote {total} prompt file(s) to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
