#!/usr/bin/env python3
"""Render a folder of chapter markdown files into a single self-contained book.html.

Usage:
    python tools/build_book.py                  # defaults to season_1/ -> book.html
    python tools/build_book.py season_2         # different source folder
    python tools/build_book.py season_1 draft.html
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def render_inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*(.+?)\*", r"<em>\1</em>", s)
    return s


def render_blocks(text: str) -> tuple[str, str]:
    """Return (title, body_html). Title is the first H1, stripped from body."""
    text = strip_comments(text).strip()
    blocks = re.split(r"\n\s*\n", text)
    title = ""
    out: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if b.startswith("# "):
            heading = b[2:].strip()
            if not title:
                title = heading
            out.append(f"<h1>{render_inline(heading)}</h1>")
        elif b.startswith("## "):
            out.append(f"<h2>{render_inline(b[3:].strip())}</h2>")
        elif b == "---" or b == "***":
            out.append('<hr aria-hidden="true">')
        else:
            para = " ".join(line.strip() for line in b.splitlines())
            out.append(f"<p>{render_inline(para)}</p>")
    return title, "\n".join(out)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "chapter"


STYLE = """
:root {
  --bg: #faf7f1;
  --fg: #1d1b18;
  --muted: #7a756c;
  --rule: #d8d1c2;
  --accent: #8a3a28;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14120f;
    --fg: #e8e2d4;
    --muted: #948d7e;
    --rule: #2b2722;
    --accent: #d77a5f;
  }
}
* { box-sizing: border-box; }
html, body { background: var(--bg); color: var(--fg); }
body {
  font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Times New Roman", serif;
  max-width: 34rem;
  margin: 3.5rem auto 6rem;
  padding: 0 1.25rem;
  line-height: 1.62;
  font-size: 1.0625rem;
  text-rendering: optimizeLegibility;
  font-kerning: normal;
  hyphens: auto;
}
header.cover {
  text-align: center;
  margin-bottom: 4rem;
  padding-bottom: 3rem;
  border-bottom: 1px solid var(--rule);
}
header.cover .eyebrow {
  font-size: 0.75rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--muted);
}
header.cover h1 {
  font-size: 2.2rem;
  font-weight: 600;
  margin: 0.6rem 0 0.4rem;
  letter-spacing: 0.01em;
}
header.cover .subtitle {
  color: var(--muted);
  font-style: italic;
  margin: 0;
}
nav.toc { margin-bottom: 4rem; }
nav.toc h2 {
  font-size: 0.8rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin: 0 0 1rem;
  text-align: center;
}
nav.toc ol {
  list-style: none;
  padding: 0;
  margin: 0;
  counter-reset: ch;
}
nav.toc li {
  counter-increment: ch;
  padding: 0.35rem 0;
  display: flex;
  gap: 0.9rem;
  align-items: baseline;
  border-bottom: 1px dotted var(--rule);
}
nav.toc li::before {
  content: counter(ch, decimal-leading-zero);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  min-width: 1.7rem;
}
nav.toc a {
  color: var(--fg);
  text-decoration: none;
  flex: 1;
}
nav.toc a:hover { color: var(--accent); }
section.chapter { margin-top: 4rem; }
section.chapter h1 {
  font-size: 1.45rem;
  font-weight: 600;
  margin: 2rem 0 2rem;
  letter-spacing: 0.01em;
  text-align: center;
}
section.chapter p {
  margin: 0;
  text-indent: 1.3em;
  text-align: justify;
  orphans: 2;
  widows: 2;
}
section.chapter h1 + p,
section.chapter h2 + p,
section.chapter hr + p { text-indent: 0; }
section.chapter h1 + p::first-letter {
  font-size: 1.5em;
  font-weight: 600;
}
hr {
  border: none;
  text-align: center;
  margin: 1.4rem 0;
  height: 1rem;
}
hr::after {
  content: "\\2217  \\2217  \\2217";
  letter-spacing: 0.4em;
  color: var(--muted);
  font-size: 0.85rem;
}
footer.colophon {
  margin-top: 6rem;
  padding-top: 2rem;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 0.8rem;
  text-align: center;
}
@media print {
  body { max-width: none; margin: 0; font-size: 11pt; }
  header.cover, nav.toc, section.chapter { page-break-after: always; }
  section.chapter:last-of-type { page-break-after: auto; }
  a { color: inherit; text-decoration: none; }
}
"""


def build(src_dir: Path, out_path: Path) -> None:
    files = sorted(p for p in src_dir.glob("*.md") if p.is_file())
    if not files:
        print(f"No .md files found in {src_dir}", file=sys.stderr)
        sys.exit(1)

    chapters: list[tuple[str, str, str]] = []  # (slug, title, body_html)
    for f in files:
        title, body = render_blocks(f.read_text(encoding="utf-8"))
        slug = slugify(f.stem)
        chapters.append((slug, title or f.stem, body))

    toc_items = "\n".join(
        f'      <li><a href="#{slug}">{html.escape(title)}</a></li>'
        for slug, title, _ in chapters
    )
    chapter_sections = "\n".join(
        f'  <section class="chapter" id="{slug}">\n{body}\n  </section>'
        for slug, _, body in chapters
    )

    book_title = "Echo of Stone"
    subtitle = src_dir.name.replace("_", " ").title()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(book_title)} — {html.escape(subtitle)}</title>
  <style>{STYLE}</style>
</head>
<body>
  <header class="cover">
    <div class="eyebrow">{html.escape(subtitle)}</div>
    <h1>{html.escape(book_title)}</h1>
    <p class="subtitle">a draft in {len(chapters)} chapters</p>
  </header>
  <nav class="toc" aria-label="Table of contents">
    <h2>Contents</h2>
    <ol>
{toc_items}
    </ol>
  </nav>
{chapter_sections}
  <footer class="colophon">
    Built from <code>{html.escape(str(src_dir.relative_to(ROOT)))}</code>. Print this page to export as PDF.
  </footer>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)} ({len(chapters)} chapters)")


def main(argv: list[str]) -> None:
    src = Path(argv[1]) if len(argv) > 1 else Path("season_1")
    out = Path(argv[2]) if len(argv) > 2 else Path("book.html")
    if not src.is_absolute():
        src = ROOT / src
    if not out.is_absolute():
        out = ROOT / out
    if not src.is_dir():
        print(f"Source folder not found: {src}", file=sys.stderr)
        sys.exit(1)
    build(src, out)


if __name__ == "__main__":
    main(sys.argv)
