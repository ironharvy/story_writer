#!/usr/bin/env python3
"""Render a single story_output.md into a self-contained HTML file.

Intended for reviewing output from ``test_story.py`` when driving it with a
real LLM (e.g. Ollama) instead of the MockLM.

Usage:
    python scripts/render_story.py                              # .tmp/story_output.md -> .tmp/story_output.html
    python scripts/render_story.py path/to/story.md             # writes path/to/story.html
    python scripts/render_story.py path/to/story.md out.html    # explicit output path
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


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
  max-width: 38rem;
  margin: 3rem auto 5rem;
  padding: 0 1.25rem;
  line-height: 1.62;
  font-size: 1.0625rem;
  text-rendering: optimizeLegibility;
  hyphens: auto;
}
header.cover {
  text-align: center;
  margin-bottom: 3rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--rule);
}
header.cover h1 {
  font-size: 2rem;
  font-weight: 600;
  margin: 0 0 0.4rem;
}
header.cover .meta {
  color: var(--muted);
  font-size: 0.8rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
nav.toc { margin-bottom: 3rem; }
nav.toc h2 {
  font-size: 0.75rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 600;
  margin: 0 0 0.75rem;
}
nav.toc ol {
  list-style: none;
  padding: 0;
  margin: 0;
  counter-reset: sec;
}
nav.toc li {
  counter-increment: sec;
  padding: 0.3rem 0;
  display: flex;
  gap: 0.8rem;
  align-items: baseline;
  border-bottom: 1px dotted var(--rule);
}
nav.toc li::before {
  content: counter(sec, decimal-leading-zero);
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  min-width: 1.6rem;
}
nav.toc a { color: var(--fg); text-decoration: none; flex: 1; }
nav.toc a:hover { color: var(--accent); }
section.part { margin-top: 3rem; }
section.part h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 2rem 0 1rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--rule);
}
section.part h3 {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 1.5rem 0 0.5rem;
}
section.part p {
  margin: 0 0 0.9rem;
  text-align: justify;
}
section.part ul {
  margin: 0 0 1rem;
  padding-left: 1.25rem;
}
section.part li { margin: 0.2rem 0; }
hr {
  border: none;
  text-align: center;
  margin: 1.2rem 0;
  height: 1rem;
}
hr::after {
  content: "\\2217  \\2217  \\2217";
  letter-spacing: 0.4em;
  color: var(--muted);
  font-size: 0.85rem;
}
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.92em;
  background: var(--rule);
  padding: 0.05em 0.3em;
  border-radius: 3px;
}
footer.colophon {
  margin-top: 5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 0.8rem;
  text-align: center;
}
"""


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _isolate_headings(text: str) -> str:
    """Ensure ATX headings are surrounded by blank lines so block splitting
    treats them as standalone blocks. Real-world LLM output often skips one
    or both of these separators."""
    text = re.sub(r"(?<!\n\n)(\n)(#{1,3} )", r"\n\n\2", text)
    text = re.sub(r"(^|\n)(#{1,3} [^\n]+)\n(?!\n)", r"\1\2\n\n", text)
    return text


def render_inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", s)
    return s


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return s or "section"


_BULLET_RE = re.compile(r"^\s*[-*]\s+")


def render_blocks(blocks: list[str]) -> str:
    out: list[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if b.startswith("### "):
            out.append(f"<h3>{render_inline(b[4:].strip())}</h3>")
        elif b.startswith("## "):
            out.append(f"<h2>{render_inline(b[3:].strip())}</h2>")
        elif b in ("---", "***"):
            out.append('<hr aria-hidden="true">')
        elif all(_BULLET_RE.match(line) for line in b.splitlines() if line.strip()):
            items = [
                f"  <li>{render_inline(_BULLET_RE.sub('', line).strip())}</li>"
                for line in b.splitlines()
                if line.strip()
            ]
            out.append("<ul>\n" + "\n".join(items) + "\n</ul>")
        else:
            para = " ".join(line.strip() for line in b.splitlines())
            out.append(f"<p>{render_inline(para)}</p>")
    return "\n".join(out)


def parse_document(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (title, [(slug, heading, body_html), ...]).

    First H1 becomes the document title. Each H2 starts a new section. Content
    before the first H2 (after the title) is grouped under an implicit
    "Overview" section so nothing is lost.
    """
    text = _isolate_headings(strip_comments(text)).strip()
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]

    title = ""
    if blocks and blocks[0].lstrip().startswith("# "):
        title = blocks[0].lstrip()[2:].strip()
        blocks = blocks[1:]

    sections: list[tuple[str, str, list[str]]] = []
    current_heading: str | None = None
    current_blocks: list[str] = []

    for b in blocks:
        if b.lstrip().startswith("## "):
            if current_heading is not None or current_blocks:
                heading = current_heading or "Overview"
                sections.append((slugify(heading), heading, current_blocks))
            current_heading = b.lstrip()[3:].strip()
            current_blocks = []
        else:
            current_blocks.append(b)

    if current_heading is not None or current_blocks:
        heading = current_heading or "Overview"
        sections.append((slugify(heading), heading, current_blocks))

    rendered = [
        (slug, heading, render_blocks(body)) for slug, heading, body in sections
    ]
    return title or "Story Output", rendered


def build_html(title: str, sections: list[tuple[str, str, str]], source: Path) -> str:
    toc_items = "\n".join(
        f'      <li><a href="#{slug}">{html.escape(heading)}</a></li>'
        for slug, heading, _ in sections
    )
    section_html = "\n".join(
        f'  <section class="part" id="{slug}">\n    <h2>{html.escape(heading)}</h2>\n{body}\n  </section>'
        for slug, heading, body in sections
    )
    toc_block = (
        f'  <nav class="toc" aria-label="Table of contents">\n'
        f"    <h2>Contents</h2>\n"
        f"    <ol>\n{toc_items}\n    </ol>\n"
        f"  </nav>"
        if sections
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{STYLE}</style>
</head>
<body>
  <header class="cover">
    <div class="meta">story_writer preview</div>
    <h1>{html.escape(title)}</h1>
  </header>
{toc_block}
{section_html}
  <footer class="colophon">
    Rendered from <code>{html.escape(source.name)}</code>.
  </footer>
</body>
</html>
"""


def main(argv: list[str]) -> None:
    src = Path(argv[1]) if len(argv) > 1 else Path(".tmp/story_output.md")
    out = Path(argv[2]) if len(argv) > 2 else src.with_suffix(".html")

    if not src.is_file():
        print(f"Input not found: {src}", file=sys.stderr)
        sys.exit(1)

    title, sections = parse_document(src.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(title, sections, src), encoding="utf-8")
    print(f"Wrote {out} ({len(sections)} sections)")


if __name__ == "__main__":
    main(sys.argv)
