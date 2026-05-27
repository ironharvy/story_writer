#!/usr/bin/env python3
"""Scaffold an aiorchestra story repo from a one-file README idea.

Run this inside (or pass the path to) a fresh git repo whose ``README.md`` holds
the story idea. It:

  - copies the kit's ``scaffold/`` into the repo (``WORKFLOW.md``,
    ``.aiorchestra/``, ``.github/ISSUE_TEMPLATE/``, ``story/``),
  - renders ``AGENTS.md`` / ``CLAUDE.md`` from the README,
  - seeds ``story/idea.md`` with the README text,
  - vendors a ``herenow-skill/`` directory (if found) into
    ``.claude/skills/herenow/``,
  - commits, ensures the ``aiorchestra`` / ``claude`` / ``story`` /
    ``next-phase`` labels exist, optionally pushes, and opens issue #1
    ("Idea: <title>").

stdlib only. Requires the ``gh`` CLI on PATH and authenticated, with the target
repo already created on GitHub. See ``../README.md`` (the kit readme) and the
generated repo's ``WORKFLOW.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

KIT_DIR = Path(__file__).resolve().parent
SCAFFOLD_DIR = KIT_DIR / "scaffold"
TEMPLATES_DIR = KIT_DIR / "repo_md_templates"
DEFAULT_HERENOW = KIT_DIR.parent / "herenow-skill"

INPUT_LABELS = [
    ("aiorchestra", "5319e7", "aiorchestra: discover and run this issue"),
    ("claude", "5319e7", "Route this issue to the Claude Code agent"),
    ("story", "0e8a16", "Part of the story phase chain"),
    (
        "next-phase",
        "fbca04",
        "Next phase drafted and queued — add `aiorchestra` + `claude` after reviewing the previous PR",
    ),
]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def parse_readme(readme: Path) -> tuple[str, str, str]:
    """Return (title, logline, full_text) parsed from a README."""
    text = readme.read_text(encoding="utf-8")
    title = ""
    logline = ""
    for line in text.splitlines():
        s = line.strip()
        if not title and s.startswith("# "):
            title = s[2:].strip()
            continue
        if title and not logline and s and not s.startswith("#"):
            logline = s
            break
    if not title:
        title = readme.parent.resolve().name.replace("-", " ").replace("_", " ").strip().title() or "Untitled Story"
    if not logline:
        logline = title
    return title, logline, text.strip()


def copy_scaffold(repo: Path) -> None:
    for dirpath, _dirnames, filenames in os.walk(SCAFFOLD_DIR):
        rel_dir = Path(dirpath).relative_to(SCAFFOLD_DIR)
        (repo / rel_dir).mkdir(parents=True, exist_ok=True)
        for fn in filenames:
            rel = rel_dir / fn
            dst = repo / rel
            if dst.exists():
                print(f"  skip (exists): {rel}")
                continue
            shutil.copy2(Path(dirpath) / fn, dst)
            print(f"  + {rel}")


def render(tmpl_name: str, out_path: Path, ctx: dict[str, str]) -> None:
    if out_path.exists():
        print(f"  skip (exists): {out_path.name}")
        return
    text = (TEMPLATES_DIR / tmpl_name).read_text(encoding="utf-8")
    for key, val in ctx.items():
        text = text.replace("{{" + key + "}}", val)
    out_path.write_text(text, encoding="utf-8")
    print(f"  + {out_path.name}")


def seed_idea_file(repo: Path, seed: str) -> None:
    idea = repo / "story" / "idea.md"
    body = idea.read_text(encoding="utf-8") if idea.is_file() else "# Idea\n"
    if "## Seed (from README)" in body:
        return
    body = body.rstrip() + "\n\n## Seed (from README)\n\n" + seed.strip() + "\n"
    idea.write_text(body, encoding="utf-8")
    print("  seeded story/idea.md with the README idea")


def vendor_herenow(repo: Path, src: Path) -> None:
    dst = repo / ".claude" / "skills" / "herenow"
    if dst.exists():
        print("  skip (exists): .claude/skills/herenow")
        return
    if not src.is_dir():
        print(f"  note: no herenow skill at {src} — the Assemble & Publish phase will skip publishing")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    print(f"  + .claude/skills/herenow  (from {src})")


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["git", *args], cwd=repo, check=check)


def ensure_labels(repo: Path) -> None:
    for name, color, desc in INPUT_LABELS:
        res = run(["gh", "label", "create", name, "--color", color, "--description", desc], cwd=repo, check=False)
        if res.returncode == 0:
            print(f"  label created: {name}")
        elif "already exists" in (res.stderr + res.stdout).lower():
            print(f"  label ok: {name}")
        else:
            print(f"  warn: could not create label {name!r}: {res.stderr.strip() or res.stdout.strip()}")


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip("\n")
    return text


def create_first_issue(repo: Path, title: str) -> str:
    tmpl = (repo / ".github" / "ISSUE_TEMPLATE" / "idea.md").read_text(encoding="utf-8")
    body = strip_front_matter(tmpl).replace("<story title>", title)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
        fh.write(body)
        body_file = fh.name
    try:
        res = run(
            [
                "gh", "issue", "create",
                "--title", f"Idea: {title}",
                "--body-file", body_file,
                "--label", "aiorchestra",
                "--label", "claude",
                "--label", "story",
            ],
            cwd=repo,
        )
    finally:
        os.unlink(body_file)
    out = (res.stdout or "").strip()
    return out.splitlines()[-1] if out else "(created — see `gh issue list`)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Scaffold an aiorchestra story repo from a README idea.")
    ap.add_argument("repo_path", nargs="?", default=".", help="target git repo (default: current directory)")
    ap.add_argument(
        "--herenow-skill",
        default=str(DEFAULT_HERENOW),
        help=f"path to a herenow-skill directory to vendor in (default: {DEFAULT_HERENOW})",
    )
    ap.add_argument("--push", action="store_true", help="git push -u <remote> HEAD after committing")
    ap.add_argument("--remote", default="origin", help="remote name for --push (default: origin)")
    ap.add_argument("--no-issue", action="store_true", help="don't create issue #1 (just scaffold + commit)")
    args = ap.parse_args()

    if not shutil.which("gh"):
        die("the `gh` CLI is required and was not found on PATH")
    if not shutil.which("git"):
        die("git is required and was not found on PATH")

    repo = Path(args.repo_path).resolve()
    if not (repo / ".git").is_dir():
        die(f"{repo} is not a git repository (run `git init` there first)")
    readme = repo / "README.md"
    if not readme.is_file():
        die(f"{readme} not found — put your story idea in README.md first")

    if not args.no_issue:
        view = run(["gh", "repo", "view"], cwd=repo, check=False)
        if view.returncode != 0:
            die(
                "`gh` can't resolve this repo on GitHub — create it and push first "
                "(e.g. `gh repo create <name> --source=. --remote=origin --push`), or re-run with --no-issue"
            )

    title, logline, seed = parse_readme(readme)
    print(f"title:   {title}")
    print(f"logline: {logline}\n")

    print("scaffolding files…")
    copy_scaffold(repo)

    print("rendering agent docs…")
    ctx = {"TITLE": title, "LOGLINE": logline, "SEED": seed}
    render("AGENTS.md.tmpl", repo / "AGENTS.md", ctx)
    render("CLAUDE.md.tmpl", repo / "CLAUDE.md", ctx)

    seed_idea_file(repo, seed)
    vendor_herenow(repo, Path(args.herenow_skill).expanduser())

    print("committing…")
    git(repo, "add", "-A")
    commit = git(repo, "commit", "-m", "Scaffold aiorchestra story kit", check=False)
    if commit.returncode != 0:
        print(f"  note: nothing committed ({commit.stderr.strip() or commit.stdout.strip()})")

    print("ensuring labels…")
    ensure_labels(repo)

    if args.push:
        print(f"pushing to {args.remote}…")
        push = git(repo, "push", "-u", args.remote, "HEAD", check=False)
        if push.returncode != 0:
            print(f"  warn: push failed: {push.stderr.strip()}")

    if args.no_issue:
        print("\nDone (scaffold only). Create issue #1 yourself, e.g.:")
        print(
            f'  gh issue create --title "Idea: {title}" '
            "--body-file .github/ISSUE_TEMPLATE/idea.md --label aiorchestra --label claude --label story"
        )
        return

    print("creating issue #1…")
    url = create_first_issue(repo, title)
    print(f"\nDone. Issue #1: {url}")
    print("aiorchestra (with the claude-code provider) will pick it up via the `aiorchestra` label.")
    if not args.push:
        print(f"Reminder: you still need to `git push -u {args.remote} HEAD`.")
    print("After you merge issue #1's PR, add `aiorchestra` + `claude` to the `Premise: …` issue the agent files.")


if __name__ == "__main__":
    main()
