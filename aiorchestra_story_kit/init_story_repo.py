#!/usr/bin/env python3
"""Scaffold an aiorchestra story repo from a one-file README idea.

Run this inside (or pass the path to) the directory you want the story project
in — ideally a fresh git repo whose ``README.md`` holds the story idea, but it
will offer to set those up for you if they're missing. It:

  - copies the kit's ``scaffold/`` into the repo (``WORKFLOW.md``,
    ``.aiorchestra/``, ``.github/ISSUE_TEMPLATE/``, ``story/``),
  - renders ``AGENTS.md`` / ``CLAUDE.md`` from the README,
  - seeds ``story/idea.md`` with the README text,
  - vendors a ``herenow-skill/`` directory (if found) into
    ``.claude/skills/herenow/``,
  - commits, ensures the ``aiorchestra`` / ``claude`` / ``story`` /
    ``next-phase`` labels exist, optionally pushes, and opens issue #1
    ("Idea: <title>").

stdlib only. The script is interactive: if the target directory isn't a git repo
yet it offers to ``git init`` it, if there's no ``README.md`` it offers to create
a starter one, and if the repo isn't on GitHub yet it offers to ``gh repo
create`` it. Pass ``--yes`` to accept every prompt, or ``--no-issue`` to just
scaffold + commit (no GitHub needed). Creating issue #1 needs the ``gh`` CLI on
PATH and authenticated. See ``../README.md`` (the kit readme) and the generated
repo's ``WORKFLOW.md``.
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


def die(msg: str, *hints: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    for hint in hints:
        print(f"  → {hint}", file=sys.stderr)
    raise SystemExit(1)


_ASSUME_YES = False


def confirm(question: str, *, default: bool = True) -> bool:
    """Ask a yes/no question. Honours --yes; safe when stdin isn't a TTY."""
    if _ASSUME_YES:
        print(f"{question} [--yes]")
        return True
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        if not sys.stdin.isatty():
            return False
        ans = input(f"{question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if not ans:
        return default
    return ans in ("y", "yes")


def ask(prompt: str, *, default: str = "") -> str:
    if _ASSUME_YES or not sys.stdin.isatty():
        return default
    try:
        ans = input(f"{prompt} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return ans or default


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


def ensure_git_repo(repo: Path) -> None:
    if (repo / ".git").is_dir():
        return
    print(f"{repo} is not a git repository yet.")
    if not confirm("Initialize one here now?", default=True):
        die(
            f"{repo} is not a git repository",
            "run `git init` there first, then re-run this script",
        )
    res = git(repo, "init", check=False)
    if res.returncode != 0:
        die(f"`git init` failed: {res.stderr.strip() or res.stdout.strip()}")
    print(f"  + git init  ({repo})")


def ensure_readme(repo: Path) -> Path:
    readme = repo / "README.md"
    if readme.is_file():
        return readme
    print("No README.md found — this is where your story idea lives.")
    if not confirm("Create a starter README.md now?", default=True):
        die(
            f"{readme} not found",
            "put your story idea in README.md (a `# Title` line + a one-line logline), then re-run",
        )
    default_title = repo.name.replace("-", " ").replace("_", " ").strip().title() or "Untitled Story"
    title = ask(f"Story title [{default_title}]:", default=default_title)
    logline = ask("One-line logline (optional):", default="")
    body = f"# {title}\n"
    if logline:
        body += f"\n{logline}\n"
    else:
        body += "\n_Replace this line with a one- or two-sentence logline for your story._\n"
    readme.write_text(body, encoding="utf-8")
    print(f"  + README.md  (\"{title}\")")
    print("  (edit README.md any time to flesh out the idea — it also seeds story/idea.md)")
    return readme


def ensure_gh_ready(repo: Path, *, want_issue: bool) -> bool:
    """Return True if `gh` can open issue #1 here, False to fall back to scaffold-only."""
    if not shutil.which("gh"):
        if not want_issue:
            return False
        die(
            "the `gh` CLI is required to create issue #1 and was not found on PATH",
            "install it from https://cli.github.com/ and run `gh auth login`",
            "or re-run with --no-issue to just scaffold + commit",
        )
    auth = run(["gh", "auth", "status"], cwd=repo, check=False)
    if auth.returncode != 0:
        if not want_issue:
            return False
        die(
            "`gh` is installed but not authenticated",
            "run `gh auth login`",
            "or re-run with --no-issue to just scaffold + commit",
        )
    if not want_issue:
        return False
    view = run(["gh", "repo", "view"], cwd=repo, check=False)
    if view.returncode == 0:
        return True
    print("This repo doesn't exist on GitHub yet (needed before issue #1 can be opened).")
    if not confirm("Create it on GitHub now with `gh repo create`?", default=True):
        die(
            "`gh` can't resolve this repo on GitHub",
            "create it and push first, e.g. `gh repo create <name> --source=. --remote=origin --push`",
            "or re-run with --no-issue",
        )
    # gh repo create needs at least one commit to push.
    head = git(repo, "rev-parse", "--verify", "HEAD", check=False)
    if head.returncode != 0:
        git(repo, "add", "-A")
        git(repo, "commit", "-m", "seed idea", check=False)
    visibility = ask("Visibility — private or public? [private]:", default="private").lower()
    if visibility not in ("private", "public", "internal"):
        visibility = "private"
    res = run(
        ["gh", "repo", "create", repo.name, f"--{visibility}", "--source=.", "--remote=origin", "--push"],
        cwd=repo,
        check=False,
    )
    if res.returncode != 0:
        die(f"`gh repo create` failed: {res.stderr.strip() or res.stdout.strip()}")
    print(f"  + GitHub repo created ({visibility}) and pushed")
    return True


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
    ap.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="assume yes to all prompts (git init, create README, create GitHub repo)",
    )
    args = ap.parse_args()

    global _ASSUME_YES
    _ASSUME_YES = args.yes

    if not shutil.which("git"):
        die("git is required and was not found on PATH", "install it from https://git-scm.com/")

    repo = Path(args.repo_path).resolve()
    if not repo.is_dir():
        die(f"{repo} is not a directory", "pass the path to an (intended) git repo, or cd into one")
    print(f"target repo: {repo}\n")

    ensure_git_repo(repo)
    readme = ensure_readme(repo)
    make_issue = ensure_gh_ready(repo, want_issue=not args.no_issue)

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

    if make_issue:
        print("ensuring labels…")
        ensure_labels(repo)

    pushed = False
    if args.push:
        print(f"pushing to {args.remote}…")
        push = git(repo, "push", "-u", args.remote, "HEAD", check=False)
        if push.returncode != 0:
            print(f"  warn: push failed: {push.stderr.strip()}")
        else:
            pushed = True

    if not make_issue:
        print("\nDone (scaffold only).")
        if not pushed:
            print(f"  → push your work:   git push -u {args.remote} HEAD")
        print(
            "  → create issue #1:  gh issue create "
            f'--title "Idea: {title}" '
            "--body-file .github/ISSUE_TEMPLATE/idea.md --label aiorchestra --label claude --label story"
        )
        return

    print("creating issue #1…")
    url = create_first_issue(repo, title)
    print(f"\nDone. Issue #1: {url}")
    print("aiorchestra (with the claude-code provider) will pick it up via the `aiorchestra` label.")
    if not pushed:
        print(f"Reminder: you still need to `git push -u {args.remote} HEAD`.")
    print("After you merge issue #1's PR, add `aiorchestra` + `claude` to the `Premise: …` issue the agent files.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        raise SystemExit(130)
