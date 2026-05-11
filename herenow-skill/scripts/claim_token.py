#!/usr/bin/env python3
"""Obtain a here.now API key and write it to .env.publish (chmod 600).

Two-step flow:
  1. request a one-time sign-in code be emailed to you
  2. paste that code back here; the verified API key is written to .env.publish
     as HERENOW_API_KEY=...

The key is never printed. Load it for publishing with:

    set -a; source .env.publish; set +a
    ./scripts/publish_content.py <file-or-dir>

Stdlib only. Pinned to https://here.now.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERENOW_BASE = "https://here.now"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
KEY_NAME = "HERENOW_API_KEY"


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def post_json(url: str, body: dict) -> tuple[int, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            return e.code, {}
    except urllib.error.URLError as e:
        fail(f"network error: {getattr(e, 'reason', e)}")
    except (ValueError, UnicodeDecodeError):
        fail("server returned a non-JSON response")


def server_msg(parsed: Any, status: int) -> str:
    if isinstance(parsed, dict):
        m = parsed.get("error") or parsed.get("message")
        if isinstance(m, str) and m:
            # only used in our own error: line, not handed to an agent as output
            return re.sub(r"\s+", " ", m)[:200]
    return f"http {status}"


def in_git_repo(path: Path) -> Path | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return Path(out.stdout.strip())


def ensure_gitignored(env_path: Path) -> None:
    repo = in_git_repo(env_path.parent)
    if repo is None:
        return
    gi = repo / ".gitignore"
    name = env_path.name
    existing = gi.read_text().splitlines() if gi.exists() else []
    if any(line.strip() in (name, f"/{name}", f"{name}/") for line in existing):
        return
    with gi.open("a") as f:
        if existing and existing[-1].strip():
            f.write("\n")
        f.write(f"{name}\n")
    print(f"added {name} to {gi}", file=sys.stderr)


def write_env(env_path: Path, key: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not re.match(rf"\s*(export\s+)?{re.escape(KEY_NAME)}\s*=", line):
                lines.append(line)
    lines.append(f"{KEY_NAME}={key}")
    fd = os.open(str(env_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(env_path, stat.S_IRUSR | stat.S_IWUSR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Get a here.now API key into .env.publish.")
    ap.add_argument("--email", help="email address to send the sign-in code to (prompted if omitted)")
    ap.add_argument("--env-file", default=".env.publish", help="where to write HERENOW_API_KEY (default: .env.publish)")
    ap.add_argument("--code", help="sign-in code (prompted if omitted)")
    ap.add_argument("--base-url", default=HERENOW_BASE, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.base_url.rstrip("/") != HERENOW_BASE:
        fail("refusing to use a non-default base URL")

    email = args.email or input("here.now email: ").strip()
    if not EMAIL_RE.match(email):
        fail("that does not look like an email address")

    status, resp = post_json(f"{HERENOW_BASE}/api/auth/agent/request-code", {"email": email})
    if status < 200 or status >= 300 or (isinstance(resp, dict) and resp.get("error")):
        fail(f"request-code failed: {server_msg(resp, status)}")
    print(f"a sign-in code was emailed to {email}", file=sys.stderr)

    code = args.code or getpass.getpass("paste the sign-in code: ").strip()
    if not code:
        fail("no code provided")

    status, resp = post_json(
        f"{HERENOW_BASE}/api/auth/agent/verify-code", {"email": email, "code": code}
    )
    if status < 200 or status >= 300 or not isinstance(resp, dict) or resp.get("error"):
        fail(f"verify-code failed: {server_msg(resp, status)}")

    api_key = resp.get("apiKey") or resp.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        fail("server response did not contain an apiKey")
    api_key = api_key.strip()

    env_path = Path(args.env_file)
    ensure_gitignored(env_path)
    write_env(env_path, api_key)

    # Deliberately do NOT print the key.
    print(json.dumps({
        "ok": True,
        "stored_in": str(env_path),
        "var": KEY_NAME,
        "mode": "600",
        "next": f"set -a; source {env_path}; set +a   then run ./scripts/publish_content.py",
    }))


if __name__ == "__main__":
    main()
