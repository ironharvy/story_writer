#!/usr/bin/env python3
"""Publish a file or directory to here.now with strict, injection-resistant output.

This wrapper performs the create -> upload -> finalize flow itself, then emits a
single JSON object on stdout containing only validated, typed fields. No raw
server-controlled string is ever passed through unmodified:

  - URLs (siteUrl, claimUrl, finalizeUrl) must be https and on the here.now host.
  - slug must match a strict character class.
  - presigned upload URLs must be https; a non-here.now host is reported as a
    sanitized warning rather than silently trusted.
  - server error text is stripped to a safe character set, length-capped, and
    returned inside an explicitly fenced "error_summary" field.

Stdlib only. Requires the `file` command only as a content-type fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERENOW_BASE = "https://here.now"
HERENOW_HOST = "here.now"

SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SAFE_TEXT_RE = re.compile(r"[^A-Za-z0-9 .,:;()/_+-]")
MAX_ERR_LEN = 240

EXTRA_TYPES = {
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".md": "text/plain; charset=utf-8",
    ".svg": "image/svg+xml",
}


def die(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    # msg here is always locally constructed, never server-supplied.
    print(json.dumps({"ok": False, "stage": "client", "error_summary": msg}), file=sys.stdout)
    sys.exit(1)


def sanitize_text(value: Any, *, limit: int = MAX_ERR_LEN) -> str:
    """Reduce arbitrary server text to a short, safe, single-line string."""
    s = "" if value is None else str(value)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = SAFE_TEXT_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        s = s[:limit] + "...(truncated)"
    return s


def host_ok(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme != "https" or not p.hostname:
        return False
    h = p.hostname.lower()
    return h == HERENOW_HOST or h.endswith("." + HERENOW_HOST)


def is_https(url: str) -> bool:
    try:
        from urllib.parse import urlparse

        return urlparse(url).scheme == "https"
    except ValueError:
        return False


def url_host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in EXTRA_TYPES:
        return EXTRA_TYPES[ext]
    guessed, _ = mimetypes.guess_type(str(path))
    if guessed:
        if guessed.startswith(("text/", "application/json", "application/xml")):
            return f"{guessed}; charset=utf-8"
        return guessed
    return "application/octet-stream"


def collect_files(target: Path) -> list[tuple[str, Path]]:
    if target.is_file():
        return [(target.name, target)]
    out: list[tuple[str, Path]] = []
    for p in sorted(target.rglob("*")):
        if not p.is_file():
            continue
        if p.name == ".DS_Store":
            continue
        rel = p.relative_to(target).as_posix()
        out.append((rel, p))
    if not out:
        die("no files found under target directory")
    return out


def http_json(method: str, url: str, *, api_key: str | None, body: dict | None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    except urllib.error.URLError as e:
        die(f"network error: {sanitize_text(getattr(e, 'reason', e), limit=120)}")
    try:
        parsed = json.loads(raw.decode()) if raw else {}
    except (ValueError, UnicodeDecodeError):
        parsed = {"_unparseable": True}
    return status, parsed


def http_put_bytes(url: str, path: Path, content_type_hdr: str | None) -> int:
    with path.open("rb") as f:
        req = urllib.request.Request(url, data=f.read(), method="PUT")
    if content_type_hdr:
        req.add_header("Content-Type", content_type_hdr)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except urllib.error.URLError as e:
        die(f"upload network error: {sanitize_text(getattr(e, 'reason', e), limit=120)}")


def server_error_summary(status: int, parsed: Any) -> str:
    detail = ""
    if isinstance(parsed, dict):
        detail = sanitize_text(parsed.get("error") or parsed.get("message") or "")
        more = sanitize_text(parsed.get("details") or "")
        if more:
            detail = f"{detail} | {more}" if detail else more
    return detail or f"http {status}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish to here.now (injection-resistant wrapper).")
    ap.add_argument("target", help="file or directory to publish")
    ap.add_argument("--slug", help="update an existing publish instead of creating one")
    ap.add_argument("--api-key", help="API key (else $HERENOW_API_KEY or ~/.herenow/credentials)")
    ap.add_argument("--base-url", default=HERENOW_BASE, help=argparse.SUPPRESS)
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    if base != HERENOW_BASE:
        die("refusing to use a non-default base URL")

    target = Path(args.target)
    if not target.exists():
        die("target path does not exist")

    api_key = args.api_key or os.environ.get("HERENOW_API_KEY") or ""
    if not api_key:
        cred = Path.home() / ".herenow" / "credentials"
        if cred.is_file():
            api_key = cred.read_text().strip()
    api_key = api_key.strip() or None

    if args.slug is not None and not SLUG_RE.match(args.slug):
        die("invalid --slug format")

    files = collect_files(target)
    manifest = []
    file_map: dict[str, Path] = {}
    for rel, p in files:
        manifest.append(
            {
                "path": rel,
                "size": p.stat().st_size,
                "contentType": content_type(p),
                "hash": sha256_file(p),
            }
        )
        file_map[rel] = p

    # Step 1: create / update
    if args.slug:
        status, resp = http_json("PUT", f"{base}/api/v1/publish/{args.slug}", api_key=api_key, body={"files": manifest})
    else:
        status, resp = http_json("POST", f"{base}/api/v1/publish", api_key=api_key, body={"files": manifest})

    if status < 200 or status >= 300 or not isinstance(resp, dict) or resp.get("error"):
        print(json.dumps({"ok": False, "stage": "create", "http_status": status,
                          "error_summary": server_error_summary(status, resp)}))
        sys.exit(1)

    slug = resp.get("slug")
    if not isinstance(slug, str) or not SLUG_RE.match(slug):
        die("server returned an unrecognized slug")

    site_url = resp.get("siteUrl")
    if not isinstance(site_url, str) or not host_ok(site_url):
        die("server returned a site URL that is not on here.now")

    upload = resp.get("upload")
    if not isinstance(upload, dict):
        die("server response missing upload section")
    version_id = upload.get("versionId")
    finalize_url = upload.get("finalizeUrl")
    uploads = upload.get("uploads")
    if not isinstance(version_id, str) or not isinstance(uploads, list):
        die("server response missing versionId or uploads")
    if not isinstance(finalize_url, str) or not host_ok(finalize_url):
        die("server returned a finalize URL that is not on here.now")

    # Step 2: upload each file
    upload_host_warnings: set[str] = set()
    for u in uploads:
        if not isinstance(u, dict):
            die("malformed upload entry")
        upath = u.get("path")
        uurl = u.get("url")
        if not isinstance(upath, str) or upath not in file_map:
            die("server asked to upload an unexpected path")
        if not isinstance(uurl, str) or not is_https(uurl):
            die("server returned a non-https upload URL")
        if not host_ok(uurl):
            upload_host_warnings.add(sanitize_text(url_host(uurl), limit=80))
        hdrs = u.get("headers") if isinstance(u.get("headers"), dict) else {}
        ct = hdrs.get("Content-Type") if isinstance(hdrs.get("Content-Type"), str) else None
        code = http_put_bytes(uurl, file_map[upath], ct)
        if code < 200 or code >= 300:
            print(json.dumps({"ok": False, "stage": "upload", "http_status": code,
                              "error_summary": f"upload failed for one file (http {code})"}))
            sys.exit(1)

    # Step 3: finalize
    status, fin = http_json("POST", finalize_url, api_key=api_key, body={"versionId": version_id})
    if status < 200 or status >= 300 or not isinstance(fin, dict) or fin.get("error"):
        print(json.dumps({"ok": False, "stage": "finalize", "http_status": status,
                          "error_summary": server_error_summary(status, fin)}))
        sys.exit(1)

    claim_url = resp.get("claimUrl")
    safe_claim_url = claim_url if isinstance(claim_url, str) and host_ok(claim_url) else None
    expires_at = resp.get("expiresAt")
    safe_expires = sanitize_text(expires_at, limit=40) if isinstance(expires_at, str) else None

    result = {
        "ok": True,
        "site_url": site_url,
        "slug": slug,
        "auth_mode": "authenticated" if api_key else "anonymous",
        "persistence": "permanent" if api_key else "expires_24h",
        "expires_at": safe_expires,
        "claim_url": safe_claim_url,
        "file_count": len(file_map),
    }
    if upload_host_warnings:
        result["warning_upload_hosts"] = sorted(upload_host_warnings)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
