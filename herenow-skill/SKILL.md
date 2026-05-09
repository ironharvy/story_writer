---
name: herenow
description: >
  Publish a file or directory to here.now and get back a live URL.
  Use when the user asks to "publish this", "host this", "share this on the
  web", "deploy this", "put this online", or "make a webpage" out of local
  files. Anonymous publishes expire in 24 hours; with HERENOW_API_KEY they
  are permanent.
---

# herenow (minimal)

Thin wrapper around the `here.now` publish API.

## Usage

```bash
./scripts/publish.sh <file-or-dir> [--slug <slug>] [--api-key <key>]
```

- No key → anonymous, 24h expiry. The script prints a `claim_url` once; capture it or the site is unrecoverable.
- With key (env `HERENOW_API_KEY`, `~/.herenow/credentials`, or `--api-key`) → permanent.
- `--slug` updates an existing publish instead of creating a new one.

The script prints the live URL on stdout and `publish_result.*` diagnostics on stderr.

## Getting an API key

```bash
curl -sS https://here.now/api/auth/agent/request-code \
  -H 'content-type: application/json' \
  -d '{"email":"you@example.com"}'

# check inbox for code, then:
curl -sS https://here.now/api/auth/agent/verify-code \
  -H 'content-type: application/json' \
  -d '{"email":"you@example.com","code":"ABCD-2345"}'
```

Save the returned `apiKey` to `~/.herenow/credentials` (`chmod 600`) or export `HERENOW_API_KEY`.

## Requirements

`bash`, `curl`, `jq`, `file`, `sha256sum` (or `shasum`).
