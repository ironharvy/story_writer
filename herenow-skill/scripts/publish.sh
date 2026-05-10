#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://here.now"
CREDENTIALS_FILE="$HOME/.herenow/credentials"
API_KEY="${HERENOW_API_KEY:-}"
SLUG=""
TARGET=""

usage() {
  cat <<'USAGE' >&2
Usage: publish.sh <file-or-dir> [--slug <slug>] [--api-key <key>]
USAGE
  exit 1
}

die() { echo "error: $1" >&2; exit 1; }

for cmd in curl jq file; do
  command -v "$cmd" >/dev/null 2>&1 || die "requires $cmd"
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-key) API_KEY="$2"; shift 2 ;;
    --slug)    SLUG="$2"; shift 2 ;;
    -h|--help) usage ;;
    -*)        die "unknown option: $1" ;;
    *)         [[ -z "$TARGET" ]] && TARGET="$1" || die "unexpected arg: $1"; shift ;;
  esac
done

[[ -n "$TARGET" ]] || usage
[[ -e "$TARGET" ]] || die "no such path: $TARGET"

if [[ -z "$API_KEY" && -f "$CREDENTIALS_FILE" ]]; then
  API_KEY=$(tr -d '[:space:]' < "$CREDENTIALS_FILE")
fi

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

content_type() {
  case "${1##*.}" in
    html|htm) echo "text/html; charset=utf-8" ;;
    css)      echo "text/css; charset=utf-8" ;;
    js|mjs)   echo "text/javascript; charset=utf-8" ;;
    json)     echo "application/json; charset=utf-8" ;;
    md|txt)   echo "text/plain; charset=utf-8" ;;
    svg)      echo "image/svg+xml" ;;
    png)      echo "image/png" ;;
    jpg|jpeg) echo "image/jpeg" ;;
    pdf)      echo "application/pdf" ;;
    *)        file --brief --mime-type "$1" 2>/dev/null || echo "application/octet-stream" ;;
  esac
}

FILES_JSON="[]"
declare -A FILE_MAP

add_file() {
  local rel="$1" abs="$2"
  local sz ct h
  sz=$(wc -c < "$abs" | tr -d ' ')
  ct=$(content_type "$abs")
  h=$(sha256 "$abs")
  FILES_JSON=$(jq --arg p "$rel" --argjson s "$sz" --arg c "$ct" --arg h "$h" \
    '. + [{path:$p,size:$s,contentType:$c,hash:$h}]' <<<"$FILES_JSON")
  FILE_MAP["$rel"]="$abs"
}

if [[ -f "$TARGET" ]]; then
  add_file "$(basename "$TARGET")" "$(cd "$(dirname "$TARGET")" && pwd)/$(basename "$TARGET")"
else
  while IFS= read -r -d '' f; do
    rel="${f#$TARGET/}"
    [[ "$(basename "$rel")" == ".DS_Store" ]] && continue
    add_file "$rel" "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"
  done < <(find "$TARGET" -type f -print0 | sort -z)
fi

[[ "$(jq length <<<"$FILES_JSON")" -gt 0 ]] || die "no files to publish"

BODY=$(jq '{files: .}' <<<"$FILES_JSON")

if [[ -n "$SLUG" ]]; then
  URL="$BASE_URL/api/v1/publish/$SLUG"; METHOD="PUT"
else
  URL="$BASE_URL/api/v1/publish"; METHOD="POST"
fi

AUTH=()
[[ -n "$API_KEY" ]] && AUTH=(-H "authorization: Bearer $API_KEY")

echo "creating publish ($(jq length <<<"$FILES_JSON") files)..." >&2
RESP=$(curl -sS -X "$METHOD" "$URL" \
  "${AUTH[@]+"${AUTH[@]}"}" \
  -H "content-type: application/json" \
  -d "$BODY")

if jq -e '.error' >/dev/null 2>&1 <<<"$RESP"; then
  die "$(jq -r '.error' <<<"$RESP")"
fi

VERSION_ID=$(jq -r '.upload.versionId' <<<"$RESP")
FINALIZE_URL=$(jq -r '.upload.finalizeUrl' <<<"$RESP")
SITE_URL=$(jq -r '.siteUrl' <<<"$RESP")
OUT_SLUG=$(jq -r '.slug' <<<"$RESP")
UPLOADS=$(jq '.upload.uploads | length' <<<"$RESP")

echo "uploading $UPLOADS file(s)..." >&2
for i in $(seq 0 $((UPLOADS - 1))); do
  p=$(jq -r ".upload.uploads[$i].path" <<<"$RESP")
  u=$(jq -r ".upload.uploads[$i].url" <<<"$RESP")
  ct=$(jq -r ".upload.uploads[$i].headers[\"Content-Type\"] // empty" <<<"$RESP")
  local_file="${FILE_MAP[$p]}"
  ct_args=(); [[ -n "$ct" ]] && ct_args=(-H "Content-Type: $ct")
  code=$(curl -sS -o /dev/null -w '%{http_code}' -X PUT "$u" \
    "${ct_args[@]+"${ct_args[@]}"}" --data-binary "@$local_file")
  [[ "$code" -ge 200 && "$code" -lt 300 ]] || die "upload failed for $p (HTTP $code)"
done

echo "finalizing..." >&2
FIN=$(curl -sS -X POST "$FINALIZE_URL" \
  "${AUTH[@]+"${AUTH[@]}"}" \
  -H "content-type: application/json" \
  -d "{\"versionId\":\"$VERSION_ID\"}")
if jq -e '.error' >/dev/null 2>&1 <<<"$FIN"; then
  die "finalize failed: $(jq -r '.error' <<<"$FIN")"
fi

CLAIM_URL=$(jq -r '.claimUrl // empty' <<<"$RESP")
EXPIRES=$(jq -r '.expiresAt // empty' <<<"$RESP")
AUTH_MODE="anonymous"; [[ -n "$API_KEY" ]] && AUTH_MODE="authenticated"

echo "$SITE_URL"
{
  echo "publish_result.site_url=$SITE_URL"
  echo "publish_result.slug=$OUT_SLUG"
  echo "publish_result.auth_mode=$AUTH_MODE"
  echo "publish_result.expires_at=$EXPIRES"
  [[ "$CLAIM_URL" == https://* ]] && echo "publish_result.claim_url=$CLAIM_URL"
} >&2
