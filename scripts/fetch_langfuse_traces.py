#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _build_auth_header(public_key: str, secret_key: str) -> str:
    token = f"{public_key}:{secret_key}".encode("utf-8")
    return f"Basic {base64.b64encode(token).decode('ascii')}"


def _fetch_json(url: str, auth_header: str, timeout: int) -> dict:
    req = Request(url)
    req.add_header("Authorization", auth_header)
    req.add_header("Accept", "application/json")

    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _read_list_response(payload: dict) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return data
    return []


def _parse_iso8601(ts: str | None) -> datetime | None:
    if not ts or not isinstance(ts, str):
        return None
    raw = ts.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _text_len(value) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except TypeError:
        return len(str(value))


def _extract_error_hints(obj) -> list[str]:
    hints: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_l = str(key).lower()
            if key_l in {"error", "errormessage", "statusmessage"} and value:
                hints.append(f"{key}: {value}")
            elif key_l == "level" and isinstance(value, str) and value.lower() in {"error", "fatal"}:
                hints.append(f"level: {value}")
            else:
                hints.extend(_extract_error_hints(value))
    elif isinstance(obj, list):
        for item in obj:
            hints.extend(_extract_error_hints(item))
    return hints


def _summarize(input_path: Path, output_path: Path | None, since: datetime | None, until: datetime | None, limit: int | None) -> int:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Failed to parse input JSON: {exc}", file=sys.stderr)
        return 2

    details = payload.get("details") if isinstance(payload, dict) else None
    if not isinstance(details, list):
        print("Input JSON has no `details` list. Re-run fetch with --include-details.", file=sys.stderr)
        return 2

    summaries: list[dict] = []
    for trace in details:
        if not isinstance(trace, dict):
            continue

        ts = _parse_iso8601(trace.get("timestamp"))
        if since and (ts is None or ts < since):
            continue
        if until and (ts is None or ts > until):
            continue

        observations = trace.get("observations") if isinstance(trace.get("observations"), list) else []
        obs_items = []
        for obs in observations:
            if not isinstance(obs, dict):
                continue
            obs_items.append(
                {
                    "id": obs.get("id"),
                    "type": obs.get("type"),
                    "name": obs.get("name"),
                    "model": obs.get("model") or obs.get("modelId"),
                    "level": obs.get("level"),
                    "latency_ms": obs.get("latency"),
                    "input_chars": _text_len(obs.get("input")),
                    "output_chars": _text_len(obs.get("output")),
                    "error_hints": sorted(set(_extract_error_hints(obs))),
                }
            )

        summaries.append(
            {
                "trace_id": trace.get("id"),
                "timestamp": trace.get("timestamp"),
                "name": trace.get("name"),
                "latency_ms": trace.get("latency"),
                "total_cost": trace.get("totalCost"),
                "input_chars": _text_len(trace.get("input")),
                "output_chars": _text_len(trace.get("output")),
                "error_hints": sorted(set(_extract_error_hints(trace))),
                "observation_count": len(obs_items),
                "observations": obs_items,
                "html_path": trace.get("htmlPath"),
            }
        )

    summaries.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
    if limit is not None and limit > 0:
        summaries = summaries[:limit]

    print(f"Summarized {len(summaries)} traces from {input_path}")
    for item in summaries:
        print(
            f"- {item['timestamp']} {item['trace_id']} obs={item['observation_count']} "
            f"in={item['input_chars']} out={item['output_chars']} errors={len(item['error_hints'])}"
        )
        for obs in item["observations"]:
            model = obs["model"] or "-"
            print(
                f"    · {obs['type']}/{obs['name']} model={model} "
                f"in={obs['input_chars']} out={obs['output_chars']} errors={len(obs['error_hints'])}"
            )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "generated_at": _iso_utc(datetime.now(timezone.utc)),
                    "source": str(input_path),
                    "since": _iso_utc(since) if since else None,
                    "until": _iso_utc(until) if until else None,
                    "trace_count": len(summaries),
                    "traces": summaries,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote summary JSON -> {output_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch or summarize Langfuse traces.")
    parser.add_argument("--mode", choices=["fetch", "summarize"], default="fetch", help="Operation mode")
    parser.add_argument("--host", default=None, help="Langfuse host (e.g. https://cloud.langfuse.com)")
    parser.add_argument("--public-key", default=None, help="Langfuse public key")
    parser.add_argument("--secret-key", default=None, help="Langfuse secret key")
    parser.add_argument("--limit", type=int, default=50, help="Number of traces to fetch")
    parser.add_argument("--hours", type=int, default=24, help="Fetch traces from the last N hours")
    parser.add_argument("--name", default=None, help="Optional trace name filter")
    parser.add_argument("--user-id", default=None, help="Optional user id filter")
    parser.add_argument("--include-details", action="store_true", help="Fetch per-trace detail payloads")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--output", default=".tmp/langfuse_traces.json", help="Path to output JSON file")
    parser.add_argument("--input", default=".tmp/langfuse_traces.json", help="Input JSON path for summarize mode")
    parser.add_argument("--since", default=None, help="Summarize only traces at/after this ISO timestamp")
    parser.add_argument("--until", default=None, help="Summarize only traces at/before this ISO timestamp")
    parser.add_argument("--summary-hours", type=int, default=None, help="Summarize only traces from the last N hours")
    parser.add_argument("--summary-limit", type=int, default=None, help="Max summarized traces to include")
    args = parser.parse_args()

    load_dotenv()

    if args.mode == "summarize":
        since = _parse_iso8601(args.since) if args.since else None
        until = _parse_iso8601(args.until) if args.until else None
        if args.summary_hours is not None:
            since = datetime.now(timezone.utc) - timedelta(hours=max(args.summary_hours, 1))
        summary_output = Path(args.output) if args.output else None
        return _summarize(
            input_path=Path(args.input),
            output_path=summary_output,
            since=since,
            until=until,
            limit=args.summary_limit,
        )

    host = (
        args.host
        or os.environ.get("LANGFUSE_HOST")
        or os.environ.get("LANGFUSE_BASE_URL")
        or ""
    ).rstrip("/")
    public_key = args.public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = args.secret_key or os.environ.get("LANGFUSE_SECRET_KEY")

    missing = [
        name
        for name, value in (
            ("LANGFUSE_HOST", host),
            ("LANGFUSE_PUBLIC_KEY", public_key),
            ("LANGFUSE_SECRET_KEY", secret_key),
        )
        if not value
    ]
    if missing:
        print(f"Missing required config: {', '.join(missing)}", file=sys.stderr)
        return 2

    start = datetime.now(timezone.utc) - timedelta(hours=max(args.hours, 1))
    params = {
        "limit": max(args.limit, 1),
        "fromTimestamp": _iso_utc(start),
    }
    if args.name:
        params["name"] = args.name
    if args.user_id:
        params["userId"] = args.user_id

    list_url = f"{host}/api/public/traces?{urlencode(params)}"
    auth_header = _build_auth_header(public_key, secret_key)

    try:
        traces_payload = _fetch_json(list_url, auth_header, timeout=args.timeout)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        print(f"Langfuse API error ({exc.code}): {body}", file=sys.stderr)
        return 1
    except URLError as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"Failed to decode Langfuse response: {exc}", file=sys.stderr)
        return 1

    traces = _read_list_response(traces_payload)
    details: list[dict] = []

    if args.include_details:
        for trace in traces:
            trace_id = trace.get("id") or trace.get("traceId")
            if not trace_id:
                continue
            detail_url = f"{host}/api/public/traces/{trace_id}"
            try:
                details.append(_fetch_json(detail_url, auth_header, timeout=args.timeout))
            except Exception as exc:
                details.append({"trace_id": trace_id, "error": str(exc)})

    output = {
        "fetched_at": _iso_utc(datetime.now(timezone.utc)),
        "host": host,
        "query": params,
        "trace_count": len(traces),
        "traces": traces,
    }
    if args.include_details:
        output["details"] = details

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Fetched {len(traces)} traces -> {output_path}")
    if not traces:
        print("No traces matched filters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
