#!/usr/bin/env python3
"""Run the QA detection suite on one or more story markdown files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add repo root so `import qa` works when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qa import run_all  # noqa: E402


SEVERITY_RANK = {"fail": 0, "warn": 1, "info": 2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="story markdown file(s)")
    args = parser.parse_args(argv)

    exit_code = 0
    for path in args.paths:
        if not path.is_file():
            print(f"!! not a file: {path}")
            exit_code = 1
            continue
        print(f"\n=== {path} ===")
        findings = run_all(path)
        findings.sort(key=lambda f: (SEVERITY_RANK.get(f.severity, 9), f.check))
        for f in findings:
            tag = {"fail": "FAIL", "warn": "WARN", "info": "info"}[f.severity]
            print(f"[{tag}] {f.check}: {f.message}")
            if f.severity == "fail":
                exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
