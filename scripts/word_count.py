#!/usr/bin/env python3
"""Count the number of words in a file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def count_words(file_path: Path) -> int:
    """Read the file at *file_path* and return the total word count."""
    text = file_path.read_text(encoding="utf-8")
    return len(text.split())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Count the number of words in a file.")
    parser.add_argument("file", type=Path, help="Path to the file to count words in.")
    args = parser.parse_args(argv)

    path: Path = args.file
    if not path.is_file():
        print(f"Error: '{path}' is not a file or does not exist.", file=sys.stderr)
        sys.exit(1)

    words = count_words(path)
    print(f"{words} {path}")


if __name__ == "__main__":
    main()
