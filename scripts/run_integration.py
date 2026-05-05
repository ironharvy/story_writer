"""Run the end-to-end story pipeline against a real LLM provider.

Reads MODEL, LLM_URL, and the provider-agnostic API key from environment
(see dspy_runtime.resolve_api_key for the lookup order). Intended for CI
integration jobs and local smoke tests.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from test_story import test_pipeline  # noqa: E402


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL"))
    parser.add_argument("--llm-url", default=os.environ.get("LLM_URL"))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--output-dir", default=".tmp")
    args = parser.parse_args()

    if not args.model:
        parser.error("MODEL env var or --model is required")

    test_pipeline(
        model_name=args.model,
        api_base=args.llm_url or None,
        api_key=None,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
