from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest


def test_ollama_is_reachable_when_available() -> None:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError):
        pytest.skip("Ollama is not reachable")

    assert "models" in payload
