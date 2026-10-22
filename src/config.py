"""Configuration loading: config.yaml plus API keys from .env.

Keys are validated lazily, only by the module that needs them:
``require_env("ENTREZ_EMAIL")`` in collect, ``require_env("ANTHROPIC_API_KEY")`` in extract_llm.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load config.yaml into a dict and populate os.environ from .env."""
    load_dotenv(ROOT / ".env")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_env(name: str) -> str:
    """Return a required environment variable or exit with a clear message."""
    load_dotenv(ROOT / ".env")
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required setting {name}. Copy .env.example to .env and set it.")
    return value


def optional_env(name: str) -> str | None:
    """Return an optional environment variable, or None if unset/empty."""
    load_dotenv(ROOT / ".env")
    return os.environ.get(name, "").strip() or None


if __name__ == "__main__":
    print(load_config())
