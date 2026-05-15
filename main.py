"""
main.py — Conduit entry point.

Loads .env.local, then runs ConduitApp.

Usage:
    python main.py
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    # Load .env.local from project root (secrets — gitignored)
    env_path = Path(__file__).parent / ".env.local"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Fallback to .env.example for structure reference only
        load_dotenv(Path(__file__).parent / ".env.example")

    # Ensure data directory exists
    char_log = os.environ.get("CONDUIT_CHAR_LOG", "./data/chars.jsonl")
    Path(char_log).parent.mkdir(parents=True, exist_ok=True)

    # Late import so dotenv is loaded before any client init
    from scripts.conduit_tui.app import ConduitApp

    app = ConduitApp()
    app.run()


if __name__ == "__main__":
    main()
