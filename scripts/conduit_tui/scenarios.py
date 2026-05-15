"""
scenarios.py — Test scenario rotation for the conduit conversation TUI.

On every launch, the app picks the next scenario in a rotation so the
operator gets continuous exposure to all the configurations without
having to think about which to test:

    1v1 (one human + one bot — current default behavior)
        → 1v2 (one human + two bots, polite turn-taking)
            → 1v3 (one human + three bots)
                → wrap back to 1v1

State is persisted at ~/.conduit/state.json so the rotation survives
across runs. The CONDUIT_SCENARIO env var overrides ("1v1" | "1v2" |
"1v3") for a single run without advancing the rotation pointer.

Each scenario picks bot voices from a rotating pool — taken from
data/dataset/manifest.json when it exists (the real ElevenLabs
corpus), falling back to a small hardcoded list of premade voice IDs
otherwise.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


_STATE_PATH = Path.home() / ".conduit" / "state.json"
_DATASET_MANIFEST = Path("data") / "dataset" / "manifest.json"


# Fallback voice pool — premade voices available on the ElevenLabs
# free tier. Used when no dataset/manifest is available yet.
DEFAULT_VOICE_POOL: list[tuple[str, str]] = [
    ("EXAVITQu4vr4xnSDxMaL", "Sarah"),
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger"),
    ("FGY2WhTYpPnrIDTdsKH5", "Laura"),
    ("IKne3meq5aSn9XLyUdCD", "Charlie"),
    ("JBFqnCBsd6RMkjVDRZzb", "George"),
]


_ROTATION: list[str] = ["1v1", "1v2", "1v3"]


@dataclass
class Scenario:
    """A single test configuration for a session."""

    id: str                              # "1v1" | "1v2" | "1v3"
    n_bots: int
    bot_voices: list[tuple[str, str]] = field(default_factory=list)  # (voice_id, display_name)

    @property
    def label(self) -> str:
        if self.n_bots == 1:
            return "1 human + 1 bot (solo conversation)"
        return f"1 human + {self.n_bots} bots (meeting mode)"


def _load_voice_pool() -> list[tuple[str, str]]:
    """Read the dataset manifest if present; otherwise fall back."""
    if not _DATASET_MANIFEST.exists():
        return DEFAULT_VOICE_POOL[:]
    try:
        with _DATASET_MANIFEST.open() as fh:
            data = json.load(fh)
    except Exception:
        return DEFAULT_VOICE_POOL[:]

    # Manifest is a list of {voice_id, voice_name, phrase, wav_path}.
    # We just need distinct (voice_id, voice_name) pairs.
    seen: dict[str, str] = {}
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            vid = entry.get("voice_id")
            name = entry.get("voice_name", vid)
            if vid and vid not in seen:
                seen[vid] = name
    elif isinstance(data, dict) and "voices" in data:
        for entry in data["voices"]:
            vid = entry.get("voice_id")
            name = entry.get("voice_name", vid)
            if vid and vid not in seen:
                seen[vid] = name

    pool = [(vid, name) for vid, name in seen.items()]
    return pool if pool else DEFAULT_VOICE_POOL[:]


def _load_state() -> dict:
    if not _STATE_PATH.exists():
        return {}
    try:
        with _STATE_PATH.open() as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _STATE_PATH.open("w") as fh:
            json.dump(state, fh, indent=2)
    except Exception as exc:
        print(f"[scenarios] state save failed: {exc}")


def next_scenario(advance: bool = True) -> Scenario:
    """Return the next scenario in rotation. If CONDUIT_SCENARIO is set,
    use that as a one-shot override without advancing the pointer."""
    override = os.environ.get("CONDUIT_SCENARIO", "").strip()
    if override in _ROTATION:
        scenario_id = override
        advance = False
    else:
        state = _load_state()
        idx = int(state.get("scenario_index", 0)) % len(_ROTATION)
        scenario_id = _ROTATION[idx]
        if advance:
            state["scenario_index"] = (idx + 1) % len(_ROTATION)
            _save_state(state)

    n_bots = {"1v1": 1, "1v2": 2, "1v3": 3}[scenario_id]
    pool = _load_voice_pool()
    if len(pool) < n_bots:
        # Cycle the pool if we don't have enough distinct voices
        bot_voices = [(pool[i % len(pool)]) for i in range(n_bots)]
    else:
        bot_voices = pool[:n_bots]
    return Scenario(id=scenario_id, n_bots=n_bots, bot_voices=bot_voices)


def reset_rotation() -> None:
    """Reset the rotation pointer back to scenario 0 (1v1)."""
    state = _load_state()
    state["scenario_index"] = 0
    _save_state(state)
