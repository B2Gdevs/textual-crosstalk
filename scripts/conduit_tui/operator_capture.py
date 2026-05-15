"""
operator_capture.py — Persist the operator's voice audio per session.

The speaker classifier caches an embedding at ~/.conduit/voiceprint_user.npy
(see speaker_id.py). That cache is good for the current feature backend
but locks the operator's voice into Tier 0's 29-dim handcrafted features
— if we upgrade to Tier 1 (ONNX embedding) or Tier 2 (distilled CNN),
the cached embedding becomes incompatible and we'd need to re-enroll
from scratch with no historical data.

This module persists the RAW PCM (int16, 16kHz, mono) per session so
each upgrade can re-extract features from the same audio. Stored at:

    data/dataset/user/<session_iso_timestamp>.wav

Use append_user_chunk(chunk_int16) from the orchestrator's mic_pump
hot path; chunks are buffered in memory and flushed to disk on
finalize_session(). A short header phrase from the operator at the
start of every session (~3s) is enough for re-enrollment plus a long
form trail for verification benchmarks.
"""
from __future__ import annotations

import datetime as _dt
import wave
from pathlib import Path

import numpy as np


_DATASET_DIR = Path("data") / "dataset" / "user"
_SAMPLE_RATE = 16000


class OperatorCapture:
    """Buffer + flush raw operator audio. One instance per session."""

    def __init__(
        self, session_label: str | None = None, sample_rate: int = _SAMPLE_RATE
    ) -> None:
        self.sr = int(sample_rate)
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        suffix = f"_{session_label}" if session_label else ""
        self.path = _DATASET_DIR / f"{ts}{suffix}.wav"
        self._buf: list[np.ndarray] = []
        self._sample_count = 0
        self._sealed = False

    def append(self, chunk_int16: np.ndarray) -> None:
        """Append a chunk of AEC-cleaned mic audio. Cheap — just adds
        to a list. Flush happens once via finalize()."""
        if self._sealed or chunk_int16.size == 0:
            return
        self._buf.append(chunk_int16.astype(np.int16, copy=False))
        self._sample_count += int(chunk_int16.size)

    def finalize(self) -> Path | None:
        """Write the session wav. Idempotent — returns None if nothing
        captured. Called once at session end (or at quit)."""
        if self._sealed:
            return self.path if self.path.exists() else None
        self._sealed = True
        if self._sample_count == 0:
            return None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            samples = np.concatenate(self._buf)
            with wave.open(str(self.path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # int16
                wf.setframerate(self.sr)
                wf.writeframes(samples.tobytes())
        except Exception as exc:
            print(f"[operator_capture] write failed: {exc}")
            return None
        return self.path

    @property
    def seconds_captured(self) -> float:
        return self._sample_count / self.sr
