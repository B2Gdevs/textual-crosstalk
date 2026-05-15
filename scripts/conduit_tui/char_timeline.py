"""
char_timeline.py — CharEntry dataclass + JsonlStore (append-only, buffered)

Data format: one JSON line per character. Fields:
  char       — single character
  start_time — float seconds since session monotonic start
  end_time   — float seconds since session monotonic start
  notes      — comma-separated tags, e.g. "user,interpolated,word=hello"
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CharEntry:
    char: str
    start_time: float
    end_time: float
    notes: str


class JsonlStore:
    """Append-only JSONL writer with async flush batching.

    Flushes when buffer hits FLUSH_SIZE entries OR FLUSH_INTERVAL seconds
    elapses, whichever comes first.
    Pattern: buffer+timer pattern from local_speech.py (slm_learning) adapted
    for async and JSONL.
    """

    FLUSH_SIZE = 20
    FLUSH_INTERVAL = 0.10  # 100ms

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[CharEntry] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public API

    async def append(self, entry: CharEntry) -> None:
        async with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self.FLUSH_SIZE:
                await self._flush_unlocked()
            else:
                self._schedule_flush()

    async def flush(self) -> None:
        async with self._lock:
            await self._flush_unlocked()

    async def close(self) -> None:
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        await self.flush()

    # ------------------------------------------------------------------
    # Internal

    def _schedule_flush(self) -> None:
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._flush_after_delay())

    async def _flush_after_delay(self) -> None:
        await asyncio.sleep(self.FLUSH_INTERVAL)
        async with self._lock:
            await self._flush_unlocked()

    async def _flush_unlocked(self) -> None:
        if not self._buffer:
            return
        lines = [json.dumps(asdict(e)) for e in self._buffer]
        self._buffer.clear()
        # Write in executor so we never block the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_lines, lines)

    def _write_lines(self, lines: list[str]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def interpolate_chars(
    word: str,
    word_start: float,
    word_end: float,
    base_notes: str,
) -> list[CharEntry]:
    """Linear interpolation of char timestamps within a word.

    "hello" spanning 0.0..0.5 → 5 entries, each 0.1s wide.
    Returns empty list for empty word.
    """
    chars = list(word)
    n = len(chars)
    if n == 0:
        return []
    duration = word_end - word_start
    entries: list[CharEntry] = []
    for i, ch in enumerate(chars):
        char_start = word_start + (i / n) * duration
        char_end = word_start + ((i + 1) / n) * duration
        notes = f"{base_notes},word={word}"
        entries.append(CharEntry(char=ch, start_time=char_start, end_time=char_end, notes=notes))
    return entries
