"""
mic_capture.py — MicStream: sounddevice InputStream → asyncio bytes queue

Pattern: sounddevice InputStream callback threading adapted from
slm_learning local_speech.py (Thread/Queue pattern → asyncio.Queue).
"""
from __future__ import annotations

import asyncio
import queue
import threading

import numpy as np
import sounddevice as sd


class MicStream:
    """Capture mono 16-bit PCM from default input device.

    Usage:
        async with MicStream(sample_rate=16000) as mic:
            while True:
                chunk: bytes = await mic.read()
    """

    BLOCK_SIZE = 1024  # samples per callback

    def __init__(self, sample_rate: int = 16000) -> None:
        self._sample_rate = sample_rate
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # Level tracking (0.0..1.0 RMS)
        self._level: float = 0.0

    @property
    def level(self) -> float:
        return self._level

    async def __aenter__(self) -> "MicStream":
        self._loop = asyncio.get_running_loop()
        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.BLOCK_SIZE,
            callback=self._callback,
        )
        self._stream.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def read(self) -> bytes:
        return await self._queue.get()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        raw = indata.copy()
        # RMS level for status bar
        rms = float(np.sqrt(np.mean(raw.astype(np.float32) ** 2)))
        self._level = min(1.0, rms / 8000.0)

        chunk = raw.tobytes()
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)
