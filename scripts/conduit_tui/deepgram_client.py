"""
deepgram_client.py — DeepgramStream: WebSocket client with word-final char interpolation.

Uses deepgram-sdk >= 7. Streaming API:
  client.listen.v1.connect(...)  → AsyncIterator[AsyncV1SocketClient]
  socket.on(EventType.MESSAGE, handler)
  socket.send_media(bytes)
  socket.start_listening() — recv loop

Messages parsed into typed responses; we dispatch on `.type`:
  - "Results" → ListenV1Results with channel.alternatives[0].words
  - "UtteranceEnd"
  - "SpeechStarted"
  - "Metadata"

Latency target: chars in store within 300ms of is_final callback fire.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

from .char_timeline import CharEntry, interpolate_chars


WordCallback = Callable[[list[CharEntry], str], None]


class DeepgramStream:
    """Async Deepgram live transcription session (SDK v7+)."""

    def __init__(
        self,
        api_key: str,
        sample_rate: int = 16000,
        on_chars: Callable[[list[CharEntry]], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
        session_start: float | None = None,
    ) -> None:
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._on_chars = on_chars
        self._on_partial = on_partial
        self._session_start = session_start or time.monotonic()
        self._client: AsyncDeepgramClient | None = None
        self._cm = None
        self._socket = None
        self._listen_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._client = AsyncDeepgramClient(api_key=self._api_key)
        self._cm = self._client.listen.v1.connect(
            model="nova-3",
            language="en-US",
            encoding="linear16",
            sample_rate=self._sample_rate,
            channels=1,
            interim_results=True,
            punctuate=True,
            smart_format=True,
            utterance_end_ms=1500,
        )
        self._socket = await self._cm.__aenter__()
        self._socket.on(EventType.MESSAGE, self._on_message)
        self._socket.on(EventType.ERROR, self._on_error)
        self._listen_task = asyncio.create_task(self._socket.start_listening())

    async def send(self, audio_bytes: bytes) -> None:
        if self._socket:
            await self._socket.send_media(audio_bytes)

    async def finish(self) -> None:
        try:
            if self._socket:
                await self._socket.send_close_stream()
        except Exception:
            pass
        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _on_message(self, parsed: object) -> None:
        try:
            msg_type = getattr(parsed, "type", None)
            if msg_type != "Results":
                return

            channel = getattr(parsed, "channel", None)
            if not channel:
                return
            alts = getattr(channel, "alternatives", None) or []
            if not alts:
                return
            alt = alts[0]
            transcript = getattr(alt, "transcript", "") or ""
            is_final = bool(getattr(parsed, "is_final", False))

            if not transcript:
                return

            if not is_final:
                if self._on_partial:
                    self._on_partial(transcript)
                return

            words = getattr(alt, "words", []) or []
            all_chars: list[CharEntry] = []

            if words:
                for word_obj in words:
                    word_text = getattr(word_obj, "word", "") or getattr(word_obj, "punctuated_word", "")
                    word_start = self._session_start + float(getattr(word_obj, "start", 0.0))
                    word_end = self._session_start + float(getattr(word_obj, "end", word_start))
                    chars = interpolate_chars(
                        word_text, word_start, word_end, "user,interpolated"
                    )
                    all_chars.extend(chars)
            else:
                now = time.monotonic()
                for ch in transcript:
                    all_chars.append(
                        CharEntry(char=ch, start_time=now, end_time=now, notes="user,no-words")
                    )

            if all_chars and self._on_chars:
                self._on_chars(all_chars)

        except Exception as exc:
            print(f"[deepgram] transcript handler error: {exc}")

    async def _on_error(self, error: object) -> None:
        print(f"[deepgram] error: {error}")
