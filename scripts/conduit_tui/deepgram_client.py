"""
deepgram_client.py — DeepgramStream: WebSocket client with word-final char interpolation.

Uses deepgram-sdk >= 3. On is_final transcript events, iterates word list and
interpolates char timestamps linearly. Emits CharEntry list via callback.

Latency target: chars in store within 300ms of is_final callback fire.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

from .char_timeline import CharEntry, interpolate_chars


WordCallback = Callable[[list[CharEntry], str], None]  # (chars, partial_text)


class DeepgramStream:
    """Async Deepgram live transcription session.

    Args:
        api_key: Deepgram API key.
        sample_rate: Audio sample rate (default 16000).
        on_chars: Called with (list[CharEntry], "final") for final words.
        on_partial: Called with (str,) for interim results display.
        session_start: Monotonic time baseline for absolute timestamps.
    """

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
        self._live = None
        self._client: DeepgramClient | None = None

    async def connect(self) -> None:
        config = DeepgramClientOptions(options={"keepalive": "true"})
        self._client = DeepgramClient(self._api_key, config)
        self._live = self._client.listen.asynclive.v("1")

        self._live.on(LiveTranscriptionEvents.Transcript, self._on_transcript)
        self._live.on(LiveTranscriptionEvents.Error, self._on_error)

        options = LiveOptions(
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
        started = await self._live.start(options)
        if not started:
            raise RuntimeError("Deepgram live connection failed to start")

    async def send(self, audio_bytes: bytes) -> None:
        if self._live:
            await self._live.send(audio_bytes)

    async def finish(self) -> None:
        if self._live:
            await self._live.finish()

    async def _on_transcript(self, _client: object, result: object, **kwargs: object) -> None:
        try:
            alt = result.channel.alternatives[0]
            transcript = alt.transcript
            is_final = result.is_final

            if not transcript:
                return

            if not is_final:
                # Interim result — update partial display
                if self._on_partial:
                    self._on_partial(transcript)
                return

            # Final result — interpolate char timestamps from word list
            words = getattr(alt, "words", []) or []
            all_chars: list[CharEntry] = []

            if words:
                for word_obj in words:
                    word_text = word_obj.word
                    # Deepgram gives absolute seconds from audio start;
                    # add session_start offset for absolute monotonic time
                    word_start = self._session_start + float(word_obj.start)
                    word_end = self._session_start + float(word_obj.end)
                    chars = interpolate_chars(
                        word_text, word_start, word_end, "user,interpolated"
                    )
                    all_chars.extend(chars)
            else:
                # No word timestamps — fallback: emit chars with zero-width times
                now = time.monotonic()
                for ch in transcript:
                    all_chars.append(CharEntry(char=ch, start_time=now, end_time=now, notes="user,no-words"))

            if all_chars and self._on_chars:
                self._on_chars(all_chars)

        except Exception as exc:
            # Never let a transcript callback crash the stream
            print(f"[deepgram] transcript handler error: {exc}")

    async def _on_error(self, _client: object, error: object, **kwargs: object) -> None:
        print(f"[deepgram] error: {error}")
