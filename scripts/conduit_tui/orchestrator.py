"""
orchestrator.py — ConversationLoop: wires STT → LLM → TTS, manages turn state.

Turn flow (Part 2 — Crosstalk):
  1. Each is_final word from Deepgram feeds Crosstalk.on_word_final()
  2. Crosstalk speculatively fires LLM after SPECULATIVE_THRESHOLD_MS silence
  3. If user keeps talking, in-flight LLM task is cancelled (asyncio.Task.cancel)
  4. When SETTLED_THRESHOLD_MS passes with no new words, response commits → TTS

All char entries (user STT + bot TTS) go to JsonlStore.
UI callbacks update StatusBar + transcript logs.

Env vars:
  CROSSTALK_SPECULATIVE_THRESHOLD_MS  (default 250)
  CROSSTALK_SETTLED_THRESHOLD_MS      (default 600)
  CROSSTALK_MIN_WORDS                 (default 3)
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass


from .char_timeline import CharEntry, JsonlStore
from .crosstalk import Crosstalk
from .deepgram_client import DeepgramStream
from .llm_client import LLMClient
from .mic_capture import MicStream
from .tts_client import ElevenLabsClient


@dataclass
class TurnResult:
    user_text: str
    bot_text: str
    llm_latency: float
    provider: str
    user_chars: int
    bot_chars: int


class ConversationLoop:
    """Async conversation orchestrator.

    All UI callbacks are called from the asyncio thread and should be
    non-blocking (schedule Textual updates via app.call_from_thread if needed).
    """

    def __init__(
        self,
        store: JsonlStore,
        dg_stream: DeepgramStream,
        mic: MicStream,
        llm: LLMClient,
        tts: ElevenLabsClient,
        session_start: float,
        on_user_partial: Callable[[str], None] | None = None,
        on_user_final: Callable[[str], None] | None = None,
        on_bot_response: Callable[[str, float, str], None] | None = None,
        on_chars: Callable[[list[CharEntry]], None] | None = None,
        on_status: Callable[[str, str], None] | None = None,  # (component, state)
        on_mic_level: Callable[[float], None] | None = None,
    ) -> None:
        self._store = store
        self._dg = dg_stream
        self._mic = mic
        self._llm = llm
        self._tts = tts
        self._session_start = session_start

        self._on_user_partial = on_user_partial
        self._on_user_final = on_user_final
        self._on_bot_response = on_bot_response
        self._on_chars = on_chars
        self._on_status = on_status
        self._on_mic_level = on_mic_level

        self._history: list[dict[str, str]] = []
        self._running = False
        self._stop_event = asyncio.Event()

        # Turn accumulation (fed to Crosstalk)
        self._pending_finals: list[str] = []

        # Crosstalk coordinator — replaces dumb 1.5s silence wait
        self._crosstalk = Crosstalk(
            llm_client=self._llm,
            on_response_ready=self._on_crosstalk_response,
        )

        # Set callbacks on deepgram stream
        self._dg._on_chars = self._handle_dg_chars
        self._dg._on_partial = self._handle_dg_partial

    # ------------------------------------------------------------------
    # Entry point

    async def run(self) -> None:
        self._running = True
        self._stop_event.clear()

        mic_task = asyncio.create_task(self._mic_pump())
        level_task = asyncio.create_task(self._level_pump())

        try:
            await asyncio.wait(
                [mic_task, level_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
        finally:
            self._running = False
            mic_task.cancel()
            level_task.cancel()
            await self._crosstalk.cancel()
            await self._dg.finish()
            await self._store.close()

    async def stop(self) -> None:
        self._running = False
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Mic → Deepgram pump

    async def _mic_pump(self) -> None:
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._mic.read(), timeout=0.5)
                await self._dg.send(chunk)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[orchestrator] mic pump error: {exc}")

    async def _level_pump(self) -> None:
        while self._running:
            if self._on_mic_level:
                self._on_mic_level(self._mic.level)
            await asyncio.sleep(0.1)

    # ------------------------------------------------------------------
    # Deepgram callbacks

    def _handle_dg_chars(self, chars: list[CharEntry]) -> None:
        asyncio.create_task(self._store_and_emit_chars(chars))

    def _handle_dg_partial(self, text: str) -> None:
        if self._on_user_partial:
            self._on_user_partial(text)
        if self._on_status:
            self._on_status("stt", "live")

    async def _store_and_emit_chars(self, chars: list[CharEntry]) -> None:
        for ch in chars:
            await self._store.append(ch)
        if self._on_chars:
            self._on_chars(chars)

        # Split chars into distinct words using the "word=X" tag in notes.
        # Deepgram may bundle several words into one is_final batch; without
        # this split they'd be concatenated into a single token and the
        # MIN_WORDS_FOR_SPECULATION gate in Crosstalk would never trip.
        words_in_batch: list[str] = []
        current_chars: list[str] = []
        current_tag: str | None = None
        for c in chars:
            tag = next(
                (p for p in c.notes.split(",") if p.startswith("word=")),
                None,
            )
            if tag != current_tag and current_chars:
                words_in_batch.append("".join(current_chars))
                current_chars = []
            current_tag = tag
            if c.char.strip():
                current_chars.append(c.char)
        if current_chars:
            words_in_batch.append("".join(current_chars))

        if not words_in_batch:
            return

        for w in words_in_batch:
            self._pending_finals.append(w)

        full = " ".join(self._pending_finals)
        word_count = len(self._pending_finals)

        if self._on_user_final:
            self._on_user_final(full)
        if self._on_status:
            self._on_status("stt", "final")

        # Keep crosstalk history in sync, then notify
        self._crosstalk.set_history(self._history)
        if self._on_status:
            self._on_status("llm", "speculating")
        asyncio.create_task(self._crosstalk.on_word_final(full, word_count))

    # ------------------------------------------------------------------
    # Crosstalk response handler — called when speculation settles

    def _on_crosstalk_response(
        self, bot_text: str, latency: float, provider: str
    ) -> None:
        """Called by Crosstalk when an LLM response commits (not cancelled)."""
        user_text = " ".join(self._pending_finals).strip()
        self._pending_finals.clear()

        asyncio.create_task(self._handle_committed_response(user_text, bot_text, latency, provider))

    async def _handle_committed_response(
        self, user_text: str, bot_text: str, latency: float, provider: str
    ) -> None:
        if self._on_bot_response:
            self._on_bot_response(bot_text, latency, provider)
        if self._on_status:
            self._on_status("stt", "idle")
            self._on_status("llm", f"{latency:.2f}s")
            self._on_status("tts", "synth")

        # Update history
        if user_text:
            self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": bot_text})
        # Keep history bounded
        if len(self._history) > 20:
            self._history = self._history[-20:]

        # TTS
        try:
            audio_bytes, bot_chars = await self._tts.synthesize(bot_text, self._session_start)

            # Log bot chars first
            for ch in bot_chars:
                await self._store.append(ch)
            if self._on_chars and bot_chars:
                self._on_chars(bot_chars)

            if self._on_status:
                self._on_status("tts", "playing")

            await self._tts.play_audio(audio_bytes)

            if self._on_status:
                self._on_status("tts", "idle")

        except Exception as exc:
            print(f"[orchestrator] TTS error: {exc}")
            if self._on_status:
                self._on_status("tts", "error")
