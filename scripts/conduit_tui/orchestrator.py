"""
orchestrator.py — ConversationLoop: wires STT → LLM → TTS, manages turn state.

Turn flow:
  1. Collect is_final transcripts until 1.5s silence
  2. LLM complete on accumulated text
  3. ElevenLabs synthesize → char entries logged → audio playback

All char entries (user STT + bot TTS) go to JsonlStore.
UI callbacks update StatusBar + transcript logs.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass


from .char_timeline import CharEntry, JsonlStore
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

    SILENCE_TIMEOUT = 1.5  # seconds after last is_final before LLM call

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

        # Turn accumulation
        self._pending_finals: list[str] = []
        self._last_final_time: float = 0.0
        self._final_event = asyncio.Event()

        # Set callbacks on deepgram stream
        self._dg._on_chars = self._handle_dg_chars
        self._dg._on_partial = self._handle_dg_partial

    # ------------------------------------------------------------------
    # Entry point

    async def run(self) -> None:
        self._running = True
        self._stop_event.clear()

        mic_task = asyncio.create_task(self._mic_pump())
        turn_task = asyncio.create_task(self._turn_loop())
        level_task = asyncio.create_task(self._level_pump())

        try:
            await asyncio.wait(
                [mic_task, turn_task, level_task],
                return_when=asyncio.FIRST_EXCEPTION,
            )
        finally:
            self._running = False
            mic_task.cancel()
            turn_task.cancel()
            level_task.cancel()
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

        # Reconstruct word text for final callback
        word = "".join(c.char for c in chars if c.char.strip())
        if word:
            self._pending_finals.append(word)
            self._last_final_time = time.monotonic()
            self._final_event.set()
            full = " ".join(self._pending_finals)
            if self._on_user_final:
                self._on_user_final(full)
            if self._on_status:
                self._on_status("stt", "final")

    # ------------------------------------------------------------------
    # Turn loop: collect silence → LLM → TTS

    async def _turn_loop(self) -> None:
        while self._running:
            # Wait for at least one final transcript
            try:
                await asyncio.wait_for(self._final_event.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            self._final_event.clear()

            # Wait for silence window (1.5s after last final)
            while True:
                time_since_last = time.monotonic() - self._last_final_time
                remaining = self.SILENCE_TIMEOUT - time_since_last
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._final_event.wait(), timeout=remaining)
                    self._final_event.clear()
                except asyncio.TimeoutError:
                    break

            if not self._pending_finals:
                continue

            user_text = " ".join(self._pending_finals).strip()
            self._pending_finals.clear()

            if not user_text:
                continue

            # LLM call
            if self._on_status:
                self._on_status("stt", "idle")
                self._on_status("llm", "thinking")

            try:
                bot_text, latency, provider = await self._llm.complete(user_text, self._history)
            except Exception as exc:
                print(f"[orchestrator] LLM error: {exc}")
                if self._on_status:
                    self._on_status("llm", "error")
                continue

            if self._on_bot_response:
                self._on_bot_response(bot_text, latency, provider)
            if self._on_status:
                self._on_status("llm", f"{latency:.2f}s")
                self._on_status("tts", "synth")

            # Update history
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
