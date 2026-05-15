"""
orchestrator.py — ConversationLoop: wires STT → LLM → TTS, manages turn state.

Turn flow (Part 2 — Crosstalk):
  1. Each is_final word from Deepgram feeds Crosstalk.on_word_final()
  2. Crosstalk speculatively fires LLM after SPECULATIVE_THRESHOLD_MS silence
  3. If user keeps talking, in-flight LLM task is cancelled (asyncio.Task.cancel)
  4. When SETTLED_THRESHOLD_MS passes with no new words, response commits → TTS

Barge-in: if a Deepgram partial arrives while TTS is playing AND the
partial has at least BARGE_IN_PARTIAL_CHARS characters, the in-flight
TTS playback is cut via sounddevice.stop(). The user's next is_final
will start a fresh turn through Crosstalk normally.

All char entries (user STT + bot TTS) go to JsonlStore.
UI callbacks update StatusBar + transcript logs.

Env vars:
  CROSSTALK_SPECULATIVE_THRESHOLD_MS  (default 250)
  CROSSTALK_SETTLED_THRESHOLD_MS      (default 600)
  CROSSTALK_MIN_WORDS                 (default 3)
  BARGE_IN_PARTIAL_CHARS              (default 3 — min partial length to cut TTS)
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass


from .char_timeline import CharEntry, JsonlStore
from .crosstalk import Crosstalk
from .deepgram_client import DeepgramStream
from .llm_client import LLMClient
from .mic_capture import MicStream
from .tts_client import ElevenLabsClient


_ECHO_STRIP = str.maketrans("", "", ".,!?;:\"'()[]{}—-")


def _normalize_for_echo(text: str) -> str:
    """Lowercase + strip punctuation, collapse whitespace. Used to compare
    a Deepgram partial against the bot's outgoing TTS text without being
    tripped up by capitalization or punctuation."""
    cleaned = text.translate(_ECHO_STRIP).lower()
    return " ".join(cleaned.split())


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

        # Barge-in state
        self._tts_playing = False
        self._barge_in_chars = int(os.environ.get("BARGE_IN_PARTIAL_CHARS", "3"))
        # What the bot is currently saying — used to reject echo of bot
        # audio so we don't barge-in on ourselves.
        self._current_bot_text_norm: str = ""

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

        # Barge-in: if bot is currently speaking and the user's partial
        # has crossed the threshold, cut the playback immediately —
        # unless the partial looks like the bot's own audio echoing back
        # through the mic (self-interruption).
        if self._tts_playing and len(text.strip()) >= self._barge_in_chars:
            if self._is_bot_echo(text):
                return
            self._barge_in()

    def _is_bot_echo(self, partial: str) -> bool:
        """True if every word in the partial also appears in the bot's
        current outgoing text. The presence of even ONE word the bot
        isn't saying counts as real user speech — the human should
        always be able to interrupt with "stop", "wait", "hold on", etc.
        Conservative on purpose: false negatives (echo not filtered) just
        cause a brief self-cut of TTS; false positives (real user
        speech misclassified) would silently swallow the user's barge.
        """
        bot = self._current_bot_text_norm
        if not bot:
            return False
        norm = _normalize_for_echo(partial)
        if not norm:
            return False
        bot_words = set(bot.split())
        partial_words = norm.split()
        if not partial_words:
            return False
        # If any word in the partial does NOT appear in the bot's text,
        # it's a real user utterance — not echo. Let barge-in fire.
        return all(w in bot_words for w in partial_words)

    async def _clear_echo_guard_after(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        self._current_bot_text_norm = ""

    def _barge_in(self) -> None:
        """Cut in-flight TTS playback. Called from partial handler."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as exc:
            print(f"[orchestrator] barge-in stop error: {exc}")
        self._tts_playing = False
        if self._on_status:
            self._on_status("tts", "barged")

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

        # Echo filter: if the bot is still speaking (or just finished) and
        # this final matches what the bot is saying, drop it. Otherwise the
        # bot's audio bleeding into the mic kicks off a Crosstalk cycle
        # that makes the bot reply to itself.
        echo_candidate = " ".join(words_in_batch)
        if self._current_bot_text_norm and self._is_bot_echo(echo_candidate):
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

            self._current_bot_text_norm = _normalize_for_echo(bot_text)
            self._tts_playing = True
            try:
                await self._tts.play_audio(audio_bytes)
            finally:
                self._tts_playing = False
                # Hold the bot-text echo guard for a brief tail so a
                # final that lands right after sd.play() returns still
                # gets filtered out of Crosstalk. Kept short so the next
                # legitimate user turn isn't suppressed.
                asyncio.create_task(self._clear_echo_guard_after(0.25))

            if self._on_status:
                self._on_status("tts", "idle")

        except Exception as exc:
            print(f"[orchestrator] TTS error: {exc}")
            self._tts_playing = False
            self._current_bot_text_norm = ""
            if self._on_status:
                self._on_status("tts", "error")
