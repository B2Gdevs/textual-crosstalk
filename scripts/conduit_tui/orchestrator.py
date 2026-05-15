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

import numpy as np


from .aec import make_echo_canceller
from .char_timeline import CharEntry, JsonlStore
from .crosstalk import Crosstalk
from .deepgram_client import DeepgramStream
from .llm_client import LLMClient
from .mic_capture import MicStream
from .speaker_id import SpeakerClassifier
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
        sample_rate: int = 16000,
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

        # Acoustic echo canceller — primary defense against the
        # bot-replies-to-itself loop. Subtracts the bot's known TTS
        # output from the live mic input before Deepgram ever sees it.
        # See scripts/conduit_tui/aec.py for backend selection.
        self._aec = make_echo_canceller(mic_rate=sample_rate)
        self._sample_rate = int(sample_rate)

        # Pure-numpy speaker classifier (MFCC + delta + F0 + centroid
        # + ZCR → 29-dim, cosine similarity). Used as a tiebreaker for
        # barge-in: if Deepgram emits a partial during TTS but the
        # underlying audio classifies as the bot voice (residual echo
        # the AEC didn't fully suppress), the barge is dropped. See
        # scripts/conduit_tui/speaker_id.py.
        self._spk = SpeakerClassifier(sample_rate=sample_rate)
        # Rolling 1.5s ring buffer of AEC-cleaned mic audio, so we can
        # classify the speech that produced any given Deepgram event.
        self._mic_ring_size = int(sample_rate * 1.5)
        self._mic_ring = np.zeros(0, dtype=np.int16)
        # Snapshot of mic audio at the moment the current user utterance
        # started — used to enroll the user template on first turn.
        self._user_enrol_start_size: int | None = None
        self._spk_margin_threshold = float(
            os.environ.get("SPEAKER_MARGIN_THRESHOLD", "0.005")
        )

        # Barge-in state
        self._tts_playing = False
        self._barge_in_chars = int(os.environ.get("BARGE_IN_PARTIAL_CHARS", "3"))
        # What the bot is currently saying — used to reject echo of bot
        # audio so we don't barge-in on ourselves.
        self._current_bot_text_norm: str = ""
        # Hard gate: monotonic time until which incoming Deepgram finals
        # must NOT be forwarded to Crosstalk. While the bot's audio is
        # bleeding into the mic there's no reliable way to distinguish
        # echo from real user speech via text alone (Deepgram occasionally
        # hallucinates a novel word, which slips word-set echo detection
        # and triggers the bot to reply to itself).
        # 0.0 means "accept now". Set to `monotonic() + tts_duration` when
        # TTS starts, advanced to `monotonic() + cooldown` after natural
        # end, and forced to `monotonic()` (open immediately) on a barge.
        self._finals_gate_until: float = 0.0
        self._post_tts_cooldown_s = float(
            os.environ.get("POST_TTS_COOLDOWN_S", "0.6")
        )

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
                # AEC: subtract the bot's known TTS audio from this mic
                # chunk before it reaches Deepgram. With no active
                # reference (bot not speaking), this is a fast no-op
                # pass-through.
                mic_arr = np.frombuffer(chunk, dtype=np.int16)
                cleaned = self._aec.process(mic_arr)
                if cleaned.size > 0:
                    # Append to rolling buffer for speaker classification.
                    self._mic_ring = np.concatenate([self._mic_ring, cleaned])
                    if self._mic_ring.size > self._mic_ring_size:
                        self._mic_ring = self._mic_ring[-self._mic_ring_size:]
                    await self._dg.send(cleaned.tobytes())
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[orchestrator] mic pump error: {exc}")

    async def _level_pump(self) -> None:
        prev_gate_open: bool | None = None
        while self._running:
            if self._on_mic_level:
                self._on_mic_level(self._mic.level)
            # Surface gate state so the user can see whether the mic is
            # being listened to or held closed during TTS+cooldown.
            gate_open = time.monotonic() >= self._finals_gate_until
            if gate_open != prev_gate_open and self._on_status:
                self._on_status("stt", "listening" if gate_open else "muted")
                prev_gate_open = gate_open
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
        # through the mic OR the speaker classifier says the underlying
        # audio is actually the bot, not the user.
        if self._tts_playing and len(text.strip()) >= self._barge_in_chars:
            if self._is_bot_echo(text):
                return
            if self._classify_recent_speech() == "bot":
                return
            self._barge_in()

    def _classify_recent_speech(self) -> str:
        """Classify the last ~600ms of AEC-cleaned mic audio. Returns
        'user' | 'bot' | 'unknown'. Falls back to 'user' (= allow barge)
        whenever the classifier is uncertain or not yet enrolled —
        bias is firmly toward letting the human be heard.
        """
        if not self._spk.enrolled:
            return "unknown"
        sr = self._sample_rate
        window = self._mic_ring[-int(sr * 0.6):]
        if window.size < int(sr * 0.25):
            return "unknown"
        label, margin = self._spk.classify(window)
        if margin < self._spk_margin_threshold:
            return "unknown"
        return label

    def _is_bot_echo(self, partial: str) -> bool:
        """True if the partial looks like residual bot audio bleeding
        through the AEC, not real user speech. With AEC working we
        expect this to be rare — kept as a backstop.

        Heuristics:
          1. Partial must have at least 2 words. Single-word partials
             ("yes", "no", "stop", "wait") are too ambiguous — the bot
             likely uses some of those words too, but the user
             ABSOLUTELY needs to be able to interrupt with them.
          2. Every word must also appear in the bot's outgoing text.

        Bias is toward false negatives (echo slips through and briefly
        self-cuts TTS) over false positives (silently swallowing the
        user's barge).
        """
        bot = self._current_bot_text_norm
        if not bot:
            return False
        norm = _normalize_for_echo(partial)
        if not norm:
            return False
        partial_words = norm.split()
        if len(partial_words) < 2:
            return False
        bot_words = set(bot.split())
        return all(w in bot_words for w in partial_words)

    async def _clear_echo_guard_after(self, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        self._current_bot_text_norm = ""

    def _prepare_bot_audio(self, audio_bytes: bytes) -> None:
        """Decode the bot mp3 once, push to AEC reference, and (one
        time) enrol the speaker classifier with the bot voice. Runs in
        a worker thread via asyncio.to_thread."""
        try:
            import miniaudio
        except ImportError:
            self._aec.push_reference_mp3(audio_bytes)
            return
        try:
            decoded = miniaudio.decode(audio_bytes)
        except Exception as exc:
            print(f"[orchestrator] tts decode for prepare failed: {exc}")
            self._aec.push_reference_mp3(audio_bytes)
            return
        samples = np.frombuffer(decoded.samples, dtype=np.int16)
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels)
        self._aec.push_reference(samples, decoded.sample_rate)
        if not self._spk.bot_enrolled:
            # Take a clean middle slice to enrol from (avoid attack/decay).
            mono = samples.mean(axis=1).astype(np.int16) if samples.ndim == 2 else samples
            slice_len = int(decoded.sample_rate * 1.5)
            mid = mono.size // 2
            clip = mono[max(0, mid - slice_len // 2): mid + slice_len // 2]
            # Resample to classifier rate (16kHz) if needed.
            if decoded.sample_rate != self._sample_rate:
                ratio = self._sample_rate / decoded.sample_rate
                n_out = max(1, int(clip.size * ratio))
                src_t = np.linspace(0.0, 1.0, clip.size, dtype=np.float32)
                dst_t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
                clip = np.interp(dst_t, src_t, clip.astype(np.float32)).astype(np.int16)
            self._spk.enrol_bot(clip)

    def _barge_in(self) -> None:
        """Cut in-flight TTS playback. Called from partial handler."""
        try:
            import sounddevice as sd
            sd.stop()
        except Exception as exc:
            print(f"[orchestrator] barge-in stop error: {exc}")
        self._tts_playing = False
        # Open the finals gate immediately — the user's intentional barge
        # is the trustworthy signal that they're speaking, not the bot.
        self._finals_gate_until = time.monotonic()
        self._current_bot_text_norm = ""
        # Drop any pending AEC reference — speaker is silent now, so
        # subsequent mic chunks should not be cancelled against the
        # cut-off bot audio.
        self._aec.clear_reference()
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

        # One-time user enrollment for the speaker classifier — take the
        # mic ring buffer at the moment of the first real is_final
        # (when we're sure the user just spoke).
        if not self._spk.user_enrolled and self._mic_ring.size >= self._sample_rate // 2:
            enrol_clip = self._mic_ring.copy()
            self._spk.enrol_user(enrol_clip)

        # Hard time gate: while TTS is playing (and during the brief
        # post-TTS cooldown), drop finals before they can reach Crosstalk.
        # The bot's own audio bleeds through the mic and Deepgram cannot
        # reliably be distinguished from user speech by text-matching
        # alone — a single hallucinated word slips the echo heuristic and
        # the bot ends up replying to itself. A barge-in event opens the
        # gate immediately so the user's intentional interruption is
        # still captured in the next final.
        if time.monotonic() < self._finals_gate_until:
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
            # Block finals from reaching Crosstalk during playback. We
            # use a generous upper bound (estimated by bot text length —
            # ~13 chars per second of natural speech is a safe lower
            # bound on duration) and we replace it precisely on the
            # finally branch once we know playback actually ended.
            est_duration_s = max(1.5, len(bot_text) / 12.0)
            self._finals_gate_until = time.monotonic() + est_duration_s + 5.0
            # Push the bot's audio to the AEC reference stream BEFORE
            # playback starts. Decode runs in a worker thread so we
            # don't block the event loop. While decode is in flight,
            # mic_pump may process a chunk or two with no reference yet
            # — those arrive before the speaker has actually started
            # producing echo, so the pass-through is correct.
            # We also (once) enrol the bot's voiceprint from this PCM
            # so the speaker classifier can distinguish it from the
            # user at runtime.
            await asyncio.to_thread(
                self._prepare_bot_audio, audio_bytes
            )
            try:
                await self._tts.play_audio(audio_bytes)
            finally:
                self._tts_playing = False
                # Drain AEC. Any residual reference samples that mic_pump
                # didn't consume (rate slop at TTS edges) would otherwise
                # be cancelled against pure user speech on the next turn
                # and silence the user.
                self._aec.clear_reference()
                # Only apply the post-TTS cooldown if the gate is still
                # in the future — i.e. TTS ended naturally. If barge
                # already opened the gate (set _finals_gate_until to a
                # past timestamp), do not re-close it: the user's
                # continuation needs to flow into Crosstalk immediately.
                # This was the root cause of "can't interrupt and
                # continue" — the finally branch was overwriting the
                # barge-opened gate with a fresh 0.6s cooldown.
                now = time.monotonic()
                if self._finals_gate_until > now:
                    self._finals_gate_until = now + self._post_tts_cooldown_s
                asyncio.create_task(self._clear_echo_guard_after(0.25))

            if self._on_status:
                self._on_status("tts", "idle")

        except Exception as exc:
            print(f"[orchestrator] TTS error: {exc}")
            self._tts_playing = False
            self._current_bot_text_norm = ""
            self._finals_gate_until = time.monotonic()
            if self._on_status:
                self._on_status("tts", "error")
