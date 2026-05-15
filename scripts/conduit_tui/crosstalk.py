"""
crosstalk.py — Speculative LLM start + cancellation coordinator.

Algorithm (based on tarzain/crosstalk, commit 327b2da):
  Original repo: https://github.com/tarzain/crosstalk
  Original approach: speaker diarization + LLM "continuation prediction" to
  anticipate end-of-turn. We simplify to a silence-window approach using
  Deepgram is_final events (no diarization needed — user and bot are separate
  channels in our stack).

Our algorithm:
  1. On each is_final word from Deepgram → record last_word_time, cancel any
     in-flight speculation (user kept talking), then schedule a speculative LLM
     call after SPECULATIVE_THRESHOLD_MS if word_count >= MIN_WORDS.
  2. During the speculative sleep, if another is_final fires, the sleep wakes
     up early via the cancellation guard and bails — user kept talking.
  3. If the speculative LLM call completes and SETTLED_THRESHOLD_MS passes
     with no new words, we commit: fire on_response_ready(text).
  4. Cancellation is via asyncio.Task.cancel() — no buffered audio discarded
     (mic stays live). Deepgram WS keep-alive is not touched.

Env vars (read by app.py / orchestrator.py):
  CROSSTALK_SPECULATIVE_THRESHOLD_MS  (default 250)
  CROSSTALK_SETTLED_THRESHOLD_MS      (default 600)
  CROSSTALK_MIN_WORDS                 (default 3)
"""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable


class Crosstalk:
    """
    Coordinates speculative LLM start + cancellation based on Deepgram streaming.

    Trigger speculative LLM call when:
      - latest is_final word ended >= SPECULATIVE_THRESHOLD_MS ago (default 250ms)
      - AND we have at least MIN_WORDS_FOR_SPECULATION user words (default 3)

    Cancel the in-flight LLM call when:
      - a new is_final word arrives before SETTLED_THRESHOLD_MS (default 600ms)
        of the speculative start — user kept talking

    Commit (= invoke on_response_ready) when:
      - SETTLED_THRESHOLD_MS passed since speculative LLM result landed AND
        no new words arrived
    """

    SPECULATIVE_THRESHOLD_MS: int = int(
        os.environ.get("CROSSTALK_SPECULATIVE_THRESHOLD_MS", "250")
    )
    SETTLED_THRESHOLD_MS: int = int(
        os.environ.get("CROSSTALK_SETTLED_THRESHOLD_MS", "600")
    )
    MIN_WORDS_FOR_SPECULATION: int = int(
        os.environ.get("CROSSTALK_MIN_WORDS", "3")
    )

    def __init__(
        self,
        llm_client,
        on_response_ready: Callable[[str, float, str], None],
    ) -> None:
        """
        Args:
            llm_client: LLMClient instance (must have async .complete(text, history))
            on_response_ready: called with (bot_text, latency, provider) when
                               a speculation settles and is committed.
        """
        self._llm = llm_client
        self._on_response = on_response_ready

        self._current_task: asyncio.Task | None = None
        self._last_word_time: float = 0.0
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Public API

    def set_history(self, history: list[dict[str, str]]) -> None:
        """Keep local ref to conversation history (updated by orchestrator)."""
        self._history = history

    async def on_word_final(self, text_so_far: str, word_count: int) -> None:
        """
        Called by orchestrator on each Deepgram is_final event.

        Args:
            text_so_far: full accumulated user utterance text (space-joined words)
            word_count:  number of words accumulated so far this turn
        """
        now = time.monotonic()
        self._last_word_time = now

        # Cancel any in-flight speculation — user kept talking
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass
            self._current_task = None

        # Don't speculate on fragments too short to be meaningful
        if word_count < self.MIN_WORDS_FOR_SPECULATION:
            return

        # Schedule speculation — captures word_time in closure for guard check
        word_time = now
        self._current_task = asyncio.create_task(
            self._speculate(text_so_far, word_time)
        )

    async def cancel(self) -> None:
        """Cancel any in-flight speculation (e.g. on conversation stop)."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            try:
                await self._current_task
            except (asyncio.CancelledError, Exception):
                pass
            self._current_task = None

    # ------------------------------------------------------------------
    # Internal

    async def _speculate(self, text: str, triggered_at: float) -> None:
        """
        1. Wait SPECULATIVE_THRESHOLD_MS — bail if user kept talking.
        2. Fire LLM call.
        3. Wait SETTLED_THRESHOLD_MS after LLM result — bail if user kept talking.
        4. Commit: invoke on_response_ready.
        """
        try:
            # Phase 1: wait speculative threshold
            await asyncio.sleep(self.SPECULATIVE_THRESHOLD_MS / 1000)

            # Guard: new word arrived during sleep?
            if self._last_word_time != triggered_at:
                return

            # Phase 2: LLM call (can be cancelled mid-await)
            t0 = time.monotonic()
            bot_text, latency, provider = await self._llm.complete(
                text, self._history
            )

            # Guard again: new word arrived while LLM was running?
            if self._last_word_time != triggered_at:
                return

            # Phase 3: wait settled threshold before committing
            await asyncio.sleep(self.SETTLED_THRESHOLD_MS / 1000)

            # Final guard
            if self._last_word_time != triggered_at:
                return

            # Commit
            self._on_response(bot_text, latency, provider)

        except asyncio.CancelledError:
            # Clean cancellation — user kept talking or caller cancelled
            raise
