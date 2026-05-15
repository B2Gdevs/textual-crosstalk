"""
conversation_screen.py — ConversationScreen(Screen): main 3-zone layout.

Layout (inspired by chat_screen.py two-panel pattern from slm_learning):
  ┌─────────────────────────────────────────────┐
  │ STATUS                                       │ ← StatusBar (1 line)
  ├──────────────────┬──────────────────────────┤
  │ You said         │ LLM said                  │ ← Two RichLog panels
  │                  │                           │
  ├──────────────────┴──────────────────────────┤
  │ DataTable: char timeline                     │ ← Last 20 chars
  └─────────────────────────────────────────────┘

Palette: #0d0d0d bg, #D4A017 gold, #C92A2A red.
"""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Label, RichLog

from .char_timeline import CharEntry, JsonlStore
from .deepgram_client import DeepgramStream
from .llm_client import LLMClient
from .mic_capture import MicStream
from .orchestrator import ConversationLoop
from .status import StatusBar
from .tts_client import ElevenLabsClient


_MAX_TABLE_ROWS = 20


class ConversationScreen(Screen):
    """Main conversation screen."""

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("r", "reset_turn", "Reset turn", show=True),
        Binding("s", "stop_tts", "Stop TTS", show=True),
    ]

    DEFAULT_CSS = """
    ConversationScreen {
        background: #0d0d0d;
    }
    .panel-label {
        height: 1;
        background: #1a1a1a;
        color: #D4A017;
        text-style: bold;
        padding: 0 1;
    }
    .transcript-panel {
        width: 1fr;
        border: solid #2a2a2a;
    }
    RichLog {
        background: #0d0d0d;
        color: #cccccc;
        scrollbar-color: #D4A017;
    }
    DataTable {
        height: 8;
        background: #0d0d0d;
        border: solid #2a2a2a;
    }
    DataTable > .datatable--header {
        background: #1a1a1a;
        color: #D4A417;
    }
    DataTable > .datatable--cursor {
        background: #2a2a00;
        color: #FFD700;
    }
    Footer {
        background: #1a1a1a;
        color: #888888;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._loop_task: asyncio.Task[None] | None = None
        self._convo_loop: ConversationLoop | None = None
        self._session_start = time.monotonic()
        self._recent_chars: list[CharEntry] = []

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status_bar")
        with Horizontal():
            with Vertical(classes="transcript-panel"):
                yield Label("You said", classes="panel-label")
                yield RichLog(id="user_log", markup=True, highlight=False, wrap=True)
            with Vertical(classes="transcript-panel"):
                yield Label("LLM said", classes="panel-label")
                yield RichLog(id="bot_log", markup=True, highlight=False, wrap=True)
        yield DataTable(id="char_table", zebra_stripes=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#char_table", DataTable)
        table.add_columns("char", "start", "end", "notes")
        self._loop_task = asyncio.create_task(self._start_loop())

    async def _start_loop(self) -> None:
        """Initialize all clients and run the conversation loop."""
        status = self.query_one("#status_bar", StatusBar)

        try:
            dg_key = os.environ["DEEPGRAM_API_KEY"]
            or_key = os.environ.get("OPENROUTER_API_KEY")
            or_model = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
            groq_key = os.environ.get("GROQ_API_KEY")
            groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
            openai_key = os.environ.get("OPENAI_API_KEY")
            el_key = os.environ.get("ELEVENLABS_API_KEY")
            el_voice = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
            sample_rate = int(os.environ.get("CONDUIT_SAMPLE_RATE", "16000"))
            char_log = os.environ.get("CONDUIT_CHAR_LOG", "./data/chars.jsonl")
        except KeyError as exc:
            self._log_error(f"Missing env var: {exc}")
            return

        if not el_key:
            self._log_error("ELEVENLABS_API_KEY not set — TTS disabled")

        store = JsonlStore(char_log)

        try:
            dg_stream = DeepgramStream(
                api_key=dg_key,
                sample_rate=sample_rate,
                session_start=self._session_start,
            )
            await dg_stream.connect()
        except Exception as exc:
            self._log_error(f"Deepgram connect failed: {exc}")
            return

        try:
            llm = LLMClient(
                openrouter_api_key=or_key,
                openrouter_model=or_model,
                groq_api_key=groq_key,
                groq_model=groq_model,
                openai_api_key=openai_key,
            )
        except ValueError as exc:
            self._log_error(str(exc))
            return

        if el_key:
            tts = ElevenLabsClient(api_key=el_key, voice_id=el_voice)
        else:
            tts = _NoopTTS()  # type: ignore[assignment]

        status.set_stt("ready")

        async with MicStream(sample_rate=sample_rate) as mic:
            self._convo_loop = ConversationLoop(
                store=store,
                dg_stream=dg_stream,
                mic=mic,
                llm=llm,
                tts=tts,
                session_start=self._session_start,
                sample_rate=sample_rate,
                on_user_partial=self._on_user_partial,
                on_user_final=self._on_user_final,
                on_bot_response=self._on_bot_response,
                on_chars=self._on_chars,
                on_status=self._on_status_update,
                on_mic_level=self._on_mic_level,
            )
            await self._convo_loop.run()

    # ------------------------------------------------------------------
    # Callbacks (called from asyncio tasks — use call_from_thread only
    # if we move to background thread, here we're already in the event loop)

    def _on_user_partial(self, text: str) -> None:
        log = self.query_one("#user_log", RichLog)
        # Clear last line and rewrite partial
        log.clear()
        log.write(f"[dim](partial: {text})[/dim]")

    def _on_user_final(self, text: str) -> None:
        log = self.query_one("#user_log", RichLog)
        log.clear()
        log.write(f"[#cccccc]{text}[/#cccccc]")

    def _on_bot_response(self, text: str, latency: float, provider: str) -> None:
        log = self.query_one("#bot_log", RichLog)
        log.write(f"[#D4A017]{text}[/#D4A017]")
        status = self.query_one("#status_bar", StatusBar)
        status.set_llm(latency, provider)

    def _on_chars(self, chars: list[CharEntry]) -> None:
        table = self.query_one("#char_table", DataTable)
        for entry in chars:
            self._recent_chars.append(entry)
            table.add_row(
                repr(entry.char),
                f"{entry.start_time:.3f}",
                f"{entry.end_time:.3f}",
                entry.notes,
            )
        # Keep table bounded
        while table.row_count > _MAX_TABLE_ROWS:
            first_key = next(iter(table.rows))
            table.remove_row(first_key)

    def _on_status_update(self, component: str, state: str) -> None:
        status = self.query_one("#status_bar", StatusBar)
        if component == "stt":
            status.set_stt(state)
        elif component == "tts":
            status.set_tts(state)

    def _on_mic_level(self, level: float) -> None:
        status = self.query_one("#status_bar", StatusBar)
        status.set_mic(level)

    def _log_error(self, msg: str) -> None:
        log = self.query_one("#user_log", RichLog)
        log.write(f"[#C92A2A][ERROR] {msg}[/#C92A2A]")

    # ------------------------------------------------------------------
    # Key actions

    def action_reset_turn(self) -> None:
        if self._convo_loop:
            self._convo_loop._pending_finals.clear()
        self.query_one("#user_log", RichLog).clear()

    def action_stop_tts(self) -> None:
        import sounddevice as sd
        sd.stop()
        status = self.query_one("#status_bar", StatusBar)
        status.set_tts("stopped")

    def action_quit(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        self.app.exit()


class _NoopTTS:
    """Fallback when ElevenLabs key not set."""

    async def synthesize(self, text: str, session_start: float) -> tuple[bytes, list]:
        print(f"[tts-noop] would say: {text}")
        return b"", []

    async def play_audio(self, audio_bytes: bytes) -> None:
        pass
