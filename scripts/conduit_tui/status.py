"""
status.py — StatusBar widget: mic level, STT state, LLM latency, TTS state.

Color palette: black bg, gold #D4A017 accents, red #C92A2A for errors.
(memory: feedback_tui_aesthetic_palette.md)
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Static


class StatusBar(Static):
    """Single-line status strip across top of conversation screen."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: #1a1a1a;
        color: #D4A017;
        padding: 0 1;
    }
    """

    mic_level: reactive[float] = reactive(0.0)
    stt_state: reactive[str] = reactive("idle")
    llm_latency: reactive[float] = reactive(0.0)
    tts_state: reactive[str] = reactive("idle")
    provider: reactive[str] = reactive("")

    def render(self) -> str:
        bars_total = 5
        bars_on = round(self.mic_level * bars_total)
        mic_display = "▮" * bars_on + "░" * (bars_total - bars_on)

        latency_str = f"{self.llm_latency:.2f}s" if self.llm_latency > 0 else "—"
        provider_str = f" [{self.provider}]" if self.provider else ""

        return (
            f"mic {mic_display} · "
            f"STT: {self.stt_state} · "
            f"LLM: {latency_str}{provider_str} · "
            f"TTS: {self.tts_state}"
        )

    def set_mic(self, level: float) -> None:
        self.mic_level = level

    def set_stt(self, state: str) -> None:
        self.stt_state = state

    def set_llm(self, latency: float, provider: str = "") -> None:
        self.llm_latency = latency
        self.provider = provider

    def set_tts(self, state: str) -> None:
        self.tts_state = state
