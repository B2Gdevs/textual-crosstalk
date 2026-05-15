"""
app.py — ConduitApp(App): top-level Textual application.

Screens: ConversationScreen (default).
Key bindings: q quit, r reset turn, s stop TTS.
Palette: #0d0d0d bg, #D4A017 gold accents.
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding

from .conversation_screen import ConversationScreen


class ConduitApp(App):
    """Realtime voice conversation with unified char timeline."""

    TITLE = "Conduit — Voice Conversation"
    CSS = """
    App {
        background: #0d0d0d;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True, priority=True),
    ]

    def on_mount(self) -> None:
        self.push_screen(ConversationScreen())
