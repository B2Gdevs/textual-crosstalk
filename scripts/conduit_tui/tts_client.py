"""
tts_client.py — ElevenLabs TTS with native char-level timestamps.

Uses ElevenLabs SDK client.text_to_speech.with_timestamps.convert().
Returns (audio_bytes, list[CharEntry]) where char times are absolute
monotonic (audio_start + offset_from_elevenlabs).

Audio playback via sounddevice OutputStream.
"""
from __future__ import annotations

import asyncio
import io
import time
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from .char_timeline import CharEntry

try:
    from elevenlabs import ElevenLabs
except ImportError:
    ElevenLabs = None  # type: ignore[assignment,misc]


class ElevenLabsClient:
    """ElevenLabs TTS with char-level timestamp support."""

    SAMPLE_RATE = 44100  # ElevenLabs default mp3 → we request PCM via mpegts or use bytes directly

    def __init__(
        self,
        api_key: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_monolingual_v1",
    ) -> None:
        if ElevenLabs is None:
            raise ImportError("elevenlabs package not installed — pip install elevenlabs")
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id
        self._model_id = model_id

    async def synthesize(
        self,
        text: str,
        session_start: float,
    ) -> tuple[bytes, list[CharEntry]]:
        """Synthesize text, return (audio_mp3_bytes, char_entries).

        char_entries use absolute monotonic time: session_start + char_offset.
        Runs ElevenLabs call in executor to avoid blocking event loop.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_synthesize,
            text,
            session_start,
        )
        return result

    def _sync_synthesize(
        self,
        text: str,
        session_start: float,
    ) -> tuple[bytes, list[CharEntry]]:
        response = self._client.text_to_speech.convert_with_timestamps(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model_id,
            output_format="mp3_44100_128",
        )

        # Response has: audio_base64, alignment dict
        import base64

        audio_bytes = base64.b64decode(response.audio_base64)

        char_entries: list[CharEntry] = []
        alignment = response.alignment
        if alignment:
            characters = alignment.characters or []
            starts = alignment.character_start_times_seconds or []
            ends = alignment.character_end_times_seconds or []

            for ch, start_offset, end_offset in zip(characters, starts, ends):
                char_entries.append(
                    CharEntry(
                        char=ch,
                        start_time=session_start + float(start_offset),
                        end_time=session_start + float(end_offset),
                        notes="bot,native",
                    )
                )

        return audio_bytes, char_entries

    async def play_audio(self, audio_bytes: bytes) -> None:
        """Decode mp3 and play via sounddevice. Runs in executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._sync_play, audio_bytes)

    def _sync_play(self, audio_bytes: bytes) -> None:
        try:
            import pydub  # type: ignore[import]
            audio = pydub.AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            if audio.channels == 2:
                samples = samples.reshape(-1, 2)
            sd.play(samples, samplerate=audio.frame_rate, blocking=True)
        except ImportError:
            # pydub not available — try soundfile
            try:
                import soundfile as sf  # type: ignore[import]
                data, sr = sf.read(io.BytesIO(audio_bytes))
                sd.play(data, samplerate=sr, blocking=True)
            except Exception as exc:
                print(f"[tts] audio playback failed (install pydub or soundfile): {exc}")
        except Exception as exc:
            print(f"[tts] playback error: {exc}")
