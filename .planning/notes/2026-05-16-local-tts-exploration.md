# Local TTS — close the fully-offline gap (Piper / Coqui / OuteTTS)

**Source:** session 2026-05-16 readme gaps

ElevenLabs is the last cloud dep in the audio path. With Vosk handling STT locally and the speaker classifier ONNX (also local), TTS is the only thing standing between us and full offline operation.

Candidates:
- **Piper** (Rhasspy) — fast neural TTS, ONNX-runnable, ~50-100 MB per voice, good quality, English voices abundant.
- **Coqui TTS** (XTTS-v2) — higher quality, supports voice cloning from 6s sample, ~1.5 GB total. Heavier but voice-cloning is interesting for personalization.
- **OuteTTS** — newer, transformer-based, supports voice cloning, ~1-2 GB.
- **edge-tts** — uses Microsoft's online Edge TTS endpoint (free but cloud); fast first-byte (~200ms) but not truly offline.

Use-case fit: real-time conversational with low first-byte latency. Piper looks like the best Vosk-equivalent for TTS (small, fast, offline, ONNX-native).

When to do this: phase 09 candidate AFTER the multi-speaker meeting (phase 05). Two voices needed for meeting mode anyway; Piper has dozens of English voices that could replace the ElevenLabs voice pool entirely.

Source provenance: README 'Known gaps' #6, session 2026-05-16.
