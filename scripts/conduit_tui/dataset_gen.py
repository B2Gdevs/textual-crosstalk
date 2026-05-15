"""
dataset_gen.py — generate an ElevenLabs voice corpus for benchmark / meeting-mode use.

Usage:
    python -m scripts.conduit_tui.dataset_gen [--voices N] [--phrases path] [--out data/dataset]

Produces:
    <out>/voices/<voice_id>/<index>.wav   (int16 PCM, decoded from ElevenLabs mp3)
    <out>/manifest.json                   (list of {voice_id, voice_name, phrase, wav_path})

Idempotent: existing wav files are skipped, so re-runs only fill gaps.

Notes:
- Uses the same SDK call as scripts/conduit_tui/tts_client.py
  (convert_with_timestamps, eleven_flash_v2_5, mp3_44100_128 — free-tier safe).
- We ignore the alignment payload here; this script only needs the audio bytes.
- mp3 is decoded with miniaudio and written as a stdlib `wave` PCM file at the
  source sample rate (typically 44100 Hz, mono after channel-collapse).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

try:
    from elevenlabs import ElevenLabs
except ImportError as exc:  # pragma: no cover
    raise SystemExit("elevenlabs not installed — pip install elevenlabs") from exc

try:
    import miniaudio
except ImportError as exc:  # pragma: no cover
    raise SystemExit("miniaudio not installed — pip install miniaudio") from exc


# ---------------------------------------------------------------------------
# Defaults — premade ElevenLabs voices (free-tier compatible)
# ---------------------------------------------------------------------------

DEFAULT_VOICES: list[tuple[str, str]] = [
    ("Sarah", "EXAVITQu4vr4xnSDxMaL"),
    ("Roger", "CwhRBWXzGAHq8TQ4Fs17"),
    ("Laura", "FGY2WhTYpPnrIDTdsKH5"),
    ("Charlie", "IKne3meq5aSn9XLyUdCD"),
    ("George", "JBFqnCBsd6RMkjVDRZzb"),
]

# 30 conversational sentences, 5-15 words each, mixed meeting register.
DEFAULT_PHRASES: list[str] = [
    "I think we should look at this differently.",
    "What if we tried a smaller scope first?",
    "Yes, that matches what I was thinking.",
    "Can you walk me through that one more time?",
    "Let's circle back to the timeline before we wrap up.",
    "Honestly, the numbers from last quarter surprised me.",
    "I'm not convinced that's the real bottleneck here.",
    "Quick question — who owns this after launch?",
    "That's a fair point, I hadn't considered the latency angle.",
    "We could ship a draft this week and iterate.",
    "Sorry, can you repeat the last part? I missed it.",
    "Right, so the customer signal is pointing the other way.",
    "Let me share my screen and show you what I mean.",
    "Okay, I'll take the action item to follow up by Friday.",
    "I disagree, but only on the rollout order.",
    "If we cut feature three, does the date hold?",
    "Has anyone actually tested this end to end yet?",
    "Cool, sounds like we have rough alignment on the goal.",
    "One concern — we don't have data on the long tail.",
    "Let's park that and come back to it next week.",
    "I love that idea, let's prototype it tomorrow.",
    "Wait, that contradicts what we agreed on Monday.",
    "Hold on, I want to make sure I understand the constraint.",
    "Could be wrong, but I think the answer is yes.",
    "Great, I'll send a follow-up note with the decision.",
    "Honestly, I'd rather over-communicate than miss this.",
    "From the user's perspective, the flow still feels heavy.",
    "Alright, let's move on — anything else before we close?",
    "My gut says we should ship and learn from real traffic.",
    "Thanks everyone, this was genuinely useful, talk soon.",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

@dataclass
class ManifestEntry:
    voice_id: str
    voice_name: str
    phrase: str
    wav_path: str

    def to_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "voice_name": self.voice_name,
            "phrase": self.phrase,
            "wav_path": self.wav_path,
        }


def load_phrases(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PHRASES)
    text = path.read_text(encoding="utf-8")
    phrases = [line.strip() for line in text.splitlines() if line.strip()]
    if not phrases:
        raise SystemExit(f"--phrases file is empty: {path}")
    return phrases


def synth_mp3(client: ElevenLabs, voice_id: str, text: str) -> bytes:
    """Call ElevenLabs and return raw mp3 bytes."""
    response = client.text_to_speech.convert_with_timestamps(
        voice_id=voice_id,
        text=text,
        model_id="eleven_flash_v2_5",
        output_format="mp3_44100_128",
    )
    return base64.b64decode(response.audio_base_64)


def mp3_to_wav_bytes(mp3_bytes: bytes, wav_path: Path) -> None:
    """Decode mp3 -> int16 PCM, collapse to mono, write a stdlib wave file."""
    decoded = miniaudio.decode(mp3_bytes)
    samples = np.frombuffer(decoded.samples, dtype=np.int16)

    nch = int(decoded.nchannels)
    if nch > 1:
        # Interleaved frames -> (frames, channels); collapse to mono via mean
        samples = samples.reshape(-1, nch)
        samples = samples.mean(axis=1).astype(np.int16)

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(decoded.sample_rate))
        wf.writeframes(samples.tobytes())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voices",
        type=int,
        default=len(DEFAULT_VOICES),
        help=f"Number of premade voices to use (1..{len(DEFAULT_VOICES)}, default all)",
    )
    parser.add_argument(
        "--phrases",
        type=Path,
        default=None,
        help="Optional path to a newline-separated phrases file (default: built-in 30)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/dataset"),
        help="Output directory (default: data/dataset)",
    )
    args = parser.parse_args(argv)

    # Load env from .env.local first, then .env (matches the rest of the project)
    load_dotenv(".env.local")
    load_dotenv(".env")
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("ELEVENLABS_API_KEY not set (looked in .env.local / .env / os env)")

    if args.voices < 1 or args.voices > len(DEFAULT_VOICES):
        raise SystemExit(
            f"--voices must be between 1 and {len(DEFAULT_VOICES)} (got {args.voices})"
        )

    voices = DEFAULT_VOICES[: args.voices]
    phrases = load_phrases(args.phrases)
    out_dir: Path = args.out

    client = ElevenLabs(api_key=api_key)

    total = len(voices) * len(phrases)
    print(
        f"[dataset] {len(voices)} voices x {len(phrases)} phrases = {total} wavs -> {out_dir}",
        flush=True,
    )

    manifest: list[ManifestEntry] = []
    errors: list[tuple[str, int, str]] = []
    counter = 0
    t_start = time.monotonic()

    for voice_name, voice_id in voices:
        voice_dir = out_dir / "voices" / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        for idx, phrase in enumerate(phrases):
            counter += 1
            wav_path = voice_dir / f"{idx:03d}.wav"
            rel_path = wav_path.as_posix()

            if wav_path.exists() and wav_path.stat().st_size > 0:
                print(
                    f"[{counter}/{total}] {voice_name} phrase {idx} (skip, exists)",
                    flush=True,
                )
                manifest.append(ManifestEntry(voice_id, voice_name, phrase, rel_path))
                continue

            print(f"[{counter}/{total}] {voice_name} phrase {idx}", flush=True)
            try:
                mp3_bytes = synth_mp3(client, voice_id, phrase)
                mp3_to_wav_bytes(mp3_bytes, wav_path)
                manifest.append(ManifestEntry(voice_id, voice_name, phrase, rel_path))
            except Exception as exc:  # noqa: BLE001 — keep going, record error
                errors.append((voice_name, idx, repr(exc)))
                print(f"  ! error: {exc}", file=sys.stderr, flush=True)

    # Write manifest unconditionally (idempotent, reflects current on-disk state)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([m.to_dict() for m in manifest], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    elapsed = time.monotonic() - t_start
    print(
        f"[dataset] done: {len(manifest)} entries, {len(errors)} errors, {elapsed:.1f}s",
        flush=True,
    )
    if errors:
        print("[dataset] errors:", flush=True)
        for name, idx, msg in errors:
            print(f"  - {name} #{idx}: {msg}", flush=True)
    print(f"[dataset] manifest -> {manifest_path}", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
