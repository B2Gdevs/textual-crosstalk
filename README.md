# Conduit — Realtime Voice Conversation with Char Timeline

A Python TUI program where you speak, an LLM responds in a human voice, and every character spoken or heard is saved with millisecond-precision timestamps.

## Demo videos

- **Agents go off during the showcase** (two-way interruption getting chaotic in the best way) — https://www.loom.com/share/7e5b47ec7862425cacc0c174c75d04b5
- **Wrap-up walkthrough** — https://www.loom.com/share/33ba9b07cb694564aa2c8065b33cc3eb

## What it does

1. **Listen** — captures mic audio and streams to Deepgram for realtime transcription
2. **Transcribe** — word-level timestamps from Deepgram, linearly interpolated to per-character precision
3. **Think** — sends your transcript to an LLM (OpenRouter or Groq) and gets a response
4. **Speak** — synthesizes the response via ElevenLabs, which returns native per-character timing
5. **Store** — every character (yours + the bot's) appended to a unified JSONL file

**Data format** — one entry per character:
```json
{"char": "h", "start_time": 0.000, "end_time": 0.080, "notes": "user,interpolated,word=hello"}
{"char": "i", "start_time": 2.341, "end_time": 2.395, "notes": "bot,native"}
```

"hello world" → 11 entries (one per character, spaces included).

## Setup

**One-shot scripts** — create the venv, install deps, then launch:

```powershell
# Windows (PowerShell)
.\setup.ps1
```

```bash
# macOS / Linux
./setup.sh
```

Both scripts assume `.env.local` already exists with the keys below. They build `.venv`, `pip install -r requirements.txt`, and run `python main.py`.

**Manual setup** if you prefer step-by-step:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env.local` and fill in:

| Variable | Required | Notes |
|---|---|---|
| `DEEPGRAM_API_KEY` | Yes | console.deepgram.com |
| `OPENROUTER_API_KEY` | Yes (or Groq) | openrouter.ai |
| `GROQ_API_KEY` | Yes (or OpenRouter) | console.groq.com |
| `ELEVENLABS_API_KEY` | Yes | elevenlabs.io |
| `ELEVENLABS_VOICE_ID` | No | default: Sarah (`EXAVITQu4vr4xnSDxMaL`) — premade voice, works on the free tier. Rachel and other library voices require a paid plan. |
| `OPENAI_API_KEY` | No | fallback LLM |
| `CONDUIT_CHAR_LOG` | No | default: `./data/chars.jsonl` |
| `CONDUIT_SAMPLE_RATE` | No | default: 16000 |

## Run

```bash
python main.py
```

**Key bindings:**
- `q` — quit
- `r` — reset current turn (discard partial transcript)
- `s` — stop TTS playback

## Char data format

File: `./data/chars.jsonl` (one JSON object per line)

```python
@dataclass
class CharEntry:
    char: str        # single character (space, punctuation included)
    start_time: float  # seconds since session start (monotonic)
    end_time: float    # seconds since session start (monotonic)
    notes: str         # "user,interpolated" | "bot,native"
```

**STT chars** (`notes=user,interpolated,word=<word>`): Deepgram provides word start/end; chars are linearly interpolated within the word duration.

**TTS chars** (`notes=bot,native`): ElevenLabs `convert_with_timestamps` returns exact per-character timing from the synthesis engine.

## Crosstalk (Part 2)

Speculative LLM firing + cancellation — eliminates the fixed 1.5s silence-wait that made Part 1 feel sluggish.

**How it works:**
1. Every Deepgram `is_final` word fires `Crosstalk.on_word_final()`.
2. After `CROSSTALK_SPECULATIVE_THRESHOLD_MS` (250ms) of silence, the LLM call starts speculatively.
3. If a new `is_final` arrives before `CROSSTALK_SETTLED_THRESHOLD_MS` (600ms), the in-flight `asyncio.Task` is cancelled — user kept talking. No buffered audio discarded.
4. If silence holds for 600ms post-LLM-result, the response commits → TTS plays.

**Env vars:**

| Variable | Default | Notes |
|---|---|---|
| `CROSSTALK_SPECULATIVE_THRESHOLD_MS` | `250` | ms silence before speculative LLM start |
| `CROSSTALK_SETTLED_THRESHOLD_MS` | `600` | ms silence before committing to TTS |
| `CROSSTALK_MIN_WORDS` | `3` | minimum words to trigger speculation |
| `BARGE_IN_PARTIAL_CHARS` | `3` | min chars in a Deepgram partial that cuts in-flight TTS |

Fragments under 3 words (e.g. "um", "yeah") are never speculated on — avoids wasted LLM calls on filler.

**Barge-in:** when the bot is mid-sentence, the first Deepgram partial that reaches `BARGE_IN_PARTIAL_CHARS` characters calls `sounddevice.stop()` and the TTS state flips to `barged`. The user's subsequent `is_final` enters the normal Crosstalk path and starts a fresh turn. Raise the threshold to suppress short noises (cough, "uh"); lower it for more aggressive barge.

Implementation: `scripts/conduit_tui/crosstalk.py` (coordinator) + `scripts/conduit_tui/orchestrator.py` (wiring).
Reference: https://github.com/tarzain/crosstalk (commit 327b2da).

## Author

Benjamin Garrard — https://github.com/B2Gdevs
