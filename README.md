# Conduit — Realtime Voice Conversation with Char Timeline

A Python TUI program where you speak, an LLM responds in a human voice, and every character spoken or heard is saved with millisecond-precision timestamps.

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

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
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
| `ELEVENLABS_VOICE_ID` | No | default: Rachel (21m00Tcm4TlvDq8ikWAM) |
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

Fragments under 3 words (e.g. "um", "yeah") are never speculated on — avoids wasted LLM calls on filler.

Implementation: `scripts/conduit_tui/crosstalk.py` (coordinator) + `scripts/conduit_tui/orchestrator.py` (wiring).
Reference: https://github.com/tarzain/crosstalk (commit 327b2da).

## Share

Repo to be shared with GitHub user **RioPopper**.
