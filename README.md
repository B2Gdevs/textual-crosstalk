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
| `POST_TTS_COOLDOWN_S` | `0.6` | seconds after natural TTS end during which finals are still dropped (echo tail) |

Fragments under 3 words (e.g. "um", "yeah") are never speculated on — avoids wasted LLM calls on filler.

**Barge-in:** when the bot is mid-sentence, the first Deepgram partial that reaches `BARGE_IN_PARTIAL_CHARS` characters calls `sounddevice.stop()` and the TTS state flips to `barged`. The user's subsequent `is_final` enters the normal Crosstalk path and starts a fresh turn. Raise the threshold to suppress short noises (cough, "uh"); lower it for more aggressive barge.

**Acoustic echo cancellation (AEC).** The primary defense against the bot-replies-to-itself loop. We have the reference signal — the exact PCM the bot is about to play — so we run a real adaptive filter against the live mic stream and subtract the bot's voice before it ever reaches Deepgram.

Backend: [`pyaec`](https://pypi.org/project/pyaec/) — ctypes bindings around speexdsp's AEC + preprocessor (same algorithm used by FreeSWITCH, Asterisk, and most SIP clients). Prebuilt Windows/macOS/Linux wheels, no toolchain. Pure-numpy block-NLMS fallback (`scripts/conduit_tui/aec.py`) if `pyaec` is unavailable.

Wiring:
- `_handle_committed_response` pushes the decoded TTS audio to the AEC reference stream before playback starts (`asyncio.to_thread` so the mp3 decode doesn't block the event loop).
- `_mic_pump` runs every mic chunk through `aec.process()` before sending to Deepgram. With no active reference the call is a fast pass-through.
- Barge-in calls `aec.clear_reference()` so post-cutoff mic chunks aren't cancelled against bot audio that's no longer playing.

Measured on a synthetic round-trip test: ~20 dB ERLE (echo return loss enhancement) after convergence. Real speech with a real speaker→mic path typically does better because the reference has richer spectral content.

**Time-gate safety belt (legacy).** The pre-AEC fix is still in place as a safety net while the speex filter is converging during the first frames of a bot turn — `is_final` events are dropped from Crosstalk during TTS playback + `POST_TTS_COOLDOWN_S` afterward. With AEC working this is largely redundant, but cheap to keep. Barge-in opens the gate immediately so an intentional interruption isn't suppressed.

| Variable | Default | Notes |
|---|---|---|
| `AEC_FILTER_MS` | `128` | speex tail length in ms — coverage for speaker latency + room reverb |
| `AEC_FRAME_MS` | `10` | AEC processing frame size in ms |
| `AEC_MU` | `0.3` | step size for numpy fallback (ignored when speex is active) |

Implementation: `scripts/conduit_tui/crosstalk.py` (coordinator) + `scripts/conduit_tui/orchestrator.py` (wiring).
Reference: https://github.com/tarzain/crosstalk (commit 327b2da).

## Author

Benjamin Garrard — https://github.com/B2Gdevs
