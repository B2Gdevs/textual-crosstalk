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
| `AEC_PREPROCESS` | `0` (off) | enable speex's noise-suppress + AGC + VAD wrapping the AEC. **Off by default** — its adaptive noise floor was found to over-suppress user speech across turn boundaries (see `ERRORS-AND-ATTEMPTS.xml` → `cannot-interrupt-and-continue-2026-05-15`). Flip to `1` if you want it back. |
| `AEC_MU` | `0.3` | step size for numpy fallback (ignored when speex is active) |

## Conversational tuning

Behavior knobs that shape how forgiving / aggressive the conversation feels:

| Knob | Effect |
|---|---|
| `CROSSTALK_MIN_WORDS=1` (default) | Even 1-word follow-ups ("no", "stop", "wait") fire an LLM cycle. Raise to 2-3 to suppress filler triggers ("uh", "um"). |
| `CROSSTALK_SPECULATIVE_THRESHOLD_MS=250` | How long the user must pause before the LLM starts speculating on their utterance. Lower = snappier but more wasted LLM calls when user resumes. |
| `CROSSTALK_SETTLED_THRESHOLD_MS=600` | How long the pause must persist for the speculative response to commit and TTS to start. |
| `BARGE_IN_PARTIAL_CHARS=3` | Min chars of Deepgram partial text that cuts in-flight TTS. Lower = touchier barge. |
| `POST_TTS_COOLDOWN_S=0.6` | After natural TTS end, finals are dropped from Crosstalk for this window (echo tail). Barge bypasses it. |

**Status indicator.** The status bar shows `STT: listening` when the system is actively accepting your speech, and `STT: muted` while the finals gate is closed (during TTS + cooldown). A barge flips it to `listening` immediately. If the bar says `muted` and you want to be heard, wait the cooldown out OR speak loud enough to trigger barge.

## Speaker classifier

Pure-numpy 2-speaker classifier (user vs the specific TTS voice) used as a barge-in tiebreaker — if Deepgram emits a partial during TTS but the audio classifies as the bot voice (residual echo the AEC didn't fully suppress), barge is dropped. If it classifies as user, barge fires. Bias is toward "allow barge" when the classifier is uncertain.

**Features (29-dim per utterance):** 13 MFCC + 13 ΔMFCC + F0 (pitch) + spectral centroid + zero-crossing rate. L2-normalized, cosine similarity. Same DSP foundation Resemblyzer and SpeechBrain stand on — just with handcrafted features instead of learned ones, so no torch dependency.

**Enrollment:** automatic on first run. The user template is captured from the first AEC-cleaned user utterance and cached at `~/.conduit/voiceprint_user.npy`. The bot template is recomputed every session from the first TTS sample (the voice can change via `ELEVENLABS_VOICE_ID`). Delete the cache file to re-enroll.

**Tuning:**
| Variable | Default | Notes |
|---|---|---|
| `SPEAKER_MARGIN_THRESHOLD` | `0.005` | minimum cosine-similarity margin to trust the classification. Below this, classifier returns "unknown" and barge is allowed. Raise for stricter gating, lower for more responsiveness. |

### Upgrade path (when handcrafted features hit their ceiling)

The pure-numpy classifier gets ~85-90% accuracy on a 2-speaker closed-set, which is enough to materially improve barge gating. Two upgrade tiers if it isn't:

**Tier 1 — ONNX speaker embedding (~30-50 MB total).** Convert a pretrained ECAPA-TDNN (SpeechBrain) or x-vector model to ONNX, ship the model file + `onnxruntime`. No torch dependency. ~256-dim learned embeddings replace the 29 handcrafted features. Expected accuracy: ~95-98% EER. Trade-off: ~50 MB on disk, ~10ms per inference on CPU.

**Tier 2 — Distilled neural model (~5-15 MB).** Train (or fine-tune) a small CNN on mel-spectrograms for 2-class speaker discrimination. Quantize to int8. Could fit in <10 MB with minimal accuracy loss for the closed-set 2-speaker task. Most work; biggest payoff if Tier 1 still leaves gaps.

The current implementation in `scripts/conduit_tui/speaker_id.py` keeps the same interface (`extract_features`, `SpeakerClassifier.classify`) that either tier would slot into — `extract_features` becomes the ONNX forward pass; everything downstream is unchanged.

**Common conversational fixes if a turn isn't working:**
- *"Bot doesn't respond after I pause"* → lower `CROSSTALK_SETTLED_THRESHOLD_MS` or `CROSSTALK_MIN_WORDS`.
- *"Bot interrupts itself / replies to itself"* → confirm `pyaec` is installed (`pip show pyaec`); raise `AEC_FILTER_MS` to 256 for Bluetooth speakers.
- *"Can't interrupt the bot"* → lower `BARGE_IN_PARTIAL_CHARS` to 2; raise mic gain.
- *"Bot talks over me when I'm thinking"* → raise `CROSSTALK_SPECULATIVE_THRESHOLD_MS` and `CROSSTALK_SETTLED_THRESHOLD_MS`.
- *"After bot finishes, it takes a beat before I can speak again"* → lower `POST_TTS_COOLDOWN_S` to 0.2.

Implementation: `scripts/conduit_tui/crosstalk.py` (coordinator) + `scripts/conduit_tui/orchestrator.py` (wiring).
Reference: https://github.com/tarzain/crosstalk (commit 327b2da).

## Author

Benjamin Garrard — https://github.com/B2Gdevs
