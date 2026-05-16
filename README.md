# Conduit — Realtime Voice Conversation with Char Timeline

A Python TUI program where you speak, an LLM responds in a human voice, and every character spoken or heard is saved with millisecond-precision timestamps.

> **Note for reviewers** — this project was submitted on **2026-05-15 around 14:27 CST** at commit [`49086d9`](https://github.com/B2Gdevs/textual-crosstalk/commit/49086d9). Everything past that commit is post-submission iteration. The [Submission state vs current state](#submission-state-vs-current-state) section below partitions the two clearly and ties every change to its commit SHA and planning artifact (task / decision / error ID) so you can navigate without ambiguity. The [Known gaps and limitations](#known-gaps-and-limitations) section lists what's honestly not done yet.

## Demo videos

**Originally submitted (around the [`49086d9`](https://github.com/B2Gdevs/textual-crosstalk/commit/49086d9) commit, 2026-05-15 14:27 CST):**

- **Agents go off during the showcase** (two-way interruption getting chaotic) — https://www.loom.com/share/7e5b47ec7862425cacc0c174c75d04b5
- **Wrap-up walkthrough** — https://www.loom.com/share/33ba9b07cb694564aa2c8065b33cc3eb

**Post-submission improvement videos:**

- **Setup walkthrough + coding-agent workflow** — https://www.loom.com/share/193b775318d34848af388ca890cbf691
- **Barge-in and self-pickup fixes** (timestamped tour of the iteration) — https://www.loom.com/share/771344888110415a89d8fb329a8673de

## Submission state vs current state

This repo was submitted in a working v1 state at commit [`49086d9`](https://github.com/B2Gdevs/textual-crosstalk/commit/49086d9). What's in `main` today goes well beyond that — same project, deeper work. The list below maps every meaningful change since submission to the planning artifact that records it.

### What was in the submission (commit `49086d9`)

| Capability | Status at submission |
|---|---|
| Mic → Deepgram (streaming STT with word-level finals) | ✅ |
| OpenRouter / Groq / OpenAI LLM with fallback | ✅ |
| ElevenLabs TTS with native char-level timestamps | ✅ |
| Unified char timeline (JSONL, user + bot, ms precision) | ✅ |
| Textual TUI with mic level + status + transcript panels | ✅ |
| Crosstalk speculative LLM + cancellation | ✅ |
| One-shot setup script + Demo videos in README | ✅ |

### What's been added since (chronological, with planning anchors)

| # | Change | Commit | Planning artifacts |
|---|---|---|---|
| 1 | Multi-word LLM trigger fix — Deepgram is_final batches were being collapsed to one token | [`b4334ed`](https://github.com/B2Gdevs/textual-crosstalk/commit/b4334ed) | task `CONDUIT-PROJ-T-04-01` · error `llm-never-fires-after-pause-2026-05-15` |
| 2 | ElevenLabs free-tier TTS — model + format + voice + SDK field fixes | [`b4334ed`](https://github.com/B2Gdevs/textual-crosstalk/commit/b4334ed) | task `04-02` · decision `CONDUIT-PROJ-D-013` · error `elevenlabs-free-tier-three-walls-2026-05-15` |
| 3 | User barge-in (partial-char threshold cuts TTS) | [`18db2e6`](https://github.com/B2Gdevs/textual-crosstalk/commit/18db2e6) | task `04-03` · decision `D-014` |
| 4 | Text-layer echo guard (intermediate, later superseded) | [`398cd8a`](https://github.com/B2Gdevs/textual-crosstalk/commit/398cd8a), [`fd925d2`](https://github.com/B2Gdevs/textual-crosstalk/commit/fd925d2) | task `04-04` · decision `D-015` · error `bot-replies-to-itself-via-stt-echo-2026-05-15` |
| 5 | Repo cleanup — drop unused runtime entrypoints + SOUL.md | [`8b92a4f`](https://github.com/B2Gdevs/textual-crosstalk/commit/8b92a4f) | task `04-05` |
| 6 | Hard time-gate on finals during TTS+cooldown | [`b79eb8c`](https://github.com/B2Gdevs/textual-crosstalk/commit/b79eb8c) | task `04-07` · decision `D-016` |
| 7 | Acoustic Echo Cancellation via speexdsp (pyaec) | [`d12f366`](https://github.com/B2Gdevs/textual-crosstalk/commit/d12f366), [`0be5ec6`](https://github.com/B2Gdevs/textual-crosstalk/commit/0be5ec6) | task `04-08` · decision `D-017` · errors `user-silenced-after-first-bot-turn-2026-05-15` |
| 8 | Conversational tuning (preprocess off → on, MIN_WORDS=1, mic-gate status surfaced) | [`b5d11a8`](https://github.com/B2Gdevs/textual-crosstalk/commit/b5d11a8) | task `04-09` · decision `D-018` |
| 9 | Barge gate-reclose fix (the actual "can't interrupt and continue" root cause) | [`54c9bf6`](https://github.com/B2Gdevs/textual-crosstalk/commit/54c9bf6) | task `04-10` · error `post-barge-cooldown-reclosed-gate-2026-05-15` |
| 10 | Pure-numpy speaker classifier (Tier 0 — 29-dim MFCC + F0 + centroid + ZCR) | [`c69f46a`](https://github.com/B2Gdevs/textual-crosstalk/commit/c69f46a) | task `04-11` · decision `D-019` |
| 11 | Honest benchmark harness — measure before claiming, revert AEC preprocess to ON | [`99b090c`](https://github.com/B2Gdevs/textual-crosstalk/commit/99b090c) | task `04-12` · decision `D-020` · error `unverified-accuracy-claim-2026-05-15` |
| 12 | N-speaker classifier + VoxCeleb-O open-set mode + latency profile | [`c3a1b43`](https://github.com/B2Gdevs/textual-crosstalk/commit/c3a1b43) | task `04-13` · decision `D-021` |
| 13 | Real ElevenLabs dataset (5 voices × 30 phrases) + scenario rotation + operator capture + first project skill | [`d0e86d1`](https://github.com/B2Gdevs/textual-crosstalk/commit/d0e86d1) | tasks `06-01..06-04` · decision `D-022` |
| 14 | Real-data benchmark reveals Tier 0 ceiling (24% on real voices); Tier 1 ONNX planned | [`e9f5be8`](https://github.com/B2Gdevs/textual-crosstalk/commit/e9f5be8) | task `06-05` · error `classifier-tier0-fails-on-real-elevenlabs-2026-05-15` |
| 15 | Vosk local STT (env-switch `CONDUIT_STT=vosk`) + WER benchmark + personalized eval | [`2794adc`](https://github.com/B2Gdevs/textual-crosstalk/commit/2794adc) | tasks `06-06, 08-01` · decision `D-023` |
| 16 | **Tier 1 ONNX speaker classifier — 24% → 100% on 5-way ElevenLabs** | [`8feecf2`](https://github.com/B2Gdevs/textual-crosstalk/commit/8feecf2) | task `07-01` · decision `D-024` |
| 17 | STT benchmark live numbers + bench bug fix: Vosk beats Deepgram on WER (3.5% vs 4.0%) and first-partial latency (580 ms vs 877 ms) on this corpus | this commit | task `06-06` (re-measurement) |

### Measured benchmark snapshots

| Metric | Tier 0 (numpy MFCC) | Tier 1 (ONNX ECAPA-TDNN) |
|---|---|---|
| Speaker classifier 5-way closed-set accuracy on real ElevenLabs | **24%** (chance 20%) | **100%** (50/50) |
| Same-voice vs diff-voice cosine sim (real) | 0.992 / 0.993 (no separation) | 0.662 / 0.176 (+0.486 margin) |
| Speaker classifier latency per inference | 5.4 ms | ~60 ms |
| Install footprint | 0 MB extra | ~70 MB (onnxruntime + 25 MB model) |
| AEC ERLE (pure echo) | ~−3 dB (single tone) / ~11.5 dB (broadband) | same |
| AEC latency per 64ms chunk | 2.8 ms | same |

**The Tier 1 jump is the most important result of post-submission work.** Tier 0 hit a feature-space ceiling — its 80% on synthetic was misleading. Tier 1 (Wespeaker's `voxceleb_ECAPA512_LM.onnx`, 24.86 MB, no torch dependency) lifted 5-way real-voice accuracy from 24% → 100% on the same corpus, the same harness, the same 50 test trials. The 0.486 cosine-similarity margin between same-voice and different-voice pairs (vs ~0.001 for Tier 0) is the actual signal that lets the system distinguish speakers.

Switch backends via `CONDUIT_SPEAKER_TIER=onnx|numpy` (default `onnx` once installed). The numpy backend stays available as a no-extra-deps fallback. See `.planning/ERRORS-AND-ATTEMPTS.xml` → `classifier-tier0-fails-on-real-elevenlabs-2026-05-15` for the failure mode that Tier 1 closed.

### STT benchmark (20 samples from the ElevenLabs corpus)

`python -m scripts.conduit_tui.benchmark --stt auto --stt-samples 20` runs the same 20 ground-truth phrases through both backends and reports word-error rate + streaming latency:

| Backend | Mean WER | WER std | Latency-to-first-partial | Latency-to-final |
|---|---|---|---|---|
| Deepgram (nova-3) | 4.0% | 7.8% | 877 ms | 2352 ms |
| **Vosk (small-en-us)** | **3.5%** | **7.8%** | **580 ms** | 4138 ms |

Vosk **dominates Deepgram on this corpus** for accuracy AND first-partial latency. Deepgram still wins on time-to-final because its `utterance_end_ms=1500` aggressively closes turns; Vosk has no equivalent and final commit is driven by `Crosstalk.SETTLED_THRESHOLD_MS` silence. Most WER errors on both backends are punctuation rendering ("over-communicate" vs "overcommunicate") rather than true substitutions.

Switch via `CONDUIT_STT=deepgram|vosk` (default `deepgram` for compatibility — the existing turn-detection thresholds were tuned against its finalization timing).

### Planning artifacts in this repo

Every change above is recorded in machine-readable form under `.planning/`:

- [`.planning/ROADMAP.xml`](.planning/ROADMAP.xml) — phases 00-08 (00 setup, 01-03 original submission, 04 robustness pass, 05 multi-speaker meeting **planned**, 06 test infrastructure, 07 Tier 1 ONNX, 08 Vosk STT)
- [`.planning/DECISIONS.xml`](.planning/DECISIONS.xml) — 25 architecture decisions, each with rationale + references
- [`.planning/ERRORS-AND-ATTEMPTS.xml`](.planning/ERRORS-AND-ATTEMPTS.xml) — 8 named failure modes with the rule each one teaches the next agent
- [`.planning/tasks/*.json`](.planning/tasks/) — per-task records with attribution, status, file scope
- [`.claude/skills/`](.claude/skills/) — first project-specific skill (`aec-before-text-echo-guards`) captured for future agents

## Known gaps and limitations

Honest list of what's **not** done yet so a reviewer can calibrate. Each row links to the live GAD planning artifact (todo / note / phase) where the gap is tracked — `gad todos list --projectid conduit_proj` and `gad note list` are the CLI equivalents.

| Gap | Tracked as | Status |
|---|---|---|
| **Multi-speaker meeting (2 bots + 1 human)** | Phase 05 in [`ROADMAP.xml`](.planning/ROADMAP.xml) | planned, not started |
| **Tier 1 ONNX generalization to unseen voices** | [`tier1-generalization-test`](.planning/todos/2026-05-16-tier1-generalization-test.md) | agent todo |
| **Vosk time-to-final lag (4138 ms vs Deepgram 2352 ms)** | [`vosk-time-to-final-tuning`](.planning/todos/2026-05-16-vosk-time-to-final-tuning.md) | agent todo |
| **AEC ERLE not measured on real speaker→mic loop** | [`aec-real-loop-erle-measurement`](.planning/todos/2026-05-16-aec-real-loop-erle-measurement.md) | agent todo |
| **No real-mic integration test for the Vosk path** | [`vosk-real-mic-integration-test`](.planning/todos/2026-05-16-vosk-real-mic-integration-test.md) | operator todo |
| **No local TTS — ElevenLabs still required** | [`local-tts-exploration`](.planning/notes/2026-05-16-local-tts-exploration.md) (Piper candidate) | note, deferred |
| **No CI / automated regression gating** | [`ci-regression-harness`](.planning/notes/2026-05-16-ci-regression-harness.md) | note, deferred |
| **Speaker classifier latency 60 ms** | [`tier2-distilled-classifier-deferred`](.planning/notes/2026-05-16-tier2-distilled-classifier-deferred.md) | note, Tier 1 already at 100% so deferred |
| **Voiceprint cache per-machine** | [`by-design-tradeoffs`](.planning/notes/2026-05-16-by-design-tradeoffs.md) | by design (privacy) |
| **Voice dataset gitignored** | [`by-design-tradeoffs`](.planning/notes/2026-05-16-by-design-tradeoffs.md) | by design (billable artifact) |

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

### Scoped vs general benchmark — where we stand

Two distinct accuracy regimes apply to speaker classifiers:

| | **Open-set verification** | **Closed-set classification (ours)** |
|---|---|---|
| Task | Pick "same speaker?" yes/no from arbitrary pairs, neither speaker seen during training | Pick 1 of N enrolled speakers, all enrolled at runtime |
| Public benchmark | VoxCeleb-O (~37K trial pairs, ~7K identities) | none — every conversational system has its own corpus |
| Reference scores | ECAPA-TDNN ~0.86% EER, Resemblyzer ~5-6% EER, classical MFCC+GMM ~10-15% EER | (only meaningful for the specific scoped use case) |
| Difficulty | hard — model must generalize to unseen voices | much easier — both voices are enrolled |

Our pipeline solves the closed-set conversational case. The synthetic-corpus benchmark above measures that. To know where we stand on the **general** task — what the literature numbers describe — run:

```
python -m scripts.conduit_tui.benchmark \
    --pairs path/to/voxceleb_test_pairs.txt \
    --wav-dir path/to/voxceleb_wavs/
```

VoxCeleb test pairs require free registration at https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ (no auto-download). The harness reports EER and prints the reference scores from ECAPA-TDNN / Resemblyzer / classical MFCC-GMM so you can see the gap immediately. Expect our 29-dim handcrafted features to land in the ~15-25% EER range on open-set — well behind learned embeddings, but irrelevant to our actual gated-barge use case because we're solving the easier problem.

The point of running it: when we upgrade to Tier 1 (ONNX ECAPA-TDNN), the same harness re-runs and shows the EER drop AND the closed-set accuracy delta side by side. No more handwaving.

### Benchmarking — methodology and current numbers

Run:

```
python -m scripts.conduit_tui.benchmark
```

The harness reports two things:

1. **Classifier accuracy** on a synthetic-voice corpus (formant-synth user vs formant-synth bot, 5 unseen clips each). Clean speech, plus a stress condition where each clip has the bot's voice mixed in at −10 dB to simulate AEC residual.
2. **AEC ERLE** (echo return loss enhancement) in a pure-echo scenario, plus correlation against the user-only signal under simulated double-talk.

Last measured (Tier 0 / pure-numpy 58-dim + speexdsp / 2026-05-15):

| Metric | Result | Notes |
|---|---|---|
| Classifier accuracy — synthetic formant-synth (5-way) | **80%** | Misleadingly optimistic. The synthetic voices have artificially extreme F0 separation (120 Hz user vs 220 Hz bot) and exaggerated formant differences. |
| **Classifier accuracy — REAL ElevenLabs (5-way closed-set)** | **24%** | 12/50 correct, chance = 20%. ElevenLabs premade voices are all studio-quality narrators with overlapping prosody, formant ranges, and pitch — handcrafted MFCC-class features cannot separate them. **This is the real-world floor and it's not useful.** See `ERRORS-AND-ATTEMPTS.xml` → `classifier-tier0-fails-on-real-elevenlabs-2026-05-15` for the full finding. |
| Classifier same vs diff cos sim | 0.992 vs 0.993 | Same-voice and diff-voice pairs are statistically indistinguishable in our feature space. |
| AEC ERLE (pure echo, synthetic) | **−3.2 dB** | post-convergence on a single-tone reference. Higher (better) on real broadband speech — the standalone pyaec smoke test on richer signals hit ~11.5 dB. |
| Classifier latency | **5.36 ms** per call (500ms clip, CPU) | invoked once per Deepgram partial (~5×/sec) — negligible |
| AEC latency | **2.80 ms** per 1024-sample chunk (= 64ms audio) | 4.4% of real-time → ample headroom for STT/LLM/TTS network calls |

**Honest read:** Tier 0 classifier works only for the **binary user-vs-bot case where the two voices are dramatically different** (e.g., adult male user vs Sarah). For multi-bot scenarios in the upcoming meeting mode, Tier 1 (ONNX learned embeddings) is mandatory — phase 07 plans it. The AEC and barge state-machine fixes stand on their own and don't depend on speaker classification working perfectly.

The synthetic test is the floor, not the ceiling — handcrafted features and adaptive filters both benefit from spectral diversity that pure sinusoids don't provide. If you want a real number for your environment, pass actual recordings via `--user user.wav --bot bot.wav` (16-bit mono, any rate). Drop a 10-30 second clip of each into the harness and you'll get an honest accuracy / EER for your conditions.

### Why a smaller model can beat a bulkier one

The 500 MB resemblyzer footprint is mostly PyTorch + CUDA bindings + auxiliary numerics libraries, not the speaker model itself. The actual model weights are usually 5-20 MB. Several real reasons a smaller system can be **better** for our problem:

1. **Closed-set vs open-set.** ECAPA-TDNN (and friends) are trained to verify ANY two unseen speakers — pick 1 of 7000+ identities from a single 5-second clip. We have 2 known speakers, both enrolled. That's a fundamentally easier problem. A model with one-tenth the parameters can solve it.

2. **Runtime swap.** Take the same trained ECAPA-TDNN, convert it to ONNX (~10 MB), run it with `onnxruntime` (~30 MB) instead of PyTorch (~400 MB). Same accuracy, ~5% the install size.

3. **Quantization.** Float32 weights → int8 = 4x size reduction with single-digit accuracy loss for inference. ECAPA-TDNN-int8 fits in ~3 MB.

4. **Knowledge distillation.** Train a small "student" model to mimic a large "teacher's" outputs. DistilHuBERT achieves ~95% of HuBERT-base's performance at half the size. The student inherits accuracy it didn't earn during training.

5. **Architecture improvements.** Depthwise separable convolutions (MobileNet), squeeze-and-excitation blocks (the SE in ECAPA-SE-TDNN), conformer blocks. Often 10x smaller for the same accuracy as a 2017-era architecture.

6. **Pretrained representations + a tiny head.** Take a frozen SSL model (wav2vec2, WavLM) producing speech embeddings, slap a 2-class linear classifier on top. The head is ~1 KB; the embedding model carries the cost only once.

### Upgrade path

| Tier | Backend | Size | Expected acc on real voice | Effort |
|---|---|---|---|---|
| **0 (shipped)** | pure-numpy 29-dim MFCC + ΔMFCC + F0 + centroid + ZCR | 0 MB extra | ~85-90% (literature; measured 80% on adversarial synthetic) | done |
| **1** | ONNX-converted ECAPA-TDNN + `onnxruntime` | ~30-50 MB | ~95-98% EER | model conversion + ~50 lines wrapper code; same `extract_features` interface |
| **2** | int8-quantized distilled CNN, task-specific 2-class head | ~5-15 MB | similar to Tier 1 for our closed-set | training pipeline; biggest payoff if Tier 1 still leaves gaps |

The current implementation in `scripts/conduit_tui/speaker_id.py` keeps the same interface (`extract_features` → vector, `SpeakerClassifier.classify` → label/margin) that either tier slots into — `extract_features` becomes the ONNX forward pass; everything downstream is unchanged. The benchmark harness re-runs against the new backend and produces apples-to-apples numbers.

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
