---
name: aec-before-text-echo-guards
description: When a voice-AI pipeline replies to its own audio bleeding back through the mic, fix it at the audio layer with AEC (pyaec / speexdsp) — never at the text layer. STT hallucinates novel words from broadband noise; one hallucinated word slips any text-similarity check. Use this skill whenever building or debugging a streaming voice-LLM loop where the bot's TTS plays through speakers and the mic captures user speech.
metadata:
  type: project
  domain: audio
  source: conduit_proj phase 04 E3 + D-017
---

# Fix the self-reply loop at the audio layer, not the text layer

## When this applies

You have a voice-AI pipeline shaped like:

```
mic → STT → LLM → TTS → speaker
                          ↓ (acoustic echo)
                         mic
```

The bot starts replying to its own audio. First instinct is to filter at the text layer — "if the partial matches what I just said, ignore it". This is a trap. **Three rounds of text-layer fixes were tried and all failed:**

1. **Substring match against bot text** — over-suppresses the user when their vocabulary overlaps with the bot's.
2. **Word-overlap threshold (60% of partial words in bot text)** — same problem in milder form.
3. **All-partial-words-in-bot-text** — failed because Deepgram (and any probabilistic STT) occasionally hallucinates a novel word from coughs / breath / room noise. One hallucinated word slips the check, the "echo" transcript enters the LLM cycle, the bot replies to itself.

## The rule

**If you have the bot's reference signal (you played it — you have it), do AEC.** Acoustic echo cancellation runs an adaptive filter against the mic input using the known reference. Output is mic-minus-echo. STT sees clean residual.

Text-content filters are a structural fallback at best, never primary.

## The shortest working implementation (Python, prebuilt wheels, no compile)

```python
# requirements:
# pyaec>=1.0      # ctypes bindings around speexdsp's AEC + preprocessor
# miniaudio>=1.59 # mp3 decode if your TTS returns mp3
# numpy<2.0
```

```python
import pyaec, numpy as np

class EchoCanceller:
    """Speex AEC wrapper. process(mic) → cleaned mic, ready for STT."""

    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.F = sample_rate // 100  # 10ms frames
        self._ec = pyaec.Aec(
            frame_size=self.F,
            filter_length=sample_rate * 128 // 1000,  # 128ms tail
            sample_rate=sample_rate,
            enable_preprocess=True,  # KEEP ON — see Pitfalls below
        )
        self._ref = np.zeros(0, dtype=np.int16)
        self._mic = np.zeros(0, dtype=np.int16)

    def push_reference(self, pcm_int16):
        """Call BEFORE play(): hand the bot audio in at mic rate, mono."""
        self._ref = np.concatenate([self._ref, pcm_int16])

    def clear_reference(self):
        """Call on barge / TTS end: drop pending reference."""
        self._ref = np.zeros(0, dtype=np.int16)
        self._mic = np.zeros(0, dtype=np.int16)

    def process(self, mic_int16):
        # If pending ref is shorter than one AEC frame, drop it —
        # never let an undrainable residual stall the mic queue.
        if 0 < self._ref.size < self.F:
            self._ref = np.zeros(0, dtype=np.int16)
        if self._ref.size == 0:
            if self._mic.size:
                out = np.concatenate([self._mic, mic_int16])
                self._mic = np.zeros(0, dtype=np.int16)
                return out
            return mic_int16
        self._mic = np.concatenate([self._mic, mic_int16])
        out = []
        while self._mic.size >= self.F and self._ref.size >= self.F:
            mf, rf = self._mic[:self.F], self._ref[:self.F]
            self._mic, self._ref = self._mic[self.F:], self._ref[self.F:]
            raw = self._ec.cancel_echo(mf.tobytes(), rf.tobytes())
            # pyaec returns signed ints that are actually uint8 byte values
            b = bytes(v & 0xFF for v in raw)
            out.append(np.frombuffer(b, dtype=np.int16))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int16)
```

Wiring is two lines:

```python
# 1. In the mic→STT pump:
clean = aec.process(mic_chunk)
stt.send(clean.tobytes())

# 2. Before play_audio:
aec.push_reference(decoded_bot_pcm)   # mono, mic rate, int16
play(bot_audio)
# After play_audio (try/finally):
aec.clear_reference()
```

## Pitfalls (each one cost real time in phase 04)

1. **Don't disable preprocess.** `enable_preprocess=True` is the speex preprocessor — noise suppress + AGC + VAD + residual echo suppression. Turning it off was a wrong fix for an unrelated turn-boundary bug; without it the AEC output had max-abs values clipping the int16 range (E7 + D-020).

2. **Drain sub-frame residuals.** The framed loop `while mic_buf >= F and ref_buf >= F` will never re-enter when `ref_buf` has, say, 50 leftover samples from the end of a TTS. Mic samples will accumulate forever and the user appears mute between turns. Drop residuals shorter than one frame at process-time AND clear reference explicitly when TTS ends (E4).

3. **State-machine gate symmetry.** If you ALSO have a "drop STT finals during TTS" gate (defensible safety belt), make sure the barge path opens it AND no other path silently re-closes it. The TTS `finally` branch is the usual culprit: it sets `gate = monotonic + cooldown` unconditionally, overwriting the open state the barge just set, swallowing the user's continuation (E6). Pattern: "only close the gate if it's currently in the future."

4. **PCM format gotchas.** ElevenLabs `convert_with_timestamps` returns `audio_base_64` (with underscore — not `audio_base64`). Free-tier requires `eleven_flash_v2_5` + `mp3_44100_128`. Decode mp3 → 44.1k stereo → mean-to-mono → linear-interp to your mic rate before pushing as AEC reference.

## How to verify it's working

Build a benchmark harness BEFORE writing the accuracy claim ([[benchmark-before-accuracy-claim]] when that skill exists at the framework level):

```python
# scripts/conduit_tui/benchmark.py — simplified
ref = synth_voice(duration=3.0, f0=220, formants=[(700, 0.2), (1700, 0.13)])
echo = np.zeros_like(ref); echo[80:] = (ref[:-80] * 0.5).astype(np.int16)
aec.push_reference(ref, src_rate=SR)
cleaned = []
for i in range(0, echo.size - 1024, 1024):
    cleaned.append(aec.process(echo[i:i+1024]))
out = np.concatenate(cleaned)
erle_db = 20 * math.log10(rms(echo[-1000:]) / rms(out[-1000:]))
# Expect ~10-20 dB on broadband speech; ~3 dB on pure tones (adversarial).
```

ERLE (Echo Return Loss Enhancement) is the standard metric. Positive dB = echo is suppressed. Pure-tone tests under-state real-world performance because adaptive filters thrive on spectral diversity.

## See also

- `scripts/conduit_tui/aec.py` in conduit_proj — the production version with both speex primary and pure-numpy NLMS fallback.
- ERRORS-AND-ATTEMPTS entries `bot-replies-to-itself-via-stt-echo-2026-05-15`, `user-silenced-after-first-bot-turn-2026-05-15`, `post-barge-cooldown-reclosed-gate-2026-05-15` for the full failure chain.
- Decision `CONDUIT-PROJ-D-017` for the architectural rationale.
