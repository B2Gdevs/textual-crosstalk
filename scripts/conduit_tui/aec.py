"""
aec.py — Acoustic echo cancellation.

We have the rare luxury of knowing exactly what the bot is playing
(the decoded TTS PCM we just handed to sounddevice). That lets us run
a proper adaptive filter against the mic input and subtract the bot's
voice before it ever reaches Deepgram — the self-reply loop dissolves
at the audio layer, not the text layer.

Primary backend: pyaec (ctypes bindings around speexdsp's AEC +
preprocessor). Same algorithm used by FreeSWITCH, Asterisk, and
countless SIP clients — partitioned-block FDAF with double-talk
detection and residual echo suppression. Battle-tested over 20 years.
Prebuilt Windows wheel, no toolchain required.

Fallback backend: pure-numpy block NLMS adaptive filter. Used only if
pyaec is unavailable. Lower ERLE (echo return loss enhancement)
than speex but enough to suppress the self-reply loop.

Sample-rate normalization: mic is 16 kHz mono, ElevenLabs TTS decodes
to 44.1 kHz stereo. push_reference() handles both — channel-averages
to mono and linearly interpolates to mic_rate.

Performance budget (speex backend, 16kHz, 10ms frames):
    ~50μs per frame on a modern CPU → ~0.5% CPU.

Env knobs:
    AEC_FILTER_MS    speex tail length in ms (default 128 — covers up
                     to 128ms of room reverb + speaker latency)
    AEC_FRAME_MS     processing frame size in ms (default 10)

References:
    speexdsp AEC: https://www.speex.org/docs/manual/speex-manual/node7.html
    Haykin, Adaptive Filter Theory 5e, §10.2 (block NLMS fallback).
"""
from __future__ import annotations

import os

import numpy as np

try:
    import pyaec  # ctypes bindings → speexdsp libspeexdsp.dll
    _HAVE_PYAEC = True
except ImportError:
    pyaec = None  # type: ignore[assignment]
    _HAVE_PYAEC = False

try:
    import miniaudio  # decode mp3 → PCM for the reference path
except ImportError:
    miniaudio = None  # type: ignore[assignment]


def make_echo_canceller(mic_rate: int = 16000) -> "EchoCancellerBase":
    """Construct the best available AEC backend."""
    if _HAVE_PYAEC:
        return SpeexEchoCanceller(mic_rate=mic_rate)
    return NumpyEchoCanceller(mic_rate=mic_rate)


# ----------------------------------------------------------------------
# Common base


class EchoCancellerBase:
    """Shared interface for AEC backends."""

    def __init__(self, mic_rate: int) -> None:
        self.mic_rate = int(mic_rate)
        # Pending reference samples (mic-rate, int16) awaiting alignment
        # with mic input.
        self._ref_buf = np.zeros(0, dtype=np.int16)
        # Pending mic samples that arrived before their reference (rare;
        # only happens at TTS edges).
        self._mic_buf = np.zeros(0, dtype=np.int16)

    # Reference path -----------------------------------------------------

    def push_reference_mp3(self, mp3_bytes: bytes) -> None:
        """Decode an ElevenLabs mp3 chunk and push as reference."""
        if not mp3_bytes or miniaudio is None:
            return
        try:
            decoded = miniaudio.decode(mp3_bytes)
        except Exception as exc:
            print(f"[aec] mp3 decode failed: {exc}")
            return
        samples = np.frombuffer(decoded.samples, dtype=np.int16)
        if decoded.nchannels > 1:
            samples = samples.reshape(-1, decoded.nchannels)
        self.push_reference(samples, decoded.sample_rate)

    def push_reference(self, samples_int16: np.ndarray, src_rate: int) -> None:
        """Push a chunk of bot audio. Stereo→mono, resampled to mic rate."""
        if samples_int16.size == 0:
            return
        x = samples_int16
        if x.ndim == 2:
            x = x.mean(axis=1).astype(np.int16)
        if src_rate != self.mic_rate:
            x = _resample_linear_int16(x, src_rate, self.mic_rate)
        self._ref_buf = np.concatenate([self._ref_buf, x.astype(np.int16)])

    def clear_reference(self) -> None:
        self._ref_buf = np.zeros(0, dtype=np.int16)
        self._mic_buf = np.zeros(0, dtype=np.int16)

    # Mic path -----------------------------------------------------------

    def process(self, mic_int16: np.ndarray) -> np.ndarray:
        """Process a mic chunk; return AEC-cleaned int16."""
        raise NotImplementedError


# ----------------------------------------------------------------------
# Speex backend (primary)


class SpeexEchoCanceller(EchoCancellerBase):
    """pyaec-based echo canceller wrapping speexdsp's AEC + preprocessor."""

    def __init__(self, mic_rate: int = 16000) -> None:
        super().__init__(mic_rate)
        frame_ms = int(os.environ.get("AEC_FRAME_MS", "10"))
        tail_ms = int(os.environ.get("AEC_FILTER_MS", "128"))
        self.frame_size = int(mic_rate * frame_ms / 1000)
        self.filter_length = int(mic_rate * tail_ms / 1000)
        self._ec = pyaec.Aec(
            frame_size=self.frame_size,
            filter_length=self.filter_length,
            sample_rate=mic_rate,
            enable_preprocess=True,  # noise suppress + AGC + VAD
        )

    def process(self, mic_int16: np.ndarray) -> np.ndarray:
        F = self.frame_size
        # If the pending reference is shorter than one AEC frame, drop
        # it — it's a leftover tail from a TTS that just ended and we
        # cannot cancel a partial frame anyway. Without this, an
        # un-drainable residual would stall mic_buf forever and the
        # mic_pump would silently send nothing to Deepgram (user can't
        # be heard between turns).
        if 0 < self._ref_buf.size < F:
            self._ref_buf = np.zeros(0, dtype=np.int16)

        if self._ref_buf.size == 0 and self._mic_buf.size == 0:
            return mic_int16
        if self._ref_buf.size == 0:
            return self._flush_passthrough(mic_int16)

        self._mic_buf = np.concatenate([self._mic_buf, mic_int16])

        out_chunks: list[np.ndarray] = []
        while self._mic_buf.size >= F and self._ref_buf.size >= F:
            mic_frame = self._mic_buf[:F]
            ref_frame = self._ref_buf[:F]
            self._mic_buf = self._mic_buf[F:]
            self._ref_buf = self._ref_buf[F:]
            # pyaec returns a list of signed ints that are actually int8
            # byte values — sign-extended in the ctypes bridge. Convert
            # by masking to uint8 then reinterpreting as int16.
            raw = self._ec.cancel_echo(mic_frame.tobytes(), ref_frame.tobytes())
            b = bytes(v & 0xFF for v in raw)
            out_chunks.append(np.frombuffer(b, dtype=np.int16))

        # If the loop didn't run because ref_buf shrank below one frame
        # mid-processing, flush whatever's in mic_buf as pass-through so
        # we don't strand the user's voice in the queue.
        if not out_chunks and self._ref_buf.size < F:
            return self._flush_passthrough(np.zeros(0, dtype=np.int16))

        if not out_chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(out_chunks)

    def _flush_passthrough(self, mic_int16: np.ndarray) -> np.ndarray:
        if self._mic_buf.size == 0:
            return mic_int16
        out = np.concatenate([self._mic_buf, mic_int16])
        self._mic_buf = np.zeros(0, dtype=np.int16)
        return out


# ----------------------------------------------------------------------
# Numpy fallback


class NumpyEchoCanceller(EchoCancellerBase):
    """Block NLMS adaptive filter. Used if pyaec is unavailable."""

    def __init__(
        self,
        mic_rate: int = 16000,
        filter_ms: int | None = None,
        mu: float | None = None,
    ) -> None:
        super().__init__(mic_rate)
        if filter_ms is None:
            filter_ms = int(os.environ.get("AEC_FILTER_MS", "25"))
        if mu is None:
            mu = float(os.environ.get("AEC_MU", "0.3"))
        self.L = max(8, int(filter_ms * mic_rate / 1000))
        self.mu = float(mu)
        self.eps = 1e-6
        self.w = np.zeros(self.L, dtype=np.float32)
        self.ref_history = np.zeros(self.L - 1, dtype=np.float32)

    def process(self, mic_int16: np.ndarray) -> np.ndarray:
        # Drop residual reference shorter than the filter length — we
        # can't run NLMS without a full window, and leaving it queued
        # would stall the mic buffer forever (Deepgram would never see
        # user speech between turns).
        if 0 < self._ref_buf.size < self.L:
            self._ref_buf = np.zeros(0, dtype=np.int16)

        if self._ref_buf.size == 0 and self._mic_buf.size == 0:
            return mic_int16
        if self._ref_buf.size == 0:
            return self._flush_passthrough(mic_int16)

        self._mic_buf = np.concatenate([self._mic_buf, mic_int16])
        n = int(min(self._mic_buf.size, self._ref_buf.size))
        if n < self.L:
            # Same defense — flush mic to passthrough if we can't form
            # a window. Drop the residual reference.
            self._ref_buf = np.zeros(0, dtype=np.int16)
            return self._flush_passthrough(np.zeros(0, dtype=np.int16))

        mic_block = self._mic_buf[:n].astype(np.float32) / 32768.0
        ref_block = self._ref_buf[:n].astype(np.float32) / 32768.0

        full_ref = np.concatenate([self.ref_history, ref_block])
        R = np.lib.stride_tricks.sliding_window_view(full_ref, self.L)

        y_hat = R @ self.w
        e = mic_block - y_hat

        ref_power = float(np.dot(ref_block, ref_block)) / max(n, 1)
        denom = n * (ref_power + self.eps)
        self.w = self.w + (self.mu / denom) * (R.T @ e)

        self.ref_history = full_ref[-(self.L - 1):]
        self._mic_buf = self._mic_buf[n:]
        self._ref_buf = self._ref_buf[n:]
        return np.clip(e * 32768.0, -32768, 32767).astype(np.int16)

    def _flush_passthrough(self, mic_int16: np.ndarray) -> np.ndarray:
        if self._mic_buf.size == 0:
            return mic_int16
        out = np.concatenate([self._mic_buf, mic_int16])
        self._mic_buf = np.zeros(0, dtype=np.int16)
        return out


# Default alias kept for backwards compatibility with earlier wiring.
EchoCanceller = SpeexEchoCanceller if _HAVE_PYAEC else NumpyEchoCanceller


# ----------------------------------------------------------------------
# Helpers


def _resample_linear_int16(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interp resample int16 → int16 at a new rate."""
    if src_rate == dst_rate:
        return x
    src_n = int(x.size)
    if src_n == 0:
        return np.zeros(0, dtype=np.int16)
    dst_n = max(1, int(round(src_n * dst_rate / src_rate)))
    src_t = np.linspace(0.0, 1.0, src_n, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, dst_n, dtype=np.float32)
    out = np.interp(dst_t, src_t, x.astype(np.float32))
    return np.clip(out, -32768, 32767).astype(np.int16)
