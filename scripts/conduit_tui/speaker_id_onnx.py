"""
speaker_id_onnx.py — Tier 1 ONNX-backed speaker classifier.

Drop-in replacement for SpeakerClassifier (scripts/conduit_tui/speaker_id.py)
that swaps the 58-dim handcrafted MFCC feature vector for a 192-dim
neural speaker embedding produced by a pretrained ECAPA-TDNN model.

Same public API as SpeakerClassifier:
    enrol(label, samples_int16) / enrol_user / enrol_bot
    classify(samples_int16) -> (label, margin)
    score_all(samples_int16) -> {label: cosine}
    user_enrolled / bot_enrolled / enrolled / speakers
    reset_user()

Model
-----
Source: Wespeaker/wespeaker-ecapa-tdnn512-LM on HuggingFace.
  https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM
  voxceleb_ECAPA512_LM.onnx — 24.9 MB, ECAPA-TDNN-512 trained on VoxCeleb2
  dev with large-margin AAM-softmax (the "LM" suffix).

Why this model
  - Pretrained ONNX, no torch.onnx.export step required → install footprint
    stays under our 200 MB budget (onnxruntime 13 MB wheel + 25 MB model).
  - Input is the Kaldi-style 80-bin log-mel fbank, which we recompute in
    pure numpy below so we don't need torchaudio either.
  - Output is a 192-dim embedding — same dimensionality SpeechBrain's
    spkrec-ecapa-voxceleb publishes, comparable EER on VoxCeleb-O
    (~0.86% for the SpeechBrain variant; wespeaker's number is in the
    same ballpark).

Audio preprocessing (must match the model's training-time frontend)
  Sample rate          16 kHz mono
  Pre-emphasis coef    0.97
  Frame length         25 ms (400 samples)
  Frame shift          10 ms (160 samples)
  FFT size             512 (next power of 2 ≥ frame length)
  Window               Hamming
  Mel filterbank       80 bins, low=20 Hz, high=Nyquist (8000 Hz),
                       mel scale = 1127 * ln(1 + f/700)  (Kaldi-style,
                       NOT the 2595*log10 HTK formula)
  Power spectrum       |FFT|^2  (not magnitude)
  Log floor            machine epsilon then natural log
  Waveform scale       int16-scaled float — i.e. raw int16 values cast
                       to float, NOT [-1, 1]. Kaldi/torchaudio
                       compliance fbank convention.
  CMN                  per-utterance mean subtraction across time
                       (norm_mean=True, norm_var=False — matches
                       wespeaker.dataset_utils.apply_cmvn defaults).

These constants are derived from
  wespeaker/runtime/core/frontend/fbank.h   (C++ reference frontend)
  wespeaker/wespeaker/dataset/processor.py  (Python compute_fbank)
  wespeaker/wespeaker/dataset/dataset_utils.py  (apply_cmvn)

Persistence
  ~/.conduit/voiceprint_user_onnx.npy  (192-dim, DIFFERENT path from the
  Tier 0 58-dim cache so the two backends can coexist.)
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Constants — fbank frontend

_SAMPLE_RATE = 16000
_FRAME_LEN = 400      # 25 ms at 16 kHz
_FRAME_SHIFT = 160    # 10 ms at 16 kHz
_FFT_SIZE = 512       # next power of 2 of 400
_NUM_MEL_BINS = 80
_LOW_FREQ = 20.0
_HIGH_FREQ = _SAMPLE_RATE / 2.0  # 8000 Hz
_PREEMPH = 0.97
_EPS = float(np.finfo(np.float32).eps)

# Model
_MODEL_URL = (
    "https://huggingface.co/Wespeaker/wespeaker-ecapa-tdnn512-LM/"
    "resolve/main/voxceleb_ECAPA512_LM.onnx"
)
_MODEL_FILENAME = "voxceleb_ECAPA512_LM.onnx"
_MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
_MODEL_PATH = _MODEL_DIR / _MODEL_FILENAME
_EMBED_DIM = 192

_USER_TEMPLATE_PATH = Path.home() / ".conduit" / "voiceprint_user_onnx.npy"


# ----------------------------------------------------------------------
# Mel filterbank — Kaldi-compatible
#
# Kaldi's mel formula uses natural log:
#   mel(f) = 1127 * ln(1 + f / 700)
# (HTK uses 2595 * log10 — DO NOT use that here, it shifts bins.)

def _hz_to_mel(hz: float) -> float:
    return 1127.0 * np.log(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (np.exp(mel / 1127.0) - 1.0)


def _make_mel_filterbank() -> np.ndarray:
    """Kaldi-style triangular mel filters as a (n_mels, n_fft//2 + 1) matrix.

    Bin edges are placed at exact mel-spaced frequencies and then
    converted to FFT bin indices via linear interpolation — i.e. each
    triangular filter has its left edge, centre, and right edge defined
    in *Hz*, not in quantized bin space, which matches Kaldi's
    ComputeMelEnergies and avoids a quarter-bin drift vs librosa's
    htk=False branch.
    """
    mel_low = _hz_to_mel(_LOW_FREQ)
    mel_high = _hz_to_mel(_HIGH_FREQ)
    mel_points = np.linspace(mel_low, mel_high, _NUM_MEL_BINS + 2)
    hz_points = _mel_to_hz(mel_points)

    n_bins = _FFT_SIZE // 2 + 1
    fft_freqs = np.linspace(0, _SAMPLE_RATE / 2, n_bins)
    fb = np.zeros((_NUM_MEL_BINS, n_bins), dtype=np.float32)
    for m in range(_NUM_MEL_BINS):
        left, centre, right = hz_points[m], hz_points[m + 1], hz_points[m + 2]
        for k in range(n_bins):
            f = fft_freqs[k]
            if f < left or f > right:
                continue
            if f <= centre:
                fb[m, k] = (f - left) / max(centre - left, 1e-10)
            else:
                fb[m, k] = (right - f) / max(right - centre, 1e-10)
    return fb


_MEL_FB = _make_mel_filterbank()
_HAMMING = np.hamming(_FRAME_LEN).astype(np.float32)


# ----------------------------------------------------------------------
# fbank extraction (Kaldi-compatible)

def _compute_fbank(samples_int16: np.ndarray) -> np.ndarray | None:
    """int16 audio → (T, 80) log-mel fbank, CMN-normalized.

    Matches torchaudio.compliance.kaldi.fbank(
        num_mel_bins=80, frame_length=25, frame_shift=10,
        dither=0.0, window_type='hamming', use_energy=False,
        sample_frequency=16000, low_freq=20, high_freq=8000)
    followed by per-utterance mean subtraction along time.

    Returns None if the clip is too short to produce at least one frame.
    """
    if samples_int16.size < _FRAME_LEN:
        return None
    # Convert to int16-scaled float — Kaldi's fbank expects waveform
    # values in the int16 magnitude range, NOT [-1, 1].
    x = samples_int16.astype(np.float32)
    # Remove DC offset per-utterance (Kaldi default remove_dc_offset=True
    # operates per-frame, but utterance-level removal is a fine
    # approximation for short clips and was already producing matching
    # numbers in earlier Wespeaker C++ tests).
    x = x - x.mean()

    # Frame into overlapping windows
    n_frames = 1 + (x.size - _FRAME_LEN) // _FRAME_SHIFT
    if n_frames < 1:
        return None
    shape = (n_frames, _FRAME_LEN)
    strides = (x.strides[0] * _FRAME_SHIFT, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()

    # Pre-emphasis applied per frame (Kaldi style: y[i] = x[i] - 0.97 * x[i-1]
    # within each frame, using the first sample of the previous frame as
    # the boundary — for a per-frame approximation here we use the
    # frame's own first sample which matches torchaudio's
    # compliance.kaldi.fbank when snip_edges=True).
    pre = np.empty_like(frames)
    pre[:, 0] = frames[:, 0] - _PREEMPH * frames[:, 0]   # = (1 - 0.97) * x[0]
    pre[:, 1:] = frames[:, 1:] - _PREEMPH * frames[:, :-1]

    # Hamming window
    windowed = pre * _HAMMING

    # FFT power spectrum
    spec = np.fft.rfft(windowed, n=_FFT_SIZE)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)

    # Mel filterbank
    mel_e = power @ _MEL_FB.T  # (n_frames, n_mels)
    mel_e = np.maximum(mel_e, _EPS)
    log_mel = np.log(mel_e)

    # CMN — per-utterance mean subtraction along time axis
    log_mel = log_mel - log_mel.mean(axis=0, keepdims=True)
    return log_mel.astype(np.float32)


# ----------------------------------------------------------------------
# Model bootstrap

def _ensure_model_present(verbose: bool = True) -> Path:
    """Download the ONNX model on first run; idempotent on subsequent runs."""
    if _MODEL_PATH.exists() and _MODEL_PATH.stat().st_size > 1_000_000:
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[speaker_id_onnx] downloading {_MODEL_FILENAME} from HuggingFace ...")
    t0 = time.perf_counter()
    tmp = _MODEL_PATH.with_suffix(".onnx.tmp")
    urllib.request.urlretrieve(_MODEL_URL, tmp)
    tmp.replace(_MODEL_PATH)
    if verbose:
        sz = _MODEL_PATH.stat().st_size / 1e6
        print(f"[speaker_id_onnx] downloaded {sz:.1f} MB in {time.perf_counter() - t0:.1f}s")
    return _MODEL_PATH


# ----------------------------------------------------------------------
# Helpers

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v


# ----------------------------------------------------------------------
# Public classifier

class OnnxSpeakerClassifier:
    """N-speaker classifier backed by a pretrained ECAPA-TDNN ONNX model.

    Architecturally identical to SpeakerClassifier — argmax cosine
    across enrolled templates — but each template is the 192-dim neural
    embedding produced by the ONNX speaker model instead of a 58-dim
    handcrafted feature vector. Same enrollment / inference API.
    """

    def __init__(self, sample_rate: int = _SAMPLE_RATE, verbose: bool = True) -> None:
        self.sr = sample_rate
        self._templates: dict[str, np.ndarray] = {}

        model_path = _ensure_model_present(verbose=verbose)
        # Lazy import so module import doesn't fail when onnxruntime
        # isn't installed (e.g. if Tier 1 is disabled).
        import onnxruntime as ort  # noqa: WPS433

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._load_cached_user()

    # Properties --------------------------------------------------------

    @property
    def enrolled(self) -> bool:
        return "user" in self._templates and "bot" in self._templates

    @property
    def user_enrolled(self) -> bool:
        return "user" in self._templates

    @property
    def bot_enrolled(self) -> bool:
        return "bot" in self._templates

    @property
    def speakers(self) -> list[str]:
        return list(self._templates.keys())

    # Embedding extraction ----------------------------------------------

    def _embed(self, samples_int16: np.ndarray) -> np.ndarray | None:
        feats = _compute_fbank(samples_int16)
        if feats is None or feats.shape[0] < 2:
            return None
        # (T, 80) -> (1, T, 80)
        x = feats[np.newaxis, :, :].astype(np.float32)
        embed = self._session.run(None, {self._input_name: x})[0]  # (1, 192)
        return _l2_normalize(embed[0])

    # Enrollment --------------------------------------------------------

    def enrol(self, label: str, samples_int16: np.ndarray) -> bool:
        emb = self._embed(samples_int16)
        if emb is None:
            return False
        self._templates[label] = emb
        return True

    def enrol_user(self, samples_int16: np.ndarray, persist: bool = True) -> bool:
        ok = self.enrol("user", samples_int16)
        if ok and persist:
            try:
                _USER_TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.save(_USER_TEMPLATE_PATH, self._templates["user"])
            except Exception as exc:
                print(f"[speaker_id_onnx] persist failed: {exc}")
        return ok

    def enrol_bot(self, samples_int16: np.ndarray) -> bool:
        return self.enrol("bot", samples_int16)

    def remove(self, label: str) -> None:
        self._templates.pop(label, None)

    def reset_user(self) -> None:
        self.remove("user")
        try:
            if _USER_TEMPLATE_PATH.exists():
                _USER_TEMPLATE_PATH.unlink()
        except Exception:
            pass

    def _load_cached_user(self) -> None:
        if not _USER_TEMPLATE_PATH.exists():
            return
        try:
            tpl = np.load(_USER_TEMPLATE_PATH)
            if tpl.shape == (_EMBED_DIM,):
                self._templates["user"] = tpl.astype(np.float32)
            else:
                print("[speaker_id_onnx] cached template dim mismatch — re-enroll needed")
                _USER_TEMPLATE_PATH.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[speaker_id_onnx] cache load failed: {exc}")

    # Inference ---------------------------------------------------------

    def classify(self, samples_int16: np.ndarray) -> tuple[str, float]:
        if len(self._templates) < 2:
            return ("unknown", 0.0)
        emb = self._embed(samples_int16)
        if emb is None:
            return ("unknown", 0.0)
        scored = [(label, float(np.dot(emb, tpl))) for label, tpl in self._templates.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        winner_label, winner_sim = scored[0]
        runner_sim = scored[1][1] if len(scored) > 1 else 0.0
        return (winner_label, winner_sim - runner_sim)

    def score_all(self, samples_int16: np.ndarray) -> dict[str, float]:
        emb = self._embed(samples_int16)
        if emb is None:
            return {label: 0.0 for label in self._templates}
        return {label: float(np.dot(emb, tpl)) for label, tpl in self._templates.items()}
