"""
benchmark.py — Measurements for the speaker classifier and AEC.

Run:
    python -m scripts.conduit_tui.benchmark

What this reports:
    SpeakerClassifier
      - accuracy on clean utterances (user vs bot, both seen and unseen)
      - accuracy under additive echo (user voice + attenuated bot voice)
      - Equal Error Rate (EER) — single-threshold operating point
      - confusion matrix
      - mean / std of decision margin per class

    AEC
      - Echo Return Loss Enhancement (ERLE) — pure-echo case
      - residual leakage during double-talk (user + bot)

This is a closed-set evaluation (2 enrolled speakers, both known at
runtime) which is what our conversational pipeline actually solves —
distinct from the open-set verification problem the public benchmarks
(VoxCeleb EER, etc.) optimize for. The deltas reported here are
specific to our 29-dim handcrafted feature set and our TTS voice;
when we swap in an ONNX speaker embedding (Tier 1 in the roadmap)
this same harness re-runs and produces an apples-to-apples comparison.

Synthetic test signals approximate human voice spectra: F0 + harmonics
+ formant shaping + breath noise. They're a stand-in for a real
recorded test corpus — substitute paths to wav files via the --user
and --bot flags if you have one.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import string
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.conduit_tui.aec import make_echo_canceller
from scripts.conduit_tui.speaker_id import SpeakerClassifier, extract_features


SR = 16000
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "data" / "dataset" / "manifest.json"
_USER_DIR = _REPO_ROOT / "data" / "dataset" / "user"


# ----------------------------------------------------------------------
# Synthetic voice generation


def _synth_voice(
    duration_s: float,
    f0: float,
    formants: list[tuple[float, float]],  # (Hz, amplitude)
    breath: float = 0.05,
    seed: int = 0,
) -> np.ndarray:
    """Crude formant-synthesis voice: F0 + harmonics + formants + noise."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    sig = 0.4 * np.sin(2 * np.pi * f0 * t)
    # Harmonics at 2f0, 3f0, 4f0 with decreasing amplitude
    for h, amp in enumerate([0.25, 0.15, 0.08, 0.04], start=2):
        sig += amp * np.sin(2 * np.pi * f0 * h * t)
    # Formants
    for hz, amp in formants:
        sig += amp * np.sin(2 * np.pi * hz * t + rng.uniform(0, 2 * np.pi))
    # Breath
    sig += breath * rng.standard_normal(n)
    # Light envelope to make it less robotic
    env = 0.85 + 0.15 * np.sin(2 * np.pi * 5 * t)
    sig *= env
    return (sig / np.max(np.abs(sig)) * 8000).astype(np.int16)


def _load_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        if sw != 2:
            raise ValueError(f"{path}: expected 16-bit, got {sw * 8}-bit")
        data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1).astype(np.int16)
    if sr != SR:
        ratio = SR / sr
        n_out = max(1, int(data.size * ratio))
        src_t = np.linspace(0.0, 1.0, data.size, dtype=np.float32)
        dst_t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
        data = np.interp(dst_t, src_t, data.astype(np.float32)).astype(np.int16)
    return data


# ----------------------------------------------------------------------
# Test corpus


def _build_synth_corpus(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[tuple[str, np.ndarray]]]:
    """Returns (user_enrollment, bot_enrollment, [(label, clip)...])."""
    # User: adult male-ish — F0 around 110-130 Hz, lower formants
    user_enrol = _synth_voice(
        duration_s=2.0,
        f0=120,
        formants=[(450, 0.18), (1200, 0.10), (2400, 0.05)],
        seed=seed,
    )
    # Bot ≈ ElevenLabs Sarah (female, warm) — F0 around 200-240 Hz,
    # brighter formants
    bot_enrol = _synth_voice(
        duration_s=2.0,
        f0=220,
        formants=[(700, 0.20), (1700, 0.13), (3200, 0.07)],
        seed=seed + 1,
    )

    clips: list[tuple[str, np.ndarray]] = []

    # Multiple "unseen" user variations — small F0 drift, formant shifts
    for i, (f0, fs) in enumerate([
        (115, [(450, 0.18), (1200, 0.10), (2400, 0.05)]),
        (125, [(440, 0.18), (1220, 0.10), (2380, 0.05)]),
        (118, [(460, 0.18), (1180, 0.10), (2410, 0.05)]),
        (130, [(450, 0.18), (1200, 0.10), (2450, 0.05)]),
        (108, [(440, 0.18), (1190, 0.10), (2400, 0.05)]),
    ]):
        clips.append(("user", _synth_voice(1.0, f0, fs, seed=seed + 10 + i)))

    # Multiple unseen bot variations
    for i, (f0, fs) in enumerate([
        (215, [(700, 0.20), (1700, 0.13), (3200, 0.07)]),
        (225, [(710, 0.20), (1680, 0.13), (3220, 0.07)]),
        (220, [(690, 0.20), (1720, 0.13), (3180, 0.07)]),
        (235, [(700, 0.20), (1690, 0.13), (3210, 0.07)]),
        (210, [(720, 0.20), (1700, 0.13), (3200, 0.07)]),
    ]):
        clips.append(("bot", _synth_voice(1.0, f0, fs, seed=seed + 100 + i)))

    return user_enrol, bot_enrol, clips


def _mix(a: np.ndarray, b: np.ndarray, b_scale: float = 0.5) -> np.ndarray:
    """Add b to a with attenuation. Lengths matched to min."""
    n = min(a.size, b.size)
    out = a[:n].astype(np.int32) + (b[:n].astype(np.int32) * b_scale).astype(np.int32)
    return np.clip(out, -32768, 32767).astype(np.int16)


# ----------------------------------------------------------------------
# Metrics


@dataclass
class ClassifierResult:
    n: int
    n_correct: int
    confusion: dict[tuple[str, str], int]
    margins_user: list[float]
    margins_bot: list[float]


def _eval_classifier(
    clf: SpeakerClassifier, clips: list[tuple[str, np.ndarray]]
) -> ClassifierResult:
    confusion: dict[tuple[str, str], int] = {}
    margins_user: list[float] = []
    margins_bot: list[float] = []
    correct = 0
    for true_label, clip in clips:
        pred, margin = clf.classify(clip)
        confusion[(true_label, pred)] = confusion.get((true_label, pred), 0) + 1
        if pred == true_label:
            correct += 1
        (margins_user if pred == "user" else margins_bot).append(margin)
    return ClassifierResult(
        n=len(clips), n_correct=correct, confusion=confusion,
        margins_user=margins_user, margins_bot=margins_bot,
    )


def _equal_error_rate(scores_user: list[float], scores_bot: list[float]) -> float:
    """EER over the margin score. Sweeps threshold; finds where false
    accept rate ≈ false reject rate. Both arrays are signed margins —
    bot scores are negated so 'user >= threshold' means accept-as-user."""
    if not scores_user or not scores_bot:
        return float("nan")
    s_user = np.array(scores_user, dtype=np.float64)
    s_bot = -np.array(scores_bot, dtype=np.float64)
    all_scores = np.concatenate([s_user, s_bot])
    thresholds = np.unique(np.concatenate([all_scores, [all_scores.min() - 1, all_scores.max() + 1]]))
    best = (float("inf"), float("nan"))
    for t in thresholds:
        fa = float(np.sum(s_bot >= t)) / s_bot.size  # accepts of bot
        fr = float(np.sum(s_user < t)) / s_user.size  # rejects of user
        gap = abs(fa - fr)
        if gap < best[0]:
            best = (gap, (fa + fr) / 2.0)
    return best[1]


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


# ----------------------------------------------------------------------
# Benchmarks


def benchmark_classifier(verbose: bool = True) -> None:
    print("\n=== Speaker Classifier Benchmark ===")
    user_enrol, bot_enrol, clips = _build_synth_corpus()

    clf = SpeakerClassifier(sample_rate=SR)
    clf.reset_user()
    clf.enrol_user(user_enrol, persist=False)
    clf.enrol_bot(bot_enrol)

    print(f"Enrolled: user={clf.user_enrolled}, bot={clf.bot_enrolled}")
    print(f"Test corpus: {len(clips)} synthetic clips "
          f"({sum(1 for l,_ in clips if l=='user')} user, "
          f"{sum(1 for l,_ in clips if l=='bot')} bot)")

    # Clean-speech accuracy
    res = _eval_classifier(clf, clips)
    acc = res.n_correct / max(res.n, 1)
    print(f"\n[clean] accuracy: {res.n_correct}/{res.n} = {acc*100:.1f}%")
    print(f"[clean] confusion:")
    for true_label in ["user", "bot"]:
        for pred_label in ["user", "bot", "unknown"]:
            count = res.confusion.get((true_label, pred_label), 0)
            if count:
                print(f"    true={true_label} pred={pred_label} : {count}")

    # Echo-stress: user speech + attenuated bot in same clip
    stress = []
    for true_label, clip in clips:
        if true_label == "user":
            # Mix user with bot at -10 dB (residual after AEC) — should still classify as user
            interferer = bot_enrol[: clip.size]
            mixed = _mix(clip, interferer, b_scale=0.3)
            stress.append(("user", mixed))
        else:
            stress.append(("bot", clip))
    stress_res = _eval_classifier(clf, stress)
    stress_acc = stress_res.n_correct / max(stress_res.n, 1)
    print(f"\n[user+bot mix @ -10dB] accuracy: "
          f"{stress_res.n_correct}/{stress_res.n} = {stress_acc*100:.1f}%")

    # EER over all clips
    # Collect signed margins where +ve means classified as user
    user_scores = []
    bot_scores = []
    for true_label, clip in clips:
        pred, m = clf.classify(clip)
        signed = m if pred == "user" else -m
        if true_label == "user":
            user_scores.append(signed)
        else:
            bot_scores.append(signed)
    eer = _equal_error_rate(user_scores, bot_scores)
    print(f"\n[EER] equal error rate: {eer*100:.1f}%")

    if verbose:
        print(f"\n[margins] user: mean={np.mean(res.margins_user):.4f} "
              f"std={np.std(res.margins_user):.4f}")
        print(f"[margins] bot:  mean={np.mean(res.margins_bot):.4f} "
              f"std={np.std(res.margins_bot):.4f}")


def benchmark_aec() -> None:
    print("\n=== AEC Benchmark ===")
    aec = make_echo_canceller(mic_rate=SR)
    print(f"backend: {type(aec).__name__}")

    duration = 3.0
    # Reference signal: bot speech-like (rich spectrum)
    ref = _synth_voice(duration, f0=220, formants=[(700, 0.2), (1700, 0.13), (3200, 0.07)])
    # Echo path: 0.5 attenuation, 80-sample (5ms) delay
    echo = np.zeros_like(ref)
    echo[80:] = (ref[:-80] * 0.5).astype(np.int16)

    # Pure-echo case
    aec.push_reference(ref, src_rate=SR)
    out_chunks = []
    chunk_size = 1024
    for i in range(0, len(echo) - chunk_size + 1, chunk_size):
        cleaned = aec.process(echo[i:i + chunk_size])
        out_chunks.append(cleaned)
    cleaned = np.concatenate(out_chunks) if out_chunks else np.zeros(0, dtype=np.int16)
    # Compute ERLE over the back half (post-convergence)
    half = max(1, len(cleaned) // 2)
    echo_post = echo[len(echo) - len(cleaned) + half : len(echo) - len(cleaned) + len(cleaned)]
    cleaned_post = cleaned[half:]
    if _rms(cleaned_post) > 0:
        erle = 20 * math.log10(_rms(echo_post) / _rms(cleaned_post))
        print(f"[pure echo] ERLE post-convergence: {erle:.1f} dB")
    else:
        print("[pure echo] full suppression (cleaned RMS = 0)")

    # Double-talk: user voice + bot echo
    aec_dt = make_echo_canceller(mic_rate=SR)
    aec_dt.push_reference(ref, src_rate=SR)
    user_voice = _synth_voice(duration, f0=120, formants=[(450, 0.18), (1200, 0.10), (2400, 0.05)])
    mic_dt = _mix(user_voice, echo, b_scale=1.0)
    dt_out = []
    for i in range(0, len(mic_dt) - chunk_size + 1, chunk_size):
        dt_out.append(aec_dt.process(mic_dt[i:i + chunk_size]))
    dt_cleaned = np.concatenate(dt_out) if dt_out else np.zeros(0, dtype=np.int16)
    user_post = user_voice[len(user_voice) - len(dt_cleaned):]
    if dt_cleaned.size and user_post.size:
        # We want dt_cleaned ≈ user_post (bot removed, user preserved).
        # Correlation as a proxy.
        n = min(dt_cleaned.size, user_post.size)
        a = dt_cleaned[:n].astype(np.float64)
        b = user_post[:n].astype(np.float64)
        a -= a.mean(); b -= b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        corr = float((a * b).sum() / denom) if denom > 0 else 0.0
        print(f"[double-talk] correlation(cleaned, user-only): {corr:.3f} "
              "(closer to 1.0 = bot well removed, user preserved)")


def benchmark_open_set_pairs(pairs_path: Path, wav_dir: Path) -> None:
    """Open-set verification benchmark on a VoxCeleb-style pairs file.

    Each line of `pairs_path` is "<label> <wav_a> <wav_b>" where label
    is 1 (same speaker) or 0 (different speakers). wav files live
    under `wav_dir`. Reports Equal Error Rate (EER) and the
    distribution of cosine similarities for same- vs different-speaker
    pairs.

    This is the metric the public benchmarks report — open-set means
    speakers in the test pairs were NOT seen during model training. Our
    handcrafted-feature classifier wasn't trained on anything, so this
    measures how generalizable the 29-dim feature space is across
    arbitrary speakers — a strictly harder task than our scoped 2-speaker
    closed-set conversation use case.

    Format expected (VoxCeleb-O compatible):
        1 id10001/abc/00001.wav id10001/abc/00002.wav
        0 id10001/abc/00001.wav id10999/xyz/00001.wav
    """
    print(f"\n=== Open-Set Verification (VoxCeleb-style pairs) ===")
    print(f"pairs: {pairs_path}")
    print(f"wav_dir: {wav_dir}")
    if not pairs_path.exists():
        print("[skip] pairs file not found")
        return
    if not wav_dir.exists():
        print("[skip] wav_dir not found")
        return

    pairs: list[tuple[int, Path, Path]] = []
    with pairs_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label, a, b = parts
            pairs.append((int(label), wav_dir / a, wav_dir / b))
    print(f"loaded {len(pairs)} pairs")
    if not pairs:
        return

    same_scores: list[float] = []
    diff_scores: list[float] = []
    skipped = 0
    for i, (label, a, b) in enumerate(pairs):
        try:
            wa = _load_wav_int16(a)
            wb = _load_wav_int16(b)
        except Exception:
            skipped += 1
            continue
        fa = extract_features(wa)
        fb = extract_features(wb)
        if fa is None or fb is None:
            skipped += 1
            continue
        sim = float(np.dot(_l2_normalize(fa), _l2_normalize(fb)))
        (same_scores if label == 1 else diff_scores).append(sim)
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(pairs)} pairs")
    print(f"scored: same={len(same_scores)} diff={len(diff_scores)} skipped={skipped}")

    if same_scores and diff_scores:
        eer = _equal_error_rate(same_scores, [-s for s in diff_scores])
        print(f"[open-set] EER: {eer * 100:.2f}%")
        print(f"  same-speaker cos sim: mean={np.mean(same_scores):.3f} std={np.std(same_scores):.3f}")
        print(f"  diff-speaker cos sim: mean={np.mean(diff_scores):.3f} std={np.std(diff_scores):.3f}")
        print("  Reference points (lower = better):")
        print("    ECAPA-TDNN (SpeechBrain, 2020):   ~0.86% EER on VoxCeleb-O")
        print("    Resemblyzer (Resemble AI, 2018):  ~5-6% EER")
        print("    Classical MFCC + UBM-GMM (2010):  ~10-15% EER")


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-10 else v


def benchmark_latency() -> None:
    """Measure per-call CPU time for the speaker classifier and AEC.

    Latency budget for a conversational turn is dominated by network
    LLM and TTS — but the local audio pipeline must stay under ~10ms
    per mic chunk to not introduce jitter into Deepgram's STT path.
    These measurements verify that.
    """
    import time
    print("\n=== Latency ===")

    # Classifier latency
    user_enrol, bot_enrol, _ = _build_synth_corpus()
    clf = SpeakerClassifier(sample_rate=SR)
    clf.reset_user()
    clf.enrol_user(user_enrol, persist=False)
    clf.enrol_bot(bot_enrol)

    test = user_enrol[: SR // 2]  # 500ms clip
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        clf.classify(test)
    t1 = time.perf_counter()
    per_call_ms = (t1 - t0) / n * 1000
    print(f"[classifier] {per_call_ms:.2f} ms per classify (500ms clip)")

    # AEC latency — per-chunk processing
    aec = make_echo_canceller(mic_rate=SR)
    ref = bot_enrol
    aec.push_reference(ref, src_rate=SR)
    chunk = np.zeros(1024, dtype=np.int16)
    # Build mic backlog so AEC actually processes (not just passthrough)
    mic = ref[:8192]
    t0 = time.perf_counter()
    for i in range(0, mic.size - 1024, 1024):
        aec.process(mic[i:i + 1024])
    t1 = time.perf_counter()
    if (mic.size // 1024) > 0:
        per_chunk_ms = (t1 - t0) / (mic.size // 1024) * 1000
        print(f"[aec] {per_chunk_ms:.2f} ms per 1024-sample (64ms) chunk")
    # Wall-clock budget reference
    print("  Wall-clock budget per mic chunk @ 16kHz, 1024 samples = 64ms.")
    print("  Anything under ~10ms leaves ample headroom for STT / LLM / TTS network calls.")


# ----------------------------------------------------------------------
# WER + STT eval


_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}]")


def _normalize_text(s: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    return s.split()


def _wer(reference: str, hypothesis: str) -> float:
    """Word-level error rate via Levenshtein distance on tokenized words.

    Returns edit-distance / len(reference). If the reference is empty,
    returns 0.0 when the hypothesis is also empty, else 1.0.
    Pure-python; O(N*M) memory.
    """
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    # Two-row dynamic programming
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,        # deletion
                cur[j - 1] + 1,     # insertion
                prev[j - 1] + cost  # substitution / match
            )
        prev, cur = cur, prev
    distance = prev[m]
    return distance / n


def _load_manifest() -> list[dict]:
    if not _MANIFEST_PATH.exists():
        return []
    try:
        with _MANIFEST_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[manifest] failed to load {_MANIFEST_PATH}: {exc}")
        return []


def _try_import_stt(backend: str):
    """Return a callable stt_factory(api_key) → STT instance, or None.

    The factory returns an object with .connect()/.send(bytes)/.finish()
    async methods and exposes ._on_chars / ._on_partial callback slots.
    """
    backend = backend.lower()
    if backend == "deepgram":
        try:
            from scripts.conduit_tui.deepgram_client import DeepgramStream
        except Exception as exc:
            print(f"[stt] deepgram import failed: {exc}")
            return None
        api_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
        if not api_key or api_key == "0":
            # Try .env.local
            env_path = _REPO_ROOT / ".env.local"
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        if line.startswith("DEEPGRAM_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                except Exception:
                    pass
        if not api_key or api_key == "0":
            print("[stt] DEEPGRAM_API_KEY not set (or set to '0'); cannot run deepgram benchmark")
            return None

        def factory():
            return DeepgramStream(api_key=api_key, sample_rate=SR)

        return factory

    if backend == "vosk":
        try:
            from scripts.conduit_tui.vosk_client import VoskStream  # type: ignore
        except Exception as exc:
            print(f"[stt] vosk import failed: {exc}")
            return None

        def factory():
            return VoskStream(sample_rate=SR)  # type: ignore[name-defined]

        return factory

    return None


async def _run_stt_clip(
    stt_factory,
    pcm_int16: np.ndarray,
    ground_truth: str,
) -> tuple[float, float, float, str]:
    """Run a single clip through an STT session.

    Returns (wer, latency_first_partial_s, latency_final_s, final_text).
    Latencies are NaN if no partial / final was observed.
    """
    stt = stt_factory()
    final_text_parts: list[str] = []
    partial_seen: list[float] = []
    final_seen: list[float] = []
    last_partial_text = [""]

    def _on_chars(chars):
        # DeepgramStream signals "final" via _on_chars (it only emits
        # chars on is_final). Treat each emission as a final segment.
        # The CharEntry list lacks explicit space separators — we
        # reconstruct word boundaries from the `notes` field, which
        # `interpolate_chars` stamps with "word=<word_text>" per word.
        try:
            parts: list[str] = []
            prev_word = None
            for c in chars:
                ch = getattr(c, "char", "") or ""
                notes = getattr(c, "notes", "") or ""
                # Extract word=... tag
                word_tag = None
                for tok in notes.split(","):
                    if tok.startswith("word="):
                        word_tag = tok[5:]
                        break
                if prev_word is not None and word_tag != prev_word:
                    parts.append(" ")
                parts.append(ch)
                prev_word = word_tag
            text = "".join(parts).strip()
        except Exception:
            text = ""
        if text:
            final_text_parts.append(text)
            final_seen.append(time.perf_counter())

    def _on_partial(text: str):
        partial_seen.append(time.perf_counter())
        last_partial_text[0] = text or ""

    # Attach callbacks (works for DeepgramStream — both attrs exist post-init)
    try:
        stt._on_chars = _on_chars  # type: ignore[attr-defined]
        stt._on_partial = _on_partial  # type: ignore[attr-defined]
    except Exception:
        pass

    await stt.connect()

    start = time.perf_counter()
    chunk_samples = 1024
    bytes_per_sample = 2
    # Real-time simulation: 1024 samples @ 16 kHz = 64 ms
    chunk_dur = chunk_samples / SR
    pcm_bytes = pcm_int16.astype(np.int16).tobytes()
    for off in range(0, len(pcm_bytes), chunk_samples * bytes_per_sample):
        chunk = pcm_bytes[off : off + chunk_samples * bytes_per_sample]
        await stt.send(chunk)
        # Small sleep to approximate real-time. Use a fraction of the
        # nominal chunk duration so the test runs faster than wall-clock.
        await asyncio.sleep(chunk_dur * 0.25)

    # Drain — give the server a moment to flush final results.
    # Deepgram's utterance_end_ms is 1500ms; we need to wait at least
    # that long after the last send for the final to land.
    await asyncio.sleep(2.0)
    await stt.finish()
    # Extra wait for any trailing final to land via the listen task
    await asyncio.sleep(0.3)

    lat_partial = (partial_seen[0] - start) if partial_seen else float("nan")
    lat_final = (final_seen[-1] - start) if final_seen else float("nan")

    final_text = " ".join(final_text_parts).strip()
    if not final_text:
        # Fall back to the last partial if no final ever arrived
        final_text = last_partial_text[0]

    wer = _wer(ground_truth, final_text)
    return wer, lat_partial, lat_final, final_text


def benchmark_stt(stt_factory, n_samples: int = 20, backend_label: str = "stt") -> None:
    print(f"\n=== STT Benchmark ({backend_label}) ===")
    manifest = _load_manifest()
    if not manifest:
        print("[skip] no manifest at data/dataset/manifest.json")
        return
    rng = random.Random(42)
    sample = rng.sample(manifest, min(n_samples, len(manifest)))
    print(f"sampling {len(sample)} of {len(manifest)} manifest clips")

    wers: list[float] = []
    lat_partials: list[float] = []
    lat_finals: list[float] = []
    failures = 0

    for i, entry in enumerate(sample):
        wav_rel = entry.get("wav_path", "")
        phrase = entry.get("phrase", "")
        wav_path = _REPO_ROOT / wav_rel if wav_rel else None
        if not wav_path or not wav_path.exists():
            failures += 1
            continue
        try:
            pcm = _load_wav_int16(wav_path)
        except Exception as exc:
            print(f"  [{i+1}/{len(sample)}] load failed: {exc}")
            failures += 1
            continue
        try:
            wer, lp, lf, hyp = asyncio.run(_run_stt_clip(stt_factory, pcm, phrase))
        except Exception as exc:
            print(f"  [{i+1}/{len(sample)}] stt session failed: {exc}")
            failures += 1
            continue
        wers.append(wer)
        if not math.isnan(lp):
            lat_partials.append(lp)
        if not math.isnan(lf):
            lat_finals.append(lf)
        print(
            f"  [{i+1}/{len(sample)}] wer={wer*100:5.1f}%  "
            f"lat_partial={lp*1000:6.0f}ms  lat_final={lf*1000:6.0f}ms  "
            f"ref={phrase!r}  hyp={hyp!r}"
        )

    if not wers:
        print(f"[stt] no clips scored ({failures} failures); is the backend reachable?")
        return

    mean_wer = float(np.mean(wers))
    std_wer = float(np.std(wers))
    mp = float(np.mean(lat_partials)) if lat_partials else float("nan")
    mf = float(np.mean(lat_finals)) if lat_finals else float("nan")
    print(f"\n[stt:{backend_label}] mean WER: {mean_wer*100:.1f}%  std: {std_wer*100:.1f}%  ({len(wers)} clips)")
    print(f"[stt:{backend_label}] mean latency-to-first-partial: {mp*1000:.0f} ms ({len(lat_partials)} clips)")
    print(f"[stt:{backend_label}] mean latency-to-final:          {mf*1000:.0f} ms ({len(lat_finals)} clips)")
    if failures:
        print(f"[stt:{backend_label}] {failures} clip(s) failed and were excluded")


# ----------------------------------------------------------------------
# Personalized speaker eval


def _chunk_signal(pcm: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Simple non-overlapping slicing into fixed-length chunks."""
    out: list[np.ndarray] = []
    for i in range(0, pcm.size - chunk_samples + 1, chunk_samples):
        out.append(pcm[i : i + chunk_samples])
    return out


def benchmark_speaker_personalized() -> None:
    """Personalized speaker eval using the operator's session wavs.

    Reflects the actual operator voice + microphone + room, not synthetic
    formants. Skips cleanly if no operator wavs are present.
    """
    print("\n=== Speaker Personalized Benchmark ===")
    if not _USER_DIR.exists():
        print(f"[skip] no operator wavs yet at {_USER_DIR.relative_to(_REPO_ROOT)} —")
        print("       run the TUI a few times to populate (operator audio is")
        print("       persisted by operator_capture.py on session finalize).")
        return

    user_wavs = sorted(_USER_DIR.glob("*.wav"))
    if not user_wavs:
        print(f"[skip] no operator wavs yet in {_USER_DIR.relative_to(_REPO_ROOT)} —")
        print("       run the TUI a few times to populate.")
        return

    print(f"found {len(user_wavs)} operator wav(s)")

    # Use the first wav as enrollment, remainder as test material.
    enrol_path = user_wavs[0]
    try:
        enrol_pcm = _load_wav_int16(enrol_path)
    except Exception as exc:
        print(f"[skip] enrollment wav {enrol_path.name} failed to load: {exc}")
        return

    if enrol_pcm.size < SR:
        print(f"[skip] enrollment wav {enrol_path.name} too short ({enrol_pcm.size} samples)")
        return

    enrol_feat = extract_features(enrol_pcm)
    if enrol_feat is None:
        print(f"[skip] enrollment feature extraction returned None for {enrol_path.name}")
        return
    enrol_norm = _l2_normalize(enrol_feat)

    # Gather all 2-second chunks across all wavs (incl. tail of enrol wav).
    chunk_samples = 2 * SR
    user_chunks: list[np.ndarray] = []
    # From the enrollment wav, skip the first chunk (used for enrolment)
    enrol_chunks = _chunk_signal(enrol_pcm, chunk_samples)
    user_chunks.extend(enrol_chunks[1:])
    for p in user_wavs[1:]:
        try:
            pcm = _load_wav_int16(p)
        except Exception:
            continue
        user_chunks.extend(_chunk_signal(pcm, chunk_samples))

    if not user_chunks:
        print("[skip] not enough user audio for 2s chunks after enrolment")
        return

    print(f"derived {len(user_chunks)} user test chunk(s) of 2.0s each")

    # Bot pool: random ElevenLabs clips from the manifest.
    manifest = _load_manifest()
    if not manifest:
        print("[skip] no manifest — cannot build user-vs-bot trials")
        return

    rng = random.Random(123)
    n_user_pairs = 50
    n_bot_pairs = 50

    # User-vs-user trials: pair the enrolment template against random user chunks
    # (with replacement if needed to hit the target count).
    user_scores: list[float] = []
    for _ in range(n_user_pairs):
        chunk = rng.choice(user_chunks)
        feat = extract_features(chunk)
        if feat is None:
            continue
        sim = float(np.dot(enrol_norm, _l2_normalize(feat)))
        user_scores.append(sim)

    # User-vs-bot trials
    bot_scores: list[float] = []
    bot_pool = list(manifest)
    rng.shuffle(bot_pool)
    bot_idx = 0
    attempts = 0
    while len(bot_scores) < n_bot_pairs and attempts < n_bot_pairs * 4:
        attempts += 1
        entry = bot_pool[bot_idx % len(bot_pool)]
        bot_idx += 1
        wav_rel = entry.get("wav_path", "")
        wav_path = _REPO_ROOT / wav_rel if wav_rel else None
        if not wav_path or not wav_path.exists():
            continue
        try:
            pcm = _load_wav_int16(wav_path)
        except Exception:
            continue
        if pcm.size < chunk_samples:
            # Use the whole clip if it's shorter than 2s
            seg = pcm
        else:
            # Random 2-second slice
            start = rng.randint(0, pcm.size - chunk_samples)
            seg = pcm[start : start + chunk_samples]
        feat = extract_features(seg)
        if feat is None:
            continue
        sim = float(np.dot(enrol_norm, _l2_normalize(feat)))
        bot_scores.append(sim)

    print(f"scored: user-vs-user={len(user_scores)}  user-vs-bot={len(bot_scores)}")

    if not user_scores or not bot_scores:
        print("[skip] insufficient trial scores")
        return

    eer = _equal_error_rate(user_scores, [-s for s in bot_scores])
    print(f"[personalized] EER (user-vs-bot, corpus-level): {eer*100:.1f}%")
    print(f"  user-vs-user cos sim: mean={np.mean(user_scores):.3f} std={np.std(user_scores):.3f}")
    print(f"  user-vs-bot  cos sim: mean={np.mean(bot_scores):.3f} std={np.std(bot_scores):.3f}")
    print("  Note: cosine similarity over 58-dim handcrafted features —")
    print("        compare to the synthetic-classifier EER above for the same speaker model.")


# ----------------------------------------------------------------------
# CLI


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker / AEC / STT benchmarks")
    parser.add_argument("--user", type=Path, help="optional user wav (16k mono) for enrollment")
    parser.add_argument("--bot", type=Path, help="optional bot wav (16k mono) for enrollment")
    parser.add_argument("--pairs", type=Path, help="VoxCeleb-style trials file (label wav_a wav_b)")
    parser.add_argument("--wav-dir", type=Path, help="root directory for wav files referenced by --pairs")
    parser.add_argument(
        "--stt",
        choices=["vosk", "deepgram", "none", "auto"],
        default="auto",
        help="STT backend for WER eval: 'none' to skip, 'auto' to try both",
    )
    parser.add_argument(
        "--stt-samples",
        type=int,
        default=20,
        help="number of manifest clips to sample for STT eval (default 20)",
    )
    parser.add_argument(
        "--personalized",
        action="store_true",
        help="run personalized speaker eval against data/dataset/user/*.wav",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # Synthetic + AEC + latency always run (cheap and offline).
    benchmark_classifier(verbose=not args.quiet)
    benchmark_aec()
    benchmark_latency()

    # Open-set pairs only if explicitly provided.
    if args.pairs and args.wav_dir:
        benchmark_open_set_pairs(args.pairs, args.wav_dir)
    else:
        print("\n[open-set] Skipped — pass --pairs <file> --wav-dir <dir> to compare against")
        print("           VoxCeleb-style benchmarks (the metric public papers report).")
        print("           See README 'Scoped vs general benchmark' for how to obtain the dataset.")

    # STT eval
    if args.stt == "none":
        print("\n[stt] Skipped — --stt none")
    else:
        backends_to_try = ["deepgram", "vosk"] if args.stt == "auto" else [args.stt]
        any_ran = False
        for backend in backends_to_try:
            factory = _try_import_stt(backend)
            if factory is None:
                print(f"\n[stt:{backend}] not available — skipping")
                continue
            any_ran = True
            try:
                benchmark_stt(factory, n_samples=args.stt_samples, backend_label=backend)
            except Exception as exc:
                print(f"\n[stt:{backend}] benchmark crashed: {exc}")
        if not any_ran:
            print("\n[stt] No STT backend available — install vosk or set DEEPGRAM_API_KEY")
            print("      (set --stt none to silence this check).")

    # Personalized speaker eval — always attempted by default; skips
    # cleanly when no operator wavs exist. The --personalized flag is
    # retained as a no-op kept for explicitness in scripts that want to
    # signal intent (and for future suppression of the default).
    _ = args.personalized  # silence "unused" linters; flag is documented
    benchmark_speaker_personalized()

    return 0


if __name__ == "__main__":
    sys.exit(main())
