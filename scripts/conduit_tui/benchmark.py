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
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.conduit_tui.aec import make_echo_canceller
from scripts.conduit_tui.speaker_id import SpeakerClassifier


SR = 16000


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker / AEC benchmarks")
    parser.add_argument("--user", type=Path, help="optional user wav (16k mono) for enrollment")
    parser.add_argument("--bot", type=Path, help="optional bot wav (16k mono) for enrollment")
    parser.add_argument("--pairs", type=Path, help="VoxCeleb-style trials file (label wav_a wav_b)")
    parser.add_argument("--wav-dir", type=Path, help="root directory for wav files referenced by --pairs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    benchmark_classifier(verbose=not args.quiet)
    benchmark_aec()
    benchmark_latency()
    if args.pairs and args.wav_dir:
        benchmark_open_set_pairs(args.pairs, args.wav_dir)
    else:
        print("\n[open-set] Skipped — pass --pairs <file> --wav-dir <dir> to compare against")
        print("           VoxCeleb-style benchmarks (the metric public papers report).")
        print("           See README 'Scoped vs general benchmark' for how to obtain the dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
