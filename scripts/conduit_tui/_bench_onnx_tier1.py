"""Standalone benchmark for the Tier 1 ONNX speaker classifier.

5-way closed-set ElevenLabs benchmark:
  - enrol each of the 5 voices with its first wav (index 000)
  - test against 10 unseen wavs per voice (indices 001-010) -> 50 trials
  - report top-1 accuracy, confusion matrix
  - report mean cosine sim for same-voice and diff-voice pairs across
    the entire 30-clip-per-voice corpus
  - report per-classify latency

Direct comparison to Tier 0's 24% baseline on the same corpus.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Make 'scripts' importable as a top-level package no matter where we run from.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.conduit_tui.benchmark import _load_wav_int16  # noqa: E402
from scripts.conduit_tui.speaker_id_onnx import OnnxSpeakerClassifier  # noqa: E402


DATASET_ROOT = REPO_ROOT / "data" / "dataset"
MANIFEST = DATASET_ROOT / "manifest.json"
VOICES_DIR = DATASET_ROOT / "voices"


def _load_manifest() -> dict[str, list[Path]]:
    """voice_id -> [Path, ...] sorted by wav filename."""
    with MANIFEST.open() as fh:
        entries = json.load(fh)
    grouped: dict[str, list[Path]] = {}
    for e in entries:
        vid = e["voice_id"]
        wav = REPO_ROOT / e["wav_path"]
        grouped.setdefault(vid, []).append(wav)
    for vid in grouped:
        grouped[vid].sort()
    return grouped


def main() -> int:
    print("=" * 70)
    print("Tier 1 ONNX Speaker Classifier — 5-way Closed-Set Benchmark")
    print("=" * 70)

    if not MANIFEST.exists():
        print(f"[err] missing manifest at {MANIFEST}")
        return 1

    grouped = _load_manifest()
    voices = sorted(grouped.keys())
    print(f"corpus: {len(voices)} voices x {len(next(iter(grouped.values())))} clips each")
    for v in voices:
        print(f"  {v}  ({len(grouped[v])} wavs)")

    print()
    print("Loading classifier (downloads model on first run) ...")
    clf = OnnxSpeakerClassifier(verbose=True)
    clf.reset_user()  # drop any cached user template from previous tier
    print(f"embed dim   : 192")
    import os
    from scripts.conduit_tui.speaker_id_onnx import _MODEL_PATH
    sz = os.path.getsize(_MODEL_PATH) / 1e6
    print(f"model file  : {_MODEL_PATH}")
    print(f"model size  : {sz:.2f} MB")

    # ------------------------------------------------------------------
    # Enrollment: clip 000 of each voice -> template named by voice_id
    print()
    print("Enrolling each voice with clip 000 ...")
    for vid in voices:
        enrol_wav = grouped[vid][0]
        audio = _load_wav_int16(enrol_wav)
        ok = clf.enrol(vid, audio)
        if not ok:
            print(f"  [err] failed to enrol {vid} from {enrol_wav}")
            return 1
        print(f"  enrolled {vid}  ({audio.size / 16000:.2f}s)")

    # ------------------------------------------------------------------
    # Closed-set 5-way accuracy on indices 001..010 (10 per voice, 50 total)
    print()
    print("Closed-set 5-way evaluation (10 unseen clips per voice) ...")
    n = 0
    correct = 0
    confusion: dict[tuple[str, str], int] = {}
    latencies: list[float] = []
    for vid in voices:
        for wav in grouped[vid][1:11]:
            audio = _load_wav_int16(wav)
            t0 = time.perf_counter()
            pred, margin = clf.classify(audio)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            n += 1
            if pred == vid:
                correct += 1
            confusion[(vid, pred)] = confusion.get((vid, pred), 0) + 1

    acc = correct / max(n, 1)
    print(f"\n[result] closed-set top-1 accuracy: {correct}/{n} = {acc * 100:.1f}%")
    print(f"[result] per-classify latency: mean={np.mean(latencies):.1f}ms  "
          f"p50={np.percentile(latencies, 50):.1f}ms  p95={np.percentile(latencies, 95):.1f}ms")

    print("\n[confusion] (rows=true, columns=predicted)")
    header = "true \\ pred       " + "  ".join(v[:12].ljust(12) for v in voices)
    print(header)
    for tv in voices:
        row = [str(confusion.get((tv, pv), 0)).rjust(12) for pv in voices]
        print(f"  {tv[:16].ljust(16)} " + "  ".join(row))

    # ------------------------------------------------------------------
    # Same-voice vs diff-voice cosine sims (full corpus, all-vs-all pairs).
    print()
    print("Computing pairwise cosine sims (this may take a few seconds) ...")
    # Compute one embedding per wav
    embeds: dict[str, list[np.ndarray]] = {}
    for vid in voices:
        es: list[np.ndarray] = []
        for wav in grouped[vid]:
            audio = _load_wav_int16(wav)
            e = clf._embed(audio)  # noqa: SLF001 — internal helper, fine for bench
            if e is not None:
                es.append(e)
        embeds[vid] = es
        print(f"  {vid}: {len(es)} embeddings")

    same_scores: list[float] = []
    diff_scores: list[float] = []
    for vid in voices:
        es = embeds[vid]
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                same_scores.append(float(np.dot(es[i], es[j])))
        for other in voices:
            if other <= vid:
                continue
            for ea in es:
                for eb in embeds[other]:
                    diff_scores.append(float(np.dot(ea, eb)))

    same_mean = float(np.mean(same_scores)) if same_scores else float("nan")
    same_std = float(np.std(same_scores)) if same_scores else float("nan")
    diff_mean = float(np.mean(diff_scores)) if diff_scores else float("nan")
    diff_std = float(np.std(diff_scores)) if diff_scores else float("nan")
    print(f"\n[cosine] same-voice : mean={same_mean:.4f}  std={same_std:.4f}  n={len(same_scores)}")
    print(f"[cosine] diff-voice : mean={diff_mean:.4f}  std={diff_std:.4f}  n={len(diff_scores)}")
    print(f"[cosine] separation : {same_mean - diff_mean:+.4f}  (larger = better discrimination)")

    # ------------------------------------------------------------------
    # Final summary
    print()
    print("=" * 70)
    print("SUMMARY vs Tier 0 baseline")
    print("=" * 70)
    print(f"  Tier 0 (58-dim handcrafted MFCC):  24.0%  acc on this corpus")
    print(f"  Tier 1 (ONNX ECAPA-TDNN 192-dim):  {acc * 100:.1f}%  acc on this corpus")
    delta = (acc * 100) - 24.0
    print(f"  delta:                             {delta:+.1f} pp")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
