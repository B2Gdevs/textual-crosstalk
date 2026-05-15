"""
vosk_client.py — VoskStream: local-only STT alternative to DeepgramStream.

Surface mirrors scripts/conduit_tui/deepgram_client.py so the orchestrator
can be switched in via the CONDUIT_STT env var without touching its wiring:

  - async def connect()             — initialize Vosk recognizer + model
  - async def send(audio_bytes)     — feed 16-bit PCM mono @ 16 kHz
  - async def finish()              — close + cleanup
  - attribute _on_chars             — list[CharEntry] callback on final
  - attribute _on_partial           — str callback on interim transcripts

Model: vosk-model-small-en-us-0.15 (~40 MB on disk). Auto-downloaded on
first connect to ~/.conduit/models/vosk-small-en-us/. Idempotent.

Contract notes vs DeepgramStream:
  * Vosk does NOT emit utterance_end events. The orchestrator's
    Crosstalk coordinator already keys off SETTLED_THRESHOLD_MS silence
    (a wall-clock window with no new is_final words), so this is a
    safe no-op — Vosk is_final fires per acoustic-final, and silence
    after the last final triggers commit.
  * Vosk has no smart_format / punctuate flag in the small model — the
    transcript will be lowercase with no punctuation. UI surfaces the
    raw text; downstream LLM prompt is robust to either form.
  * Word timestamps are produced via `Result()` JSON `result[]` array,
    one entry per word with `start`/`end`/`word`. We feed each into
    interpolate_chars() the same way Deepgram words are handled.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .char_timeline import CharEntry, interpolate_chars


# Model bootstrap — small English model, ~40 MB compressed.
_MODEL_URL = (
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
)
_MODEL_DIR_NAME = "vosk-small-en-us"
_MODEL_ZIP_ROOT = "vosk-model-small-en-us-0.15"  # folder inside the zip


def _default_model_root() -> Path:
    """~/.conduit/models/  (gitignored — see .gitignore)."""
    return Path.home() / ".conduit" / "models"


def _model_is_present(model_dir: Path) -> bool:
    """Vosk needs the unpacked model folder with am/ + graph/ + conf/."""
    if not model_dir.is_dir():
        return False
    needed = ("am", "conf", "graph")
    return all((model_dir / sub).is_dir() for sub in needed)


def _download_model(model_dir: Path) -> None:
    """Synchronous download + unzip. Run inside asyncio.to_thread.

    Idempotent — checks _model_is_present() first; cleans up any partial
    download on failure so the next attempt starts fresh.
    """
    if _model_is_present(model_dir):
        return

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir.parent / f"{_MODEL_ZIP_ROOT}.zip"

    print(f"[vosk] downloading model from {_MODEL_URL}")
    print(f"[vosk] target: {model_dir}")
    try:
        # urlretrieve with a progress callback so the user can see this
        # isn't hung — small model ~40 MB.
        last_pct = [-1]

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            pct = min(100, int(block_num * block_size * 100 / total_size))
            if pct != last_pct[0] and pct % 10 == 0:
                print(f"[vosk]   {pct}%  ({total_size / (1024*1024):.1f} MB)")
                last_pct[0] = pct

        urllib.request.urlretrieve(_MODEL_URL, zip_path, reporthook=_report)
    except Exception as exc:
        if zip_path.exists():
            zip_path.unlink()
        raise RuntimeError(f"vosk model download failed: {exc}") from exc

    print(f"[vosk] unzipping -> {model_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_dir.parent)
        unpacked = model_dir.parent / _MODEL_ZIP_ROOT
        if unpacked.is_dir():
            if model_dir.exists():
                shutil.rmtree(model_dir)
            unpacked.rename(model_dir)
    finally:
        if zip_path.exists():
            zip_path.unlink()

    if not _model_is_present(model_dir):
        raise RuntimeError(
            f"vosk model directory looks incomplete after unzip: {model_dir}"
        )
    print(f"[vosk] model ready at {model_dir}")


class VoskStream:
    """Async local Vosk STT session — DeepgramStream-compatible surface."""

    def __init__(
        self,
        model_path: Path | None = None,
        sample_rate: int = 16000,
        session_start: float | None = None,
        on_chars: Callable[[list[CharEntry]], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
    ) -> None:
        self._model_path = (
            Path(model_path)
            if model_path is not None
            else _default_model_root() / _MODEL_DIR_NAME
        )
        self._sample_rate = int(sample_rate)
        self._session_start = session_start or time.monotonic()
        self._on_chars = on_chars
        self._on_partial = on_partial

        # Lazy: created in connect()
        self._model = None
        self._recognizer = None
        # Vosk KaldiRecognizer is not safe to call concurrently from
        # multiple coroutines. We run AcceptWaveform on a worker thread
        # and serialize calls with an asyncio.Lock.
        self._lock: asyncio.Lock | None = None
        # Track last partial to avoid spamming the UI with duplicates.
        self._last_partial: str = ""

    async def connect(self) -> None:
        """Bootstrap the model (download if missing) + create recognizer."""
        # Import vosk lazily so the module loads even on machines that
        # never use the Vosk path.
        import vosk

        # Quiet Vosk's internal Kaldi logging — it's extremely chatty by
        # default and stomps the TUI.
        try:
            vosk.SetLogLevel(-1)
        except Exception:
            pass

        await asyncio.to_thread(_download_model, self._model_path)

        # Model() load is CPU-bound (~200ms for the small model). Push
        # to a worker thread so the event loop stays responsive.
        def _load() -> tuple[object, object]:
            model = vosk.Model(str(self._model_path))
            rec = vosk.KaldiRecognizer(model, self._sample_rate)
            rec.SetWords(True)
            return model, rec

        self._model, self._recognizer = await asyncio.to_thread(_load)
        self._lock = asyncio.Lock()

    async def send(self, audio_bytes: bytes) -> None:
        """Feed 16-bit PCM mono @ sample_rate.

        Vosk's AcceptWaveform returns True when a final segment is ready
        (full Result()). False means partial — fetch via PartialResult().
        """
        if self._recognizer is None or self._lock is None:
            return

        async with self._lock:
            try:
                accepted = await asyncio.to_thread(
                    self._recognizer.AcceptWaveform, audio_bytes
                )
                if accepted:
                    raw = await asyncio.to_thread(self._recognizer.Result)
                    self._dispatch_final(raw)
                else:
                    raw = await asyncio.to_thread(self._recognizer.PartialResult)
                    self._dispatch_partial(raw)
            except Exception as exc:
                print(f"[vosk] send error: {exc}")

    async def finish(self) -> None:
        """Flush + drop recognizer. Idempotent."""
        if self._recognizer is not None and self._lock is not None:
            async with self._lock:
                try:
                    raw = await asyncio.to_thread(self._recognizer.FinalResult)
                    self._dispatch_final(raw)
                except Exception as exc:
                    print(f"[vosk] finish error: {exc}")
        self._recognizer = None
        self._model = None
        self._lock = None

    # ------------------------------------------------------------------
    # Internal — JSON → CharEntry / partial dispatch

    def _dispatch_partial(self, raw_json: str) -> None:
        try:
            payload = json.loads(raw_json or "{}")
        except json.JSONDecodeError:
            return
        text = (payload.get("partial") or "").strip()
        if not text or text == self._last_partial:
            return
        self._last_partial = text
        if self._on_partial:
            try:
                self._on_partial(text)
            except Exception as exc:
                print(f"[vosk] partial cb error: {exc}")

    def _dispatch_final(self, raw_json: str) -> None:
        try:
            payload = json.loads(raw_json or "{}")
        except json.JSONDecodeError:
            return
        text = (payload.get("text") or "").strip()
        if not text:
            self._last_partial = ""
            return
        words = payload.get("result") or []

        all_chars: list[CharEntry] = []
        if words:
            for w in words:
                word_text = (w.get("word") or "").strip()
                if not word_text:
                    continue
                # Vosk timestamps are seconds since the start of the
                # recognizer's audio stream — same semantics as the
                # Deepgram offsets that DeepgramStream adds session_start
                # to. Mirror that convention.
                word_start = self._session_start + float(w.get("start", 0.0))
                word_end = self._session_start + float(
                    w.get("end", w.get("start", 0.0))
                )
                all_chars.extend(
                    interpolate_chars(
                        word_text, word_start, word_end, "user,interpolated"
                    )
                )
        else:
            # No word-level info — fall back to flat now-stamped chars so
            # downstream Crosstalk still sees text + a word boundary.
            now = time.monotonic()
            for ch in text:
                all_chars.append(
                    CharEntry(
                        char=ch, start_time=now, end_time=now, notes="user,no-words"
                    )
                )

        self._last_partial = ""
        if all_chars and self._on_chars:
            try:
                self._on_chars(all_chars)
            except Exception as exc:
                print(f"[vosk] chars cb error: {exc}")


# ---------------------------------------------------------------------------
# Smoke test — `python -m scripts.conduit_tui.vosk_client`
#
# Reads one wav from the dataset, streams it 1024 samples at a time, and
# prints partials + finals. Validates against the manifest ground truth.

async def _smoke_test() -> int:  # pragma: no cover - manual harness
    import wave
    import sys
    import numpy as np

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / "data" / "dataset" / "manifest.json"
    if not manifest_path.exists():
        print(f"[smoke] no manifest at {manifest_path}", file=sys.stderr)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # First Sarah/000 entry is convenient + short.
    sample = next(
        (m for m in manifest if m["wav_path"].endswith("EXAVITQu4vr4xnSDxMaL/000.wav")),
        manifest[0],
    )
    wav_path = repo_root / sample["wav_path"]
    expected = sample["phrase"]
    print(f"[smoke] wav = {wav_path}")
    print(f"[smoke] expected = {expected!r}")

    # Load + resample to 16 kHz mono int16.
    with wave.open(str(wav_path), "rb") as wf:
        src_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    target_rate = 16000
    if src_rate != target_rate:
        ratio = target_rate / src_rate
        n_out = max(1, int(pcm.size * ratio))
        src_t = np.linspace(0.0, 1.0, pcm.size, dtype=np.float32)
        dst_t = np.linspace(0.0, 1.0, n_out, dtype=np.float32)
        pcm = np.interp(dst_t, src_t, pcm.astype(np.float32)).astype(np.int16)
    print(f"[smoke] pcm: {pcm.size} samples @ {target_rate} Hz")

    partials: list[str] = []
    finals: list[str] = []
    char_batches: list[list[CharEntry]] = []
    first_partial_t: list[float | None] = [None]
    first_send_t: list[float | None] = [None]

    def on_partial(text: str) -> None:
        if first_partial_t[0] is None and first_send_t[0] is not None:
            first_partial_t[0] = time.perf_counter()
        partials.append(text)

    def on_chars(chars: list[CharEntry]) -> None:
        char_batches.append(chars)
        # Reconstruct words from the `word=` tag in CharEntry.notes —
        # interpolate_chars emits chars per word without spaces, so a
        # naive "".join would smush "i think we" into "ithinkwe".
        words: list[str] = []
        current_word_tag: str | None = None
        current_chars: list[str] = []
        for c in chars:
            tag = next(
                (p for p in c.notes.split(",") if p.startswith("word=")),
                None,
            )
            if tag != current_word_tag and current_chars:
                words.append("".join(current_chars))
                current_chars = []
            current_word_tag = tag
            current_chars.append(c.char)
        if current_chars:
            words.append("".join(current_chars))
        finals.append(" ".join(words))

    stream = VoskStream(sample_rate=target_rate, on_chars=on_chars, on_partial=on_partial)
    t0 = time.perf_counter()
    await stream.connect()
    print(f"[smoke] connect() took {time.perf_counter() - t0:.2f}s")

    chunk_samples = 1024
    chunk_bytes = chunk_samples * 2  # int16 → 2 bytes
    raw_bytes = pcm.tobytes()

    for offset in range(0, len(raw_bytes), chunk_bytes):
        if first_send_t[0] is None:
            first_send_t[0] = time.perf_counter()
        await stream.send(raw_bytes[offset : offset + chunk_bytes])

    await stream.finish()

    full_final = " ".join(finals).strip()
    print(f"[smoke] partials seen   : {len(partials)}")
    if partials:
        print(f"[smoke] first partial   : {partials[0]!r}")
        print(f"[smoke] last partial    : {partials[-1]!r}")
    if (
        first_send_t[0] is not None
        and first_partial_t[0] is not None
    ):
        latency_ms = (first_partial_t[0] - first_send_t[0]) * 1000
        print(f"[smoke] first-byte-to-first-partial latency = {latency_ms:.1f} ms")
    else:
        print("[smoke] no partial emitted before first final")

    print(f"[smoke] final transcript : {full_final!r}")
    print(f"[smoke] expected phrase  : {expected!r}")

    # Normalize: lowercase, strip punctuation, collapse whitespace.
    import string

    def _norm(s: str) -> str:
        s = s.lower().translate(str.maketrans("", "", string.punctuation))
        return " ".join(s.split())

    match = _norm(full_final) == _norm(expected)
    print(f"[smoke] match (normalized): {match}")
    if not match:
        print(f"[smoke]   got:      {_norm(full_final)!r}")
        print(f"[smoke]   expected: {_norm(expected)!r}")
    return 0 if match else 1


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(asyncio.run(_smoke_test()))
