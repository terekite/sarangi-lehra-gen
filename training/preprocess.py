#!/usr/bin/env python3
"""Turn raw sarangi audio into a DDSP-ready 16 kHz mono dataset.

Pipeline (per file):

  1. SEPARATE  ``with_tabla/`` files are run through ``demucs`` and only the
     melodic (non-drums) stem is kept.  ``solo/`` files skip this step.
  2. TRIM      Leading/trailing silence is removed and long sarangi-silent
     gaps are dropped with a simple energy gate.
  3. DENOISE   (optional, ``--denoise``) Light spectral noise reduction using a
     user-supplied tanpura-only clip as the noise profile.
  4. RESAMPLE  Everything is converted to 16 kHz MONO and written to
     ``training/data/processed/``.
  5. TFRECORD  (optional, ``--tfrecord``) Emit the DDSP TFRecord dataset via
     ``ddsp_prepare_tfrecord`` into ``training/data/tfrecord/`` (or print the
     exact command with ``--tfrecord-dry-run``).

The training rate is **16 kHz mono** — that is the DDSP model's internal rate.
(The final *runtime* audio is resampled to 44.1 kHz elsewhere; that is not this
script's job.)

Heavy dependencies (demucs, ddsp, librosa, ...) are imported lazily *inside*
functions so this file stays importable and ``--help`` works even when nothing
is installed.  Install everything with::

    pip install -r training/requirements-train.txt

Examples::

    # Process both buckets with defaults
    python training/preprocess.py

    # Only the solo bucket, and also build the TFRecord
    python training/preprocess.py --only solo --tfrecord

    # With spectral denoise using a tanpura-only reference clip
    python training/preprocess.py --denoise --noise-clip refs/tanpura_only.wav

    # Just show the ddsp_prepare_tfrecord command, don't run it
    python training/preprocess.py --tfrecord --tfrecord-dry-run
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Paths (all relative to this file so it works from any CWD)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw")
SOLO_DIR = os.path.join(RAW_DIR, "solo")
WITH_TABLA_DIR = os.path.join(RAW_DIR, "with_tabla")
PROCESSED_DIR = os.path.join(HERE, "data", "processed")
TFRECORD_DIR = os.path.join(HERE, "data", "tfrecord")

TARGET_SR = 16000  # DDSP training rate: 16 kHz mono
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif")

_INSTALL_HINT = (
    "  ->  pip install -r training/requirements-train.txt\n"
    "     (training deps are Colab/dev-machine only; never shipped in the app)"
)


# ---------------------------------------------------------------------------
# Lazy import helpers — fail with a clear, actionable message
# ---------------------------------------------------------------------------
def _require(module_name: str, pip_name: str | None = None):
    """Import ``module_name`` or exit with an install hint."""
    try:
        return __import__(module_name)
    except ImportError:
        pip_name = pip_name or module_name
        sys.exit(
            f"\n[preprocess] Missing dependency '{pip_name}'.\n{_INSTALL_HINT}\n"
        )


def _load_audio(path: str, sr: int | None = None):
    """Load an audio file as float32 mono ``(samples, sample_rate)``.

    Uses librosa (which pulls in soundfile / audioread). Imported lazily.
    """
    librosa = _require("librosa")
    # mono=True downmixes; sr=None keeps native rate, sr=<n> resamples.
    y, out_sr = librosa.load(path, sr=sr, mono=True)
    return y, out_sr


def _write_wav(path: str, y, sr: int) -> None:
    sf = _require("soundfile")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Step 1: source separation (demucs) for with_tabla/ files
# ---------------------------------------------------------------------------
def separate_melodic_stem(in_path: str, model: str = "htdemucs") -> str:
    """Run demucs on ``in_path`` and return the path to the melodic stem.

    demucs splits into drums/bass/other/vocals.  Tabla lands mostly in
    ``drums`` (with some bleed into ``other``); we keep ``other`` — the
    non-percussive melodic content — which is where the sarangi lives.

    Returns the path to the separated ``other.wav`` in a temp dir.  The caller
    is responsible for further processing; the temp dir is left in place until
    process exit (small, and simplifies error handling).
    """
    # Ensure demucs is importable before shelling out to its CLI.
    _require("demucs")

    out_root = tempfile.mkdtemp(prefix="demucs_")
    # demucs CLI: -n <model> --two-stems is NOT used (we want the 4-stem split
    # so 'other' is isolated from 'bass'); default writes <out>/<model>/<track>/
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "-o", out_root,
        in_path,
    ]
    print(f"[demucs] separating: {os.path.basename(in_path)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"[demucs] failed on {in_path} (exit {e.returncode}). "
                 f"Try a different --demucs-model or check the file.")

    track = os.path.splitext(os.path.basename(in_path))[0]
    stem = os.path.join(out_root, model, track, "other.wav")
    if not os.path.exists(stem):
        # Some demucs versions name the track dir differently; find it.
        matches = glob.glob(os.path.join(out_root, model, "*", "other.wav"))
        if not matches:
            sys.exit(f"[demucs] no 'other' stem produced for {in_path}")
        stem = matches[0]
    return stem


# ---------------------------------------------------------------------------
# Step 2: silence trim + energy gate
# ---------------------------------------------------------------------------
def trim_and_gate(y, sr: int, top_db: float = 30.0, min_seg_s: float = 0.4):
    """Trim edge silence and concatenate only the non-silent segments.

    - ``top_db``: anything more than ``top_db`` below the peak counts as silence.
      Sarangi bow-noise floor is fairly high, so ~30 dB is a sensible default;
      lower it (e.g. 20) to be more aggressive.
    - ``min_seg_s``: drop kept segments shorter than this (removes clicks/blips).

    Returns the gated signal (float32).  If everything is gated out, returns an
    empty array so the caller can skip the file.
    """
    librosa = _require("librosa")
    np = _require("numpy")

    # librosa.effects.split returns non-silent [start, end) sample intervals.
    intervals = librosa.effects.split(y, top_db=top_db)
    min_len = int(min_seg_s * sr)
    kept = [y[s:e] for s, e in intervals if (e - s) >= min_len]
    if not kept:
        return np.zeros(0, dtype="float32")
    return np.concatenate(kept).astype("float32")


# ---------------------------------------------------------------------------
# Step 3: optional spectral noise reduction (tanpura profile)
# ---------------------------------------------------------------------------
def denoise(y, sr: int, noise_clip_path: str | None):
    """Light spectral noise reduction.

    If ``noise_clip_path`` is given it is used as the noise profile (a
    tanpura-only / room-tone section captured from the same source). Uses the
    ``noisereduce`` library if available, else a conservative spectral-gate
    fallback implemented with numpy so no extra dep is strictly required.

    This is intentionally gentle — over-denoising strips the bow noise that is
    part of the sarangi timbre we want to keep.
    """
    np = _require("numpy")

    noise = None
    if noise_clip_path:
        if not os.path.exists(noise_clip_path):
            sys.exit(f"[denoise] --noise-clip not found: {noise_clip_path}")
        noise, _ = _load_audio(noise_clip_path, sr=sr)

    # Prefer the well-tested noisereduce lib when installed.
    try:
        import noisereduce as nr  # optional
        if noise is not None and noise.size:
            return nr.reduce_noise(y=y, sr=sr, y_noise=noise,
                                   prop_decrease=0.75).astype("float32")
        return nr.reduce_noise(y=y, sr=sr,
                               prop_decrease=0.6).astype("float32")
    except ImportError:
        print("[denoise] 'noisereduce' not installed; using gentle numpy "
              "spectral-gate fallback. (pip install noisereduce for better "
              "results.)")

    # --- numpy fallback: subtract an average noise magnitude spectrum ---
    librosa = _require("librosa")
    n_fft, hop = 2048, 512
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop)
    mag, phase = np.abs(S), np.angle(S)

    if noise is not None and noise.size:
        Sn = np.abs(librosa.stft(noise, n_fft=n_fft, hop_length=hop))
        noise_profile = Sn.mean(axis=1, keepdims=True)
    else:
        # Use the quietest 10% of frames as an estimate of the noise floor.
        frame_energy = mag.mean(axis=0)
        thresh = np.percentile(frame_energy, 10)
        quiet = mag[:, frame_energy <= thresh]
        noise_profile = (quiet.mean(axis=1, keepdims=True)
                         if quiet.size else mag.min(axis=1, keepdims=True))

    reduced = np.maximum(mag - 1.0 * noise_profile, 0.0)
    out = librosa.istft(reduced * np.exp(1j * phase), hop_length=hop,
                        length=len(y))
    return out.astype("float32")


# ---------------------------------------------------------------------------
# Per-file driver
# ---------------------------------------------------------------------------
def process_file(in_path: str, bucket: str, args) -> str | None:
    """Process one raw file end-to-end; return the processed output path."""
    np = _require("numpy")

    src = in_path
    if bucket == "with_tabla":
        src = separate_melodic_stem(in_path, model=args.demucs_model)

    y, _ = _load_audio(src, sr=None)          # native rate first
    y = trim_and_gate(y, _sr_of(src), top_db=args.top_db,
                      min_seg_s=args.min_segment)
    if y.size == 0:
        print(f"[skip] fully silent after gating: {os.path.basename(in_path)}")
        return None

    # Re-load at native sr for denoise consistency, then resample last.
    sr_native = _sr_of(src)
    if args.denoise:
        y = denoise(y, sr_native, args.noise_clip)

    # Resample to 16 kHz mono (final training rate).
    librosa = _require("librosa")
    if sr_native != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr_native, target_sr=TARGET_SR)

    # Peak-normalize gently to -1 dBFS to even out sources.
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = (y / peak) * 0.89  # ~ -1 dBFS

    base = f"{bucket}__{os.path.splitext(os.path.basename(in_path))[0]}.wav"
    out_path = os.path.join(PROCESSED_DIR, base)
    _write_wav(out_path, y, TARGET_SR)
    dur = y.size / TARGET_SR
    print(f"[ok] {base}  ({dur:5.1f}s @ {TARGET_SR} Hz mono)")
    return out_path


def _sr_of(path: str) -> int:
    """Return the native sample rate of an audio file."""
    sf = _require("soundfile")
    try:
        return sf.info(path).samplerate
    except Exception:
        # mp3/m4a may not be readable by soundfile.info; fall back to a load.
        _, sr = _load_audio(path, sr=None)
        return sr


# ---------------------------------------------------------------------------
# Step 5: TFRecord export
# ---------------------------------------------------------------------------
def build_tfrecord(dry_run: bool) -> None:
    """Emit the DDSP TFRecord via ``ddsp_prepare_tfrecord``.

    ``ddsp_prepare_tfrecord`` is a console script installed by the ``ddsp`` pip
    package.  It runs CREPE (f0) + loudness extraction and shards the dataset.
    We glob every processed .wav as the input.
    """
    os.makedirs(TFRECORD_DIR, exist_ok=True)
    inputs = os.path.join(PROCESSED_DIR, "*.wav")
    output = os.path.join(TFRECORD_DIR, "sarangi.tfrecord")
    cmd = [
        "ddsp_prepare_tfrecord",
        f"--input_audio_filepatterns={inputs}",
        f"--output_tfrecord_path={output}",
        "--num_shards=10",
        "--alsologtostderr",
    ]
    printable = " ".join(cmd)
    if dry_run:
        print("\n[tfrecord] would run:\n\n  " + printable + "\n")
        return

    if shutil.which("ddsp_prepare_tfrecord") is None:
        sys.exit(
            "[tfrecord] 'ddsp_prepare_tfrecord' not on PATH — the ddsp package "
            "is not installed.\n"
            "Run this step on Colab (see training/train_ddsp.ipynb) or:\n"
            f"{_INSTALL_HINT}\n\n"
            "The exact command it would run is:\n  " + printable
        )
    print("[tfrecord] running ddsp_prepare_tfrecord (this invokes CREPE)...")
    subprocess.run(cmd, check=True)
    print(f"[tfrecord] wrote {output}*")


# ---------------------------------------------------------------------------
# Bucket iteration
# ---------------------------------------------------------------------------
def _list_audio(d: str) -> list[str]:
    if not os.path.isdir(d):
        return []
    return sorted(
        p for p in glob.glob(os.path.join(d, "*"))
        if p.lower().endswith(AUDIO_EXTS)
    )


def run(args) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    buckets = []
    if args.only in (None, "solo"):
        buckets.append(("solo", SOLO_DIR))
    if args.only in (None, "with_tabla"):
        buckets.append(("with_tabla", WITH_TABLA_DIR))

    total = 0.0
    n_ok = 0
    for bucket, d in buckets:
        files = _list_audio(d)
        if not files:
            print(f"[warn] no audio in {d} — see training/DATA_SPEC.md")
            continue
        print(f"\n=== {bucket}: {len(files)} file(s) ===")
        for f in files:
            out = process_file(f, bucket, args)
            if out:
                n_ok += 1
                # cheap duration tally
                sf = _require("soundfile")
                info = sf.info(out)
                total += info.frames / info.samplerate

    print(f"\n[done] processed {n_ok} file(s), ~{total/60:.1f} min total "
          f"(target 10-20 min) -> {PROCESSED_DIR}")
    if total < 10 * 60:
        print("[hint] below the 10 min target — source more (DATA_SPEC.md).")

    if args.tfrecord:
        build_tfrecord(dry_run=args.tfrecord_dry_run)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="preprocess.py",
        description="Build a 16 kHz mono DDSP dataset from raw sarangi audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--only", choices=["solo", "with_tabla"], default=None,
                   help="Process just one bucket (default: both).")
    p.add_argument("--top-db", type=float, default=30.0,
                   help="Silence threshold in dB below peak for the energy "
                        "gate (lower = more aggressive trimming).")
    p.add_argument("--min-segment", type=float, default=0.4,
                   help="Drop kept non-silent segments shorter than this "
                        "(seconds).")
    p.add_argument("--denoise", action="store_true",
                   help="Apply gentle spectral noise reduction.")
    p.add_argument("--noise-clip", default=None,
                   help="Path to a tanpura-only/room-tone clip to use as the "
                        "noise profile for --denoise (optional).")
    p.add_argument("--demucs-model", default="htdemucs",
                   help="demucs model name for with_tabla/ separation.")
    p.add_argument("--tfrecord", action="store_true",
                   help="After processing, build the DDSP TFRecord via "
                        "ddsp_prepare_tfrecord.")
    p.add_argument("--tfrecord-dry-run", action="store_true",
                   help="With --tfrecord, print the command instead of "
                        "running it.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
