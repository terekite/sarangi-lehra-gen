#!/usr/bin/env python3
"""Source sarangi training audio from a **curated** list of solo recordings.

History: an earlier version keyword-searched YouTube and auto-cut everything,
including tabla sections it tried to strip with demucs. The results were bad —
demucs mush on the tabla clips, and random source fidelity from blind search.
This version fixes both:

  * **Curated sources.** A fixed, human-approved list of specific high-fidelity
    *solo* sarangi recordings (``SOURCES``) — no keyword search.
  * **Clean solo only, no demucs.** Per source we detect where tabla enters and
    keep only the **pre-tabla solo portion**, then cut continuous sections from
    it. Nothing is source-separated.
  * **Native-rate candidates.** Candidates and preview montages are kept at the
    source's native rate so you audition true fidelity, NOT the 16 kHz training
    downsample. Downsampling to the training rate happens only at ``--promote``.

Workflow (three phases, human veto in the middle)::

    # 1. Download + curate continuous solo sections into _candidates/solo/
    python training/source_data.py

    # 2. Build ~30 s montages (native rate) into ~/Desktop/sarangi-preview/
    python training/source_data.py --build-preview

    # 3. USER auditions, DELETES the bad ones, then promote copies survivors
    #    into processed/ downsampled to the chosen training rate.
    python training/source_data.py --promote --sample-rate 16000

Environment notes baked in (see the plan / DATA_SPEC.md):
  * yt-dlp is given ``--js-runtimes node`` or YouTube throttles to ~25 KiB/s.
    The full file is downloaded and the span capped in Python.

Heavy deps (torch/torchcrepe/librosa/yt_dlp) are imported lazily so ``--help``
works with nothing installed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CANDIDATES = os.path.join(DATA, "_candidates")
CAND_SOLO = os.path.join(CANDIDATES, "solo")
PREVIEW = os.path.join(DATA, "_preview")
PROCESSED = os.path.join(DATA, "processed")
DESKTOP_PREVIEW = os.path.expanduser("~/Desktop/sarangi-preview")
STATE_FILE = os.path.join(CANDIDATES, "_sources.json")   # video-id dedup ledger

SR = 16000            # feature-detection rate; frame rate = SR / BLOCK = 100 Hz
BLOCK = 160           # feature hop
FPS = SR // BLOCK     # 100 frames per second
NODE_PATH = "/opt/homebrew/bin/node"

# Curated, human-approved solo sarangi sources: (label, youtube_video_id).
# Fidelity/content notes live in the plan; the alap (unaccompanied opening) is
# the target — the extractor keeps only the pre-tabla solo portion of each.
SOURCES = [
    ("ram_narayan_lalit",      "FCm4LqPbLEM"),
    ("sultan_khan_ahirbhairav", "2gi6aG6DDMY"),
    ("sabri_khan_jog",         "DTX35Em7Y8Q"),
    ("ram_narayan_shankara",   "kX_BFUzPst4"),
    ("sultan_khan_bhimpalasi", "rpO_KT_Fsq4"),
    ("ram_narayan_kirvani",    "rLZ4Qzvj7y8"),
    ("ram_narayan_marwa",      "SXKRCJ98gVM"),
    ("sabri_khan_multani",     "gHqJzptHs5g"),
    ("sultan_khan_charukeshi", "3zhnARJC1Ks"),
    ("aruna_narayan_madhuvanti", "yXUndI6SER4"),
    ("kamal_sabri_pilu",       "Jger9JLTEdg"),
    ("kamal_sabri_shree",      "Nh9dUYUgf8Y"),
    ("ram_narayan_suprabhat",  "IhRaW4kgKJw"),
    ("munir_khan_darbari",     "-xAyRcvU4J4"),
]


# --------------------------------------------------------------------------- #
# Lazy imports
# --------------------------------------------------------------------------- #
def _np():
    import numpy as np
    return np


def _librosa():
    import librosa
    return librosa


# --------------------------------------------------------------------------- #
# yt-dlp download (full file, node JS runtime)
# --------------------------------------------------------------------------- #
def ensure_ytdlp() -> None:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        sys.exit("[source] yt-dlp not installed (pip install yt-dlp).")
    if not os.path.exists(NODE_PATH) and shutil.which("node") is None:
        print(f"[source] WARNING: node not found at {NODE_PATH} nor on PATH — "
              f"YouTube may throttle downloads to ~25 KiB/s.")


def _js_runtime_args() -> list[str]:
    node = NODE_PATH if os.path.exists(NODE_PATH) else shutil.which("node")
    return ["--js-runtimes", f"node:{node}"] if node else []


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"seen_ids": []}


def _save_state(state: dict) -> None:
    os.makedirs(CANDIDATES, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def download_audio(video_id: str, dst_dir: str) -> str | None:
    """Download the full-length bestaudio for a video id → local file path."""
    os.makedirs(dst_dir, exist_ok=True)
    out_tmpl = os.path.join(dst_dir, f"{video_id}.%(ext)s")
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestaudio/best",
        "--no-playlist",
        "-o", out_tmpl,
        *_js_runtime_args(),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        tail = (e.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
        print(f"  [dl-fail] {video_id}: {tail[0]}")
        return None
    hits = glob.glob(os.path.join(dst_dir, f"{video_id}.*"))
    return hits[0] if hits else None


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def _load_native_mono(path: str, max_min: float | None):
    """Load a file as native-rate mono float32, optionally capped to max_min.

    Returns (y, sr_native). The candidate is saved at this native rate; a 16 kHz
    copy is derived only for feature detection.
    """
    librosa = _librosa()
    dur = None if max_min is None else max_min * 60.0
    y, sr = librosa.load(path, sr=None, mono=True, duration=dur)
    return y.astype("float32"), int(sr)


def _to_16k(y, sr_native):
    if sr_native == SR:
        return y
    return _librosa().resample(y, orig_sr=sr_native, target_sr=SR)


# --------------------------------------------------------------------------- #
# Frame-rate features @ 100 Hz (input MUST be 16 kHz mono)
# --------------------------------------------------------------------------- #
def _crepe_f0_per(y, device: str):
    """torchcrepe 'tiny' f0 (Hz) + periodicity at hop=BLOCK (100 Hz frames)."""
    import torch
    import torchcrepe
    audio = torch.tensor(y, dtype=torch.float32, device=device)[None]
    f0, per = torchcrepe.predict(
        audio, SR, hop_length=BLOCK, fmin=50.0, fmax=1000.0,
        model="tiny", batch_size=1024, device=device, return_periodicity=True)
    return f0[0].cpu().numpy(), per[0].cpu().numpy()


def frame_features(y16, device: str) -> dict:
    """Per-frame features aligned at 100 Hz for a 16 kHz mono signal."""
    np = _np()
    librosa = _librosa()

    f0, per = _crepe_f0_per(y16, device)
    rms = librosa.feature.rms(y=y16, frame_length=2 * BLOCK, hop_length=BLOCK,
                              center=True)[0]
    harm, perc = librosa.effects.hpss(y16)
    rms_h = librosa.feature.rms(y=harm, frame_length=2 * BLOCK,
                                hop_length=BLOCK, center=True)[0]
    rms_p = librosa.feature.rms(y=perc, frame_length=2 * BLOCK,
                                hop_length=BLOCK, center=True)[0]
    perc_ratio = rms_p / (rms_h + rms_p + 1e-8)
    onset_env = librosa.onset.onset_strength(y=y16, sr=SR, hop_length=BLOCK)

    n = min(len(f0), len(per), len(rms), len(perc_ratio), len(onset_env))
    return {
        "f0": f0[:n], "per": per[:n], "rms": rms[:n],
        "perc_ratio": perc_ratio[:n], "onset": onset_env[:n], "n": n,
    }


def tabla_onset_frame(feat: dict, thresh: float, min_run_s: float) -> int:
    """First frame where percussive_ratio stays above ``thresh`` for min_run_s.

    That marks tabla entering; everything before it is the solo portion. Returns
    the frame count (whole file) if no sustained percussive stretch is found.
    """
    np = _np()
    over = feat["perc_ratio"] > thresh
    need = int(round(min_run_s * FPS))
    run = 0
    for i in range(len(over)):
        run = run + 1 if over[i] else 0
        if run >= need:
            return i - run + 1
    return feat["n"]


# --------------------------------------------------------------------------- #
# Playing mask + continuous regions (frame indices at 100 Hz)
# --------------------------------------------------------------------------- #
def _close_gaps(mask, max_gap_frames: int):
    m = mask.copy()
    n = len(m)
    i = 0
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            if i > 0 and mask[i - 1] and j < n and mask[j] \
                    and (j - i) <= max_gap_frames:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def _runs(mask):
    out = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def continuous_regions(feat: dict, rms_floor_pct: float, gap_max_s: float,
                       min_len_s: float, limit_frame: int | None = None):
    """Maximal continuous playing regions with no internal gap > gap_max_s.

    Playing = periodicity > 0.5 AND rms above a percentile floor. Micro-pauses
    (≤ gap_max_s) are bridged (preserved, never trimmed). Regions shorter than
    min_len_s are dropped. If ``limit_frame`` is given, regions are clipped to
    end at/before it (the solo/pre-tabla portion).
    """
    np = _np()
    per, rms = feat["per"], feat["rms"]
    floor = np.percentile(rms[rms > 0], rms_floor_pct) if np.any(rms > 0) else 0.0
    mask = (per > 0.5) & (rms > floor)
    if limit_frame is not None:
        mask = mask.copy()
        mask[limit_frame:] = False
    mask = _close_gaps(mask, int(round(gap_max_s * FPS)))
    min_len = int(round(min_len_s * FPS))
    return [(s, e) for (s, e) in _runs(mask) if (e - s) >= min_len]


def region_metrics(feat: dict, s: int, e: int) -> dict:
    np = _np()
    f0 = feat["f0"][s:e]
    per = feat["per"][s:e]
    perc = feat["perc_ratio"][s:e]
    voiced = per > 0.5
    vf0 = f0[voiced]
    return {
        "start": s, "end": e, "dur_s": (e - s) / FPS,
        "perc_ratio": float(np.mean(perc)) if len(perc) else 1.0,
        "periodicity": float(np.mean(per)) if len(per) else 0.0,
        "median_f0": float(np.median(vf0)) if len(vf0) else 0.0,
        "low_f0_frac": (float(np.mean(vf0 < 150.0)) if len(vf0) else 1.0),
    }


def section_ok(m: dict, min_len_s: float) -> bool:
    """Light QC — the human ear-veto is the real filter."""
    if m["dur_s"] < min_len_s:
        return False
    if not (150.0 <= m["median_f0"] <= 700.0):   # sarangi fundamental range
        return False
    if m["low_f0_frac"] > 0.2:                    # rejects vocal/harmonium bleed
        return False
    return True


# --------------------------------------------------------------------------- #
# Write helpers
# --------------------------------------------------------------------------- #
def _peak_normalize(y, dbfs: float = -1.0):
    np = _np()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0:
        return y
    return (y / peak * 10.0 ** (dbfs / 20.0)).astype("float32")


def _write_wav(path: str, y, sr: int):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, sr, subtype="PCM_16")


def _slug(text: str, maxlen: int = 32) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:maxlen] or "src"


# --------------------------------------------------------------------------- #
# Per-source processing → continuous solo sections (native rate)
# --------------------------------------------------------------------------- #
def process_source(path: str, label: str, vid: str, args, device: str) -> list[str]:
    np = _np()
    y_nat, sr_nat = _load_native_mono(path, args.max_source_min)
    if len(y_nat) < args.section_min * sr_nat:
        print(f"  [skip] {label}: shorter than one section")
        return []

    y16 = _to_16k(y_nat, sr_nat)
    feat = frame_features(y16, device)
    onset = tabla_onset_frame(feat, args.tabla_onset, args.tabla_run)
    solo_s = onset / FPS
    if onset < feat["n"]:
        print(f"  [tabla@] {solo_s:.0f}s (solo portion = first {solo_s:.0f}s of "
              f"{feat['n']/FPS:.0f}s)")
    else:
        print(f"  [solo] no tabla detected — whole {feat['n']/FPS:.0f}s is solo")

    regions = continuous_regions(
        feat, rms_floor_pct=args.rms_floor_pct, gap_max_s=args.gap_max,
        min_len_s=args.section_min, limit_frame=onset)
    if not regions:
        print(f"  [none] {label}: no continuous solo region ≥ "
              f"{args.section_min:.0f}s")
        return []

    written = []
    max_frames = int(round(args.section_max * FPS))
    for (s, e) in regions:
        if len(written) >= args.per_source:
            break
        e = min(e, s + max_frames)                 # clip to section_max
        m = region_metrics(feat, s, e)
        # Belt-and-suspenders: reject a section whose mean percussion is high
        # even if the onset detector didn't fire (tabla the human might miss).
        if not section_ok(m, args.section_min) or m["perc_ratio"] > args.tabla_onset:
            print(f"  [reject] [{s/FPS:.0f}-{e/FPS:.0f}s] medf0={m['median_f0']:.0f} "
                  f"lowf0={m['low_f0_frac']:.2f} perc={m['perc_ratio']:.2f}")
            continue
        # cut the NATIVE-rate audio by time so the candidate keeps full fidelity
        a = int(round(s / FPS * sr_nat))
        b = int(round(e / FPS * sr_nat))
        seg = _peak_normalize(y_nat[a:b])
        idx = len(written) + 1
        name = f"solo__ytsrc_{_slug(label)}_{vid}_{idx}.wav"
        out = os.path.join(CAND_SOLO, name)
        _write_wav(out, seg, sr_nat)
        written.append(out)
        print(f"  [solo] {name}  {len(seg)/sr_nat:5.1f}s @ {sr_nat} Hz  "
              f"perc={m['perc_ratio']:.2f} per={m['periodicity']:.2f} "
              f"medf0={m['median_f0']:.0f}")
    return written


# --------------------------------------------------------------------------- #
# Phase 1: download + curate
# --------------------------------------------------------------------------- #
def run_source(args, device: str) -> None:
    ensure_ytdlp()
    os.makedirs(CAND_SOLO, exist_ok=True)
    state = _load_state()
    seen = set(state["seen_ids"])
    minutes = _existing_candidate_minutes()
    print(f"== curate: {len(SOURCES)} curated source(s) | already have "
          f"{minutes:.1f} min | target {args.target_min:.0f} min ==")

    dl_dir = tempfile.mkdtemp(prefix="sarangi_dl_")
    try:
        for label, vid in SOURCES:
            if minutes >= args.target_min:
                break
            if vid in seen and not args.refetch:
                print(f"\n--- {label} ({vid}) — already processed, skip ---")
                continue
            print(f"\n--- {label} ({vid}) ---")
            seen.add(vid)
            state["seen_ids"] = sorted(seen)
            _save_state(state)
            path = download_audio(vid, dl_dir)
            if not path:
                continue
            try:
                process_source(path, label, vid, args, device)
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
            minutes = _existing_candidate_minutes()
            print(f"  … running total: {minutes:.1f} min")
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)

    print(f"\n[done] {minutes:.1f} min of candidate solo sections in {CAND_SOLO}")
    print("[next] python training/source_data.py --build-preview")


def _existing_candidate_minutes() -> float:
    import soundfile as sf
    total = 0.0
    for w in glob.glob(os.path.join(CAND_SOLO, "*.wav")):
        try:
            info = sf.info(w)
            total += info.frames / info.samplerate
        except Exception:  # noqa: BLE001
            pass
    return total / 60.0


# --------------------------------------------------------------------------- #
# Phase 2: build preview montages (native rate — audition true fidelity)
# --------------------------------------------------------------------------- #
def run_build_preview(args) -> None:
    import soundfile as sf
    os.makedirs(PREVIEW, exist_ok=True)
    os.makedirs(DESKTOP_PREVIEW, exist_ok=True)
    cands = sorted(glob.glob(os.path.join(CAND_SOLO, "*.wav")))
    if not cands:
        sys.exit("[preview] no candidates — run the curation phase first.")
    for c in cands:
        y, sr = sf.read(c, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        clip = y[: int(args.preview_s * sr)]
        base = os.path.basename(c)
        _write_wav(os.path.join(PREVIEW, base), clip, sr)
        sf.write(os.path.join(DESKTOP_PREVIEW, base), clip, sr, subtype="PCM_16")
    print(f"[preview] wrote {len(cands)} montage(s) (~{args.preview_s:.0f}s, "
          f"NATIVE rate) to:\n          {DESKTOP_PREVIEW}")
    print("[next] Listen, DELETE the bad ones, then:\n"
          "       python training/source_data.py --promote --sample-rate 16000")


# --------------------------------------------------------------------------- #
# Phase 3: promote survivors → processed/ (downsampled to training rate)
# --------------------------------------------------------------------------- #
def run_promote(args) -> None:
    import soundfile as sf
    librosa = _librosa()
    np = _np()
    os.makedirs(PROCESSED, exist_ok=True)
    review_dir = DESKTOP_PREVIEW if os.path.isdir(DESKTOP_PREVIEW) else PREVIEW
    survivors = {os.path.basename(p)
                 for p in glob.glob(os.path.join(review_dir, "*.wav"))}
    if not survivors:
        sys.exit(f"[promote] no montages in {review_dir} — run --build-preview.")
    n = 0
    for c in sorted(glob.glob(os.path.join(CAND_SOLO, "*.wav"))):
        base = os.path.basename(c)
        if base not in survivors:
            print(f"  [drop] {base}")
            continue
        y, sr = sf.read(c, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != args.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=args.sample_rate)
        y = _peak_normalize(y)
        _write_wav(os.path.join(PROCESSED, base), y, args.sample_rate)
        n += 1
        print(f"  [keep] {base}  → {args.sample_rate} Hz")
    print(f"[promote] copied {n} section(s) into {PROCESSED} @ "
          f"{args.sample_rate} Hz")
    print("[next] python training/train_torch_ddsp.py --fresh "
          "--hidden-size 512 --gru-layers 2 --steps 40000 --save-every 250")


# --------------------------------------------------------------------------- #
# Device + CLI
# --------------------------------------------------------------------------- #
def _pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="source_data.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--build-preview", action="store_true",
                      help="Phase 2: write ~30s native-rate montages.")
    mode.add_argument("--promote", action="store_true",
                      help="Phase 3: copy montage-survivors into processed/ "
                           "downsampled to --sample-rate.")

    p.add_argument("--target-min", type=float, default=60.0,
                   help="Stop curating once this many candidate minutes exist.")
    p.add_argument("--per-source", type=int, default=3,
                   help="Max continuous solo sections kept per recording.")
    p.add_argument("--max-source-min", type=float, default=20.0,
                   help="Cap each downloaded source to the first N minutes.")
    p.add_argument("--section-min", type=float, default=60.0,
                   help="Minimum continuous-section length (seconds).")
    p.add_argument("--section-max", type=float, default=180.0,
                   help="Maximum continuous-section length (seconds).")
    p.add_argument("--gap-max", type=float, default=1.0,
                   help="Max internal gap bridged inside a region (seconds); "
                        "preserves bow-lifts/breaths, never trims rests.")
    p.add_argument("--rms-floor-pct", type=float, default=20.0,
                   help="RMS percentile used as the playing-energy floor.")
    p.add_argument("--tabla-onset", type=float, default=0.30,
                   help="perc_ratio above which tabla is considered present.")
    p.add_argument("--tabla-run", type=float, default=5.0,
                   help="Seconds perc_ratio must stay high to mark tabla entry.")
    p.add_argument("--sample-rate", type=int, default=16000,
                   help="Training rate to downsample to at --promote.")
    p.add_argument("--preview-s", type=float, default=30.0,
                   help="Montage length per section (seconds).")
    p.add_argument("--refetch", action="store_true",
                   help="Re-process sources already in the dedup ledger.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"],
                   help="Device for CREPE feature extraction.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.build_preview:
        run_build_preview(args)
    elif args.promote:
        run_promote(args)
    else:
        device = _pick_device(args.device)
        print(f"== curation device (CREPE): {device} ==")
        run_source(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
