#!/usr/bin/env python3
"""Source sarangi training audio from YouTube as **continuous sections**.

This replaces the earlier auto-curator that sliced sources into many short
8-second windows with silence trimmed out. Fragmented, gap-trimmed snippets
don't teach the model natural sarangi phrasing. Instead this tool extracts
**continuous, uninterrupted 1–3 minute sections** (rests and short micro-pauses
inside a passage are *preserved* — never cut) so the model sees a truer picture
of the instrument's timbre and dynamics.

Per source it keeps up to **two contrasting sections**:

  * **Section A — calm/solo:** a sustained, low-percussion, in-range passage.
  * **Section B — energetic:** the most active passage (higher onset /
    pitch-change rate), optionally faster and/or with tabla. If tabla is
    present, the melodic stem is isolated with demucs (Python API) and kept.

Workflow (three phases, human veto in the middle)::

    # 1. Source + curate into _candidates/{solo,with_tabla}/ (background-friendly)
    python training/source_data.py --bucket all --target-min 40 \
        --max-per-query 2 --max-source-min 15

    # 2. Build ~20 s montages of every section into _preview/ (+ ~/Desktop)
    python training/source_data.py --build-preview

    # 3. USER auditions ~/Desktop/sarangi-preview/*.wav, DELETES the bad ones,
    #    then promote copies the survivors into processed/
    python training/source_data.py --promote

Environment notes baked in (see the plan / DATA_SPEC.md):
  * yt-dlp is given ``--js-runtimes node`` or YouTube throttles to ~25 KiB/s.
    The full file is downloaded (no streamed ``--download-sections``, which run
    at ~realtime) and the span is capped in Python.
  * The demucs *CLI* save step is broken in this environment
    (libtorchcodec load failure at torchaudio.save). Tabla separation therefore
    uses the demucs **Python API** and saves via soundfile — never the CLI.

Heavy deps (torch/torchcrepe/librosa/demucs/yt_dlp) are imported lazily so
``--help`` works with nothing installed.
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
CAND_TABLA = os.path.join(CANDIDATES, "with_tabla")
PREVIEW = os.path.join(DATA, "_preview")
PROCESSED = os.path.join(DATA, "processed")
DESKTOP_PREVIEW = os.path.expanduser("~/Desktop/sarangi-preview")
STATE_FILE = os.path.join(CANDIDATES, "_sources.json")   # video-id dedup ledger

SR = 16000            # DDSP training rate: 16 kHz mono
BLOCK = 160           # feature hop → frame rate = SR / BLOCK = 100 Hz
FPS = SR // BLOCK     # 100 frames per second
NODE_PATH = "/opt/homebrew/bin/node"

# Search queries: (label, query). Each source contributes at most
# --max-per-query videos; sections are named ytsrc_<label>_<vid>_<A|B>.
QUERIES = [
    ("ram_narayan", "Ram Narayan sarangi solo raga alaap"),
    ("sultan_khan", "Ustad Sultan Khan sarangi solo"),
    ("dhruba_ghosh", "Dhruba Ghosh sarangi raga"),
    ("aruna_narayan", "Aruna Narayan sarangi solo"),
    ("ramesh_mishra", "Pandit Ramesh Mishra sarangi"),
    ("sabir_khan", "Sabir Khan sarangi solo raga"),
    ("kamal_sabri", "Kamal Sabri sarangi"),
    ("dhani_ram", "sarangi solo alaap raga classical"),
    ("murad_ali", "Murad Ali sarangi solo"),
    ("nasir_khan", "sarangi tabla jugalbandi classical"),
    ("faiyaz_khan", "sarangi solo vilambit teental"),
    ("classical_1", "Indian classical sarangi recital full"),
]

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".opus")


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
# yt-dlp download (full file, node JS runtime, video-id dedup)
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
    """yt-dlp args to enable the node JS runtime (avoids throttling)."""
    node = NODE_PATH if os.path.exists(NODE_PATH) else shutil.which("node")
    if node:
        return ["--js-runtimes", f"node:{node}"]
    return []


def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"seen_ids": [], "minutes": 0.0}


def _save_state(state: dict) -> None:
    os.makedirs(CANDIDATES, exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def _search_ids(query: str, n: int) -> list[tuple[str, str]]:
    """Return up to ``n`` (video_id, title) hits for a search query (flat)."""
    import yt_dlp
    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in (info or {}).get("entries", []) or []:
        vid = e.get("id")
        if vid:
            out.append((vid, e.get("title", "")))
    return out


def download_audio(video_id: str, dst_dir: str) -> str | None:
    """Download the full-length bestaudio for a video id → local file path.

    Full-file native download (NOT --download-sections, which streams at
    ~realtime). node JS runtime avoids throttling.
    """
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
def _load_16k_mono(path: str, max_min: float | None):
    """Load a file as 16 kHz mono float32, optionally capped to ``max_min``."""
    librosa = _librosa()
    dur = None if max_min is None else max_min * 60.0
    y, _ = librosa.load(path, sr=SR, mono=True, duration=dur)
    return y.astype("float32")


# --------------------------------------------------------------------------- #
# Frame-rate features @ 100 Hz
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


def frame_features(y, device: str) -> dict:
    """Compute all per-frame features aligned at 100 Hz (hop=BLOCK)."""
    np = _np()
    librosa = _librosa()

    f0, per = _crepe_f0_per(y, device)
    rms = librosa.feature.rms(y=y, frame_length=2 * BLOCK, hop_length=BLOCK,
                              center=True)[0]
    # Harmonic/percussive split → per-frame percussive energy ratio (tabla proxy)
    harm, perc = librosa.effects.hpss(y)
    rms_h = librosa.feature.rms(y=harm, frame_length=2 * BLOCK,
                                hop_length=BLOCK, center=True)[0]
    rms_p = librosa.feature.rms(y=perc, frame_length=2 * BLOCK,
                                hop_length=BLOCK, center=True)[0]
    perc_ratio = rms_p / (rms_h + rms_p + 1e-8)
    onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=BLOCK)

    n = min(len(f0), len(per), len(rms), len(perc_ratio), len(onset_env))
    return {
        "f0": f0[:n],
        "per": per[:n],
        "rms": rms[:n],
        "perc_ratio": perc_ratio[:n],
        "onset": onset_env[:n],
        "n": n,
    }


# --------------------------------------------------------------------------- #
# Playing mask + continuous regions
# --------------------------------------------------------------------------- #
def _close_gaps(mask, max_gap_frames: int):
    """Fill False runs no longer than ``max_gap_frames`` between True runs."""
    np = _np()
    m = mask.copy()
    n = len(m)
    i = 0
    # find gaps (False runs) that are bounded by True on both sides
    while i < n:
        if not m[i]:
            j = i
            while j < n and not m[j]:
                j += 1
            left_true = i > 0 and mask[i - 1]
            right_true = j < n and (j < n and mask[j])
            if left_true and right_true and (j - i) <= max_gap_frames:
                m[i:j] = True
            i = j
        else:
            i += 1
    return m


def _runs(mask):
    """Return list of (start, end) index pairs for maximal True runs."""
    np = _np()
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
                       min_len_s: float) -> list[tuple[int, int]]:
    """Maximal continuous playing regions with no internal gap > gap_max_s.

    Playing = periodicity > 0.5 AND rms above a percentile floor. Micro-pauses
    (short False runs ≤ gap_max_s) are bridged so a genuinely continuous passage
    stays whole; regions shorter than min_len_s are dropped. Rests *inside* a
    region are preserved — never excised.
    """
    np = _np()
    per = feat["per"]
    rms = feat["rms"]
    floor = np.percentile(rms[rms > 0], rms_floor_pct) if np.any(rms > 0) else 0.0
    mask = (per > 0.5) & (rms > floor)
    mask = _close_gaps(mask, int(round(gap_max_s * FPS)))
    min_len = int(round(min_len_s * FPS))
    return [(s, e) for (s, e) in _runs(mask) if (e - s) >= min_len]


def region_metrics(feat: dict, s: int, e: int) -> dict:
    """Summarize a region for selection/QC."""
    np = _np()
    f0 = feat["f0"][s:e]
    per = feat["per"][s:e]
    perc = feat["perc_ratio"][s:e]
    onset = feat["onset"][s:e]
    voiced = per > 0.5
    vf0 = f0[voiced]
    # semitone-change rate on voiced frames
    if len(vf0) > 2:
        cents = 12.0 * np.log2(np.clip(vf0[1:], 1e-6, None) /
                               np.clip(vf0[:-1], 1e-6, None))
        change_rate = float(np.mean(np.abs(cents) > 0.5)) * FPS  # events/s
    else:
        change_rate = 0.0
    # onset rate: peaks in the onset envelope per second
    thr = np.percentile(onset, 75) if len(onset) else 0.0
    onset_rate = float(np.mean(onset > max(thr, 1e-6))) * FPS
    return {
        "start": s, "end": e,
        "dur_s": (e - s) / FPS,
        "perc_ratio": float(np.mean(perc)),
        "periodicity": float(np.mean(per)),
        "median_f0": float(np.median(vf0)) if len(vf0) else 0.0,
        "low_f0_frac": (float(np.mean(vf0 < 150.0)) if len(vf0) else 1.0),
        "onset_rate": onset_rate,
        "change_rate": change_rate,
        "activity": onset_rate + change_rate,
    }


def section_ok(m: dict, min_len_s: float) -> bool:
    """Section-level QC gate (light — the human veto is the real filter)."""
    if m["dur_s"] < min_len_s:
        return False
    # Floor just above the per-frame mask (per>0.5). Tabla depresses CREPE
    # periodicity, so a stricter 0.6 wrongly rejects legitimate with-tabla
    # sections; median-f0 range + the human veto are the real filters.
    if m["periodicity"] < 0.5:
        return False
    if not (150.0 <= m["median_f0"] <= 700.0):
        return False
    if m["low_f0_frac"] > 0.2:
        return False
    return True


def _clip_window(feat: dict, m: dict, max_len_s: float, prefer: str) -> tuple[int, int]:
    """Clip a region to ≤ max_len_s, choosing the sub-window that best fits.

    prefer='calm' → lowest mean perc_ratio window; prefer='active' → highest
    activity window. If the region already fits, return it whole.
    """
    np = _np()
    s, e = m["start"], m["end"]
    max_len = int(round(max_len_s * FPS))
    if (e - s) <= max_len:
        return s, e
    perc = feat["perc_ratio"]
    onset = feat["onset"]
    best_i, best_score = s, None
    step = FPS  # slide 1 s at a time
    for i in range(s, e - max_len + 1, step):
        w = slice(i, i + max_len)
        if prefer == "calm":
            score = -float(np.mean(perc[w]))
        else:
            score = float(np.mean(onset[w]))
        if best_score is None or score > best_score:
            best_score, best_i = score, i
    return best_i, best_i + max_len


# --------------------------------------------------------------------------- #
# demucs melodic-stem separation (Python API — CLI save is broken here)
# --------------------------------------------------------------------------- #
_DEMUCS_MODEL = None


def _get_demucs(device: str):
    global _DEMUCS_MODEL
    if _DEMUCS_MODEL is None:
        from demucs.pretrained import get_model
        _DEMUCS_MODEL = get_model("htdemucs")
        _DEMUCS_MODEL.to(device)
        _DEMUCS_MODEL.eval()
    return _DEMUCS_MODEL


def separate_other(y16k, device: str):
    """Isolate the melodic ('other') stem from a 16 kHz mono clip via demucs.

    Uses the demucs Python API end-to-end (get_model / apply_model) and returns
    a 16 kHz mono float32 array — no torchaudio.save (broken in this env).
    """
    import torch
    from demucs.apply import apply_model
    np = _np()
    librosa = _librosa()

    model = _get_demucs(device)
    sr_m = model.samplerate
    # 16k mono → model sr, stereo (demucs expects [channels, samples])
    y = librosa.resample(y16k, orig_sr=SR, target_sr=sr_m)
    wav = torch.tensor(np.stack([y, y]), dtype=torch.float32)   # [2, N]
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(model, wav[None].to(device), device=device,
                          progress=False, split=True, overlap=0.1)[0]
    out = out * ref.std() + ref.mean()
    idx = model.sources.index("other")
    other = out[idx].mean(0).detach().cpu().numpy()             # → mono
    other = librosa.resample(other, orig_sr=sr_m, target_sr=SR)
    return other.astype("float32")


# --------------------------------------------------------------------------- #
# Write helpers
# --------------------------------------------------------------------------- #
def _peak_normalize(y, dbfs: float = -1.0):
    np = _np()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= 0:
        return y
    target = 10.0 ** (dbfs / 20.0)
    return (y / peak * target).astype("float32")


def _write_wav(path: str, y):
    import soundfile as sf
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y, SR, subtype="PCM_16")


def _slug(text: str, maxlen: int = 24) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return s[:maxlen] or "src"


# --------------------------------------------------------------------------- #
# Per-source processing → up to 2 continuous sections
# --------------------------------------------------------------------------- #
def process_source(path: str, label: str, vid: str, args, device: str) -> list[str]:
    """Extract ≤2 contrasting continuous sections from one downloaded source."""
    np = _np()
    y = _load_16k_mono(path, args.max_source_min)
    if y.size < args.section_min * SR:
        print(f"  [skip] {label}/{vid}: shorter than one section after cap")
        return []

    feat = frame_features(y, device)
    regions = continuous_regions(
        feat, rms_floor_pct=args.rms_floor_pct,
        gap_max_s=args.gap_max, min_len_s=args.section_min)
    if not regions:
        print(f"  [none] {label}/{vid}: no continuous region ≥ "
              f"{args.section_min:.0f}s (gap_max={args.gap_max:.1f}s)")
        return []

    metrics = [region_metrics(feat, s, e) for (s, e) in regions]
    metrics = [m for m in metrics if section_ok(m, args.section_min)]
    if not metrics:
        print(f"  [reject] {label}/{vid}: {len(regions)} region(s) failed QC "
              f"(periodicity/f0-range)")
        return []

    written = []

    # --- Section A (calm/solo): low percussion, high periodicity ---
    a = max(metrics, key=lambda m: m["periodicity"] - m["perc_ratio"])
    as_, ae = _clip_window(feat, a, args.section_max, prefer="calm")
    if args.bucket in ("all", "solo"):
        ya = _peak_normalize(y[as_ * BLOCK: ae * BLOCK])
        name = f"solo__ytsrc_{_slug(label)}_{vid}_A.wav"
        out = os.path.join(CAND_SOLO, name)
        _write_wav(out, ya)
        written.append(out)
        print(f"  [A/solo] {name}  {(ae-as_)/FPS:5.1f}s  "
              f"perc={a['perc_ratio']:.2f} per={a['periodicity']:.2f}")

    # --- Section B (energetic): most active, distinct from A ---
    def disjoint(m):
        return m["end"] <= a["start"] or m["start"] >= a["end"]
    b_candidates = [m for m in metrics if disjoint(m)]
    if b_candidates:
        b = max(b_candidates, key=lambda m: m["activity"])
        bs, be = _clip_window(feat, b, args.section_max, prefer="active")
        seg = y[bs * BLOCK: be * BLOCK]
        tabla = b["perc_ratio"] > args.tabla_thresh
        if tabla and args.bucket in ("all", "with_tabla"):
            print(f"  [B] separating tabla (perc={b['perc_ratio']:.2f}) via "
                  f"demucs …")
            try:
                seg = separate_other(seg, args.demucs_device)
                bucket_dir, tag = CAND_TABLA, "with_tabla"
            except Exception as ex:  # noqa: BLE001
                print(f"  [B/demucs-fail] {label}/{vid}: {ex} — skipping B")
                seg = None
        elif not tabla and args.bucket in ("all", "solo"):
            bucket_dir, tag = CAND_SOLO, "solo"
        else:
            seg = None  # bucket filter excludes this B
        if seg is not None and len(seg) >= args.section_min * SR:
            yb = _peak_normalize(seg)
            name = f"{tag}__ytsrc_{_slug(label)}_{vid}_B.wav"
            out = os.path.join(bucket_dir, name)
            _write_wav(out, yb)
            written.append(out)
            print(f"  [B/{tag}] {name}  {len(yb)/SR:5.1f}s  "
                  f"act={b['activity']:.1f} perc={b['perc_ratio']:.2f}")

    return written


# --------------------------------------------------------------------------- #
# Phase 1: source + curate
# --------------------------------------------------------------------------- #
def run_source(args, device: str) -> None:
    ensure_ytdlp()
    os.makedirs(CAND_SOLO, exist_ok=True)
    os.makedirs(CAND_TABLA, exist_ok=True)
    state = _load_state()
    seen = set(state["seen_ids"])
    minutes = _existing_candidate_minutes()
    print(f"== source: target {args.target_min:.0f} min | already have "
          f"{minutes:.1f} min of candidates | {len(seen)} ids seen ==")

    dl_dir = tempfile.mkdtemp(prefix="sarangi_dl_")
    try:
        for label, query in QUERIES:
            if minutes >= args.target_min:
                break
            print(f"\n--- query [{label}] {query!r} ---")
            hits = _search_ids(query, args.max_per_query * 4)
            kept = 0
            for vid, title in hits:
                if minutes >= args.target_min or kept >= args.max_per_query:
                    break
                if vid in seen:
                    continue
                seen.add(vid)
                state["seen_ids"] = sorted(seen)
                _save_state(state)
                print(f"  [dl] {vid}  {title[:60]!r}")
                path = download_audio(vid, dl_dir)
                if not path:
                    continue
                try:
                    outs = process_source(path, label, vid, args, device)
                finally:
                    try:
                        os.remove(path)   # free disk immediately
                    except OSError:
                        pass
                if outs:
                    kept += 1
                    minutes = _existing_candidate_minutes()
                    print(f"  … running total: {minutes:.1f} min")
    finally:
        shutil.rmtree(dl_dir, ignore_errors=True)

    print(f"\n[done] {minutes:.1f} min of candidate sections in {CANDIDATES}")
    print("[next] python training/source_data.py --build-preview")


def _existing_candidate_minutes() -> float:
    import soundfile as sf
    total = 0.0
    for d in (CAND_SOLO, CAND_TABLA):
        for w in glob.glob(os.path.join(d, "*.wav")):
            try:
                info = sf.info(w)
                total += info.frames / info.samplerate
            except Exception:  # noqa: BLE001
                pass
    return total / 60.0


# --------------------------------------------------------------------------- #
# Phase 2: build preview montages
# --------------------------------------------------------------------------- #
def run_build_preview(args) -> None:
    import soundfile as sf
    np = _np()
    os.makedirs(PREVIEW, exist_ok=True)
    os.makedirs(DESKTOP_PREVIEW, exist_ok=True)
    cands = _all_candidates()
    if not cands:
        sys.exit("[preview] no candidates — run the sourcing phase first.")
    n = 0
    for c in cands:
        y, sr = sf.read(c, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        clip = y[: int(args.preview_s * sr)]
        base = os.path.basename(c)
        _write_wav(os.path.join(PREVIEW, base), clip)
        sf.write(os.path.join(DESKTOP_PREVIEW, base), clip, sr, subtype="PCM_16")
        n += 1
    print(f"[preview] wrote {n} montage(s) (~{args.preview_s:.0f}s each) to:")
    print(f"          {DESKTOP_PREVIEW}")
    print("[next] Listen, DELETE the bad ones from that folder, then:")
    print("       python training/source_data.py --promote")


def _all_candidates() -> list[str]:
    out = []
    for d in (CAND_SOLO, CAND_TABLA):
        out += sorted(glob.glob(os.path.join(d, "*.wav")))
    return out


# --------------------------------------------------------------------------- #
# Phase 3: promote survivors into processed/
# --------------------------------------------------------------------------- #
def run_promote(args) -> None:
    os.makedirs(PROCESSED, exist_ok=True)
    # Survivors = candidates whose montage still exists in the preview folder
    # the user auditioned (Desktop takes precedence; fall back to _preview).
    review_dir = DESKTOP_PREVIEW if os.path.isdir(DESKTOP_PREVIEW) else PREVIEW
    survivors = {os.path.basename(p)
                 for p in glob.glob(os.path.join(review_dir, "*.wav"))}
    if not survivors:
        sys.exit(f"[promote] no montages found in {review_dir} — run "
                 f"--build-preview first (or you deleted them all).")
    n = 0
    for c in _all_candidates():
        base = os.path.basename(c)
        if base not in survivors:
            print(f"  [drop] {base} (montage was deleted)")
            continue
        shutil.copy2(c, os.path.join(PROCESSED, base))
        n += 1
        print(f"  [keep] {base}")
    print(f"[promote] copied {n} section(s) into {PROCESSED}")
    print("[next] python training/train_torch_ddsp.py --fresh "
          "--hidden-size 512 --gru-layers 2 --steps 40000 --save-every 250")


# --------------------------------------------------------------------------- #
# Device + CLI
# --------------------------------------------------------------------------- #
def _pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="source_data.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--build-preview", action="store_true",
                      help="Phase 2: write ~20s montages of every section.")
    mode.add_argument("--promote", action="store_true",
                      help="Phase 3: copy montage-survivors into processed/.")

    p.add_argument("--bucket", choices=["all", "solo", "with_tabla"],
                   default="all", help="Which section buckets to keep.")
    p.add_argument("--target-min", type=float, default=40.0,
                   help="Stop sourcing once this many candidate minutes exist.")
    p.add_argument("--max-per-query", type=int, default=2,
                   help="Max sources kept per search query.")
    p.add_argument("--max-source-min", type=float, default=15.0,
                   help="Cap each downloaded source to the first N minutes.")
    p.add_argument("--section-min", type=float, default=60.0,
                   help="Minimum continuous-section length (seconds).")
    p.add_argument("--section-max", type=float, default=180.0,
                   help="Maximum continuous-section length (seconds).")
    p.add_argument("--gap-max", type=float, default=1.0,
                   help="Max internal gap bridged inside a continuous region "
                        "(seconds). Real sarangi has bow-lifts/breaths up to "
                        "~1s that must not split a passage; a stricter 0.6 "
                        "rejected even clean solo sections in testing. Bridging "
                        "PRESERVES these pauses (never trims internal rests).")
    p.add_argument("--rms-floor-pct", type=float, default=20.0,
                   help="RMS percentile used as the playing-energy floor.")
    p.add_argument("--tabla-thresh", type=float, default=0.15,
                   help="perc_ratio above which Section B is demucs-separated.")
    p.add_argument("--demucs-device", default="cpu", choices=["cpu", "mps"],
                   help="Device for demucs separation (cpu is most reliable).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"],
                   help="Device for CREPE feature extraction.")
    p.add_argument("--preview-s", type=float, default=20.0,
                   help="Montage length per section (seconds).")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.build_preview:
        run_build_preview(args)
    elif args.promote:
        run_promote(args)
    else:
        device = _pick_device(args.device)
        print(f"== curation device (CREPE): {device} | demucs: "
              f"{args.demucs_device} ==")
        run_source(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
