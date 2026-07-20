#!/usr/bin/env python3
"""Run one sarangi training→render→verify cycle.

Purpose
-------
A single command to run **every time you add training data**. It:

  1. Preprocesses whatever is in ``training/data/raw/{solo,with_tabla}/``
     (demucs strip on with_tabla, silence trim, 16 kHz mono, optional TFRecord).
  2. Detects a trained DDSP checkpoint at ``model/sarangi_ddsp/``.
       * checkpoint present  -> renders the matrix with the **ddsp** backend
       * checkpoint absent    -> renders **sine** previews and tells you to train
         (open ``training/train_ddsp.ipynb`` on Colab, export the checkpoint,
          then re-run this script).
  3. Renders the §7 render matrix (Sa x bpm x seed) into ``out/``.
  4. Copies every render to ``~/Desktop/sarangi-renders/`` with human-readable
     names + a ``MANIFEST.txt``, so you can listen and verify after training.

The actual GPU training is a manual Colab step (no GPU/TensorFlow locally); this
driver automates the deterministic local parts around it. A fresh agent can run
this with no prior context — see ``training/RUN_CYCLE.md`` for the full runbook.

Usage
-----
    python scripts/run_training_cycle.py                 # full cycle, auto backend
    python scripts/run_training_cycle.py --skip-preprocess
    python scripts/run_training_cycle.py --backend sine  # force preview renders
    python scripts/run_training_cycle.py --sa C,C# --bpm 80,160 --seed 0,3
    python scripts/run_training_cycle.py --no-copy       # render only, no Desktop
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generate import (  # noqa: E402  (path set above)
    DEFAULT_MATRIX_BPM,
    DEFAULT_MATRIX_SA,
    DEFAULT_MATRIX_SEED,
    SA_CHOICES,
    _build_params,
    render_one,
)
from sarangi_gen.cache_key import cache_key  # noqa: E402

LEHRA_PATH = REPO_ROOT / "lehras" / "proposed-teentaal.nagma"
CHECKPOINT_DIR = REPO_ROOT / "model" / "sarangi_ddsp"
RAW_SOLO = REPO_ROOT / "training" / "data" / "raw" / "solo"
RAW_TABLA = REPO_ROOT / "training" / "data" / "raw" / "with_tabla"
PREPROCESS = REPO_ROOT / "training" / "preprocess.py"
OUT_DIR = REPO_ROOT / "out"
DESKTOP_RENDERS = Path.home() / "Desktop" / "sarangi-renders"

_AUDIO_GLOBS = ("*.wav", "*.flac", "*.mp3", "*.m4a", "*.aiff", "*.aif")


def _has_audio(d: Path) -> bool:
    return d.is_dir() and any(next(d.glob(g), None) for g in _AUDIO_GLOBS)


def _checkpoint_present() -> bool:
    """A checkpoint 'exists' if the dir holds any TF checkpoint/saved-model files."""
    if not CHECKPOINT_DIR.is_dir():
        return False
    # PyTorch DDSP export (current): config.json + ddsp_torch.pt.
    # Legacy TF/magenta-DDSP markers kept for backward compatibility.
    markers = ("ddsp_torch.pt", "config.json", "*.index", "checkpoint",
               "*.pb", "saved_model.pb", "operative_config*.gin")
    return any(next(CHECKPOINT_DIR.rglob(m), None) for m in markers)


def _sa_slug(sa: str) -> str:
    return sa.replace("#", "sharp")


def preprocess(args) -> None:
    if args.skip_preprocess:
        print("== preprocess: skipped (--skip-preprocess) ==")
        return
    if not (_has_audio(RAW_SOLO) or _has_audio(RAW_TABLA)):
        print("== preprocess: no audio in training/data/raw/{solo,with_tabla}/ — "
              "skipping. Add data per training/DATA_SPEC.md. ==")
        return
    if not PREPROCESS.exists():
        print(f"== preprocess: {PREPROCESS} not found — skipping. ==")
        return
    cmd = [sys.executable, str(PREPROCESS)]
    if args.tfrecord:
        cmd.append("--tfrecord")
    print(f"== preprocess: {' '.join(cmd)} ==")
    rc = subprocess.call(cmd, cwd=str(REPO_ROOT))
    if rc != 0:
        print(f"!! preprocess exited {rc}. Fix data/deps (training/requirements-train.txt) "
              f"and re-run. Continuing to render with whatever exists.")


def choose_backend(requested: str) -> tuple[str, bool]:
    """Return (backend, is_real). is_real=True only for a genuine ddsp render."""
    have_ckpt = _checkpoint_present()
    if requested == "ddsp":
        if not have_ckpt:
            print("!! --backend ddsp requested but no checkpoint at "
                  f"{CHECKPOINT_DIR}. Falling back to sine previews.")
            return "sine", False
        return "ddsp", True
    if requested == "sine":
        return "sine", False
    # auto
    if have_ckpt:
        print(f"== checkpoint found at {CHECKPOINT_DIR} -> rendering with DDSP ==")
        return "ddsp", True
    print("== no checkpoint yet -> rendering SINE previews. Train first: open "
          "training/train_ddsp.ipynb on Colab, export to model/sarangi_ddsp/, "
          "then re-run this script. ==")
    return "sine", False


def matrix(args) -> list[tuple[str, float, int]]:
    sa_vals = [s.strip() for s in args.sa.split(",")] if args.sa else DEFAULT_MATRIX_SA
    bpm_vals = [float(b) for b in args.bpm.split(",")] if args.bpm else DEFAULT_MATRIX_BPM
    seed_vals = [int(s) for s in args.seed.split(",")] if args.seed else DEFAULT_MATRIX_SEED
    for sa in sa_vals:
        if sa not in SA_CHOICES:
            sys.exit(f"--sa {sa!r} not in schema enum {SA_CHOICES}")
    return list(itertools.product(sa_vals, bpm_vals, seed_vals))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["auto", "sine", "ddsp"], default="auto",
                   help="auto (default): ddsp if a checkpoint exists, else sine.")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Do not run training/preprocess.py; render only.")
    p.add_argument("--tfrecord", action="store_true",
                   help="Ask preprocess.py to also emit the DDSP TFRecord.")
    p.add_argument("--sa", default=None, help="Comma-separated Sa list (default matrix).")
    p.add_argument("--bpm", default=None, help="Comma-separated bpm list (default matrix).")
    p.add_argument("--seed", default=None, help="Comma-separated seed list (default matrix).")
    p.add_argument("--avartans", type=int, default=4, help="Avartans per render (default 4).")
    p.add_argument("--lehra", type=Path, default=LEHRA_PATH, help="Lehra file to render.")
    p.add_argument("--no-copy", action="store_true",
                   help="Skip copying renders to ~/Desktop/sarangi-renders/.")
    args = p.parse_args(argv)

    if not args.lehra.exists():
        sys.exit(f"lehra not found: {args.lehra}")
    nagma_text = args.lehra.read_text()

    print("========== sarangi training-cycle ==========")
    preprocess(args)
    backend, is_real = choose_backend(args.backend)
    combos = matrix(args)
    print(f"== rendering {len(combos)} loop(s) with backend={backend} "
          f"(avartans={args.avartans}) ==")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rendered: list[dict] = []
    for sa, bpm, seed in combos:
        params = _build_params(nagma_text, bpm=bpm, sa=sa, taal="teentaal",
                               avartans=args.avartans, seed=seed,
                               instrument="sarangi", fmt="wav")
        key = cache_key(params)
        out_path = OUT_DIR / f"{key}.wav"
        frames = render_one(params, backend=backend, out_path=out_path, verbose=False)
        print(f"  ok  sa={sa:<3} bpm={bpm:<6} seed={seed}  {key}.wav  ({frames} frames)")
        rendered.append({"sa": sa, "bpm": bpm, "seed": seed, "key": key,
                         "frames": frames, "path": out_path})

    if not args.no_copy:
        copy_to_desktop(rendered, backend, is_real, args.avartans, args.lehra)

    print("========== cycle complete ==========")
    if not is_real:
        print("NOTE: these are SINE previews (no trained model yet). After training "
              "on Colab and exporting to model/sarangi_ddsp/, re-run this script to "
              "get real sarangi renders.")
    return 0


def copy_to_desktop(rendered, backend, is_real, avartans, lehra) -> None:
    DESKTOP_RENDERS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "ddsp" if is_real else "sine-preview"
    manifest = [
        "sarangi-lehra-gen — render manifest",
        f"generated : {stamp}",
        f"backend   : {backend}  ({kind})",
        f"lehra     : {lehra.name}",
        f"avartans  : {avartans}",
        "",
        "Each file below is one lehra loop. Filenames encode the render params;",
        "the trailing 24-hex string is NiceNagma's cache key (the shipped name).",
        "Verify: loop it ~20x and listen for a click at sam; it should be seamless.",
        "",
    ]
    for r in rendered:
        nice = (f"sarangi_sa{_sa_slug(r['sa'])}_bpm{int(r['bpm'])}"
                f"_seed{r['seed']}_{kind}_{r['key']}.wav")
        dst = DESKTOP_RENDERS / nice
        shutil.copy2(r["path"], dst)
        manifest.append(f"{nice}")
        manifest.append(f"    Sa={r['sa']}  bpm={r['bpm']}  seed={r['seed']}  "
                        f"frames={r['frames']}  cache_key={r['key']}")
    (DESKTOP_RENDERS / "MANIFEST.txt").write_text("\n".join(manifest) + "\n")
    print(f"== copied {len(rendered)} render(s) -> {DESKTOP_RENDERS} "
          f"(+ MANIFEST.txt) ==")


if __name__ == "__main__":
    raise SystemExit(main())
