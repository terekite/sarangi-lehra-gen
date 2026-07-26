#!/usr/bin/env python3
"""Train the PyTorch DDSP sarangi timbre model — locally, on CPU/MPS, no TF.

Pipeline
--------
  1. FEATURES  For every 16 kHz mono clip in ``training/data/processed/`` extract
     per-frame f0 (torchcrepe) + A-weighted loudness. Cached to
     ``training/data/ddsp_features/`` so a resume never recomputes CREPE.
  2. TRAIN     Self-supervised reconstruction: the model resynthesizes each audio
     window from its own (f0, loudness) and is optimized against a multi-scale
     STFT loss. Runs on Metal (MPS) if available, else CPU.
  3. CHECKPOINT  Full training state (model + optimizer + step + RNG + config) is
     written **atomically every --save-every steps** to ``model/sarangi_ddsp/``.
     Re-running **auto-resumes** from the latest checkpoint — stop any time
     (Ctrl-C, or just kill the process / close the machine) and rerun to continue.

Usage
-----
    python training/train_torch_ddsp.py                 # fresh or auto-resume
    python training/train_torch_ddsp.py --steps 20000   # target step count
    python training/train_torch_ddsp.py --fresh         # ignore any checkpoint
    python training/train_torch_ddsp.py --device cpu    # force CPU

The exported checkpoint (``model/sarangi_ddsp/``) is exactly what the runtime
``ddsp`` backend loads — after training, ``python scripts/run_training_cycle.py``
auto-detects it and renders real sarangi.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sarangi_gen.backends._ddsp_torch import (  # noqa: E402
    CKPT_CONFIG, CKPT_WEIGHTS, DDSPConfig, DDSPSynth,
)

# Data is gitignored and lives in the MAIN checkout; allow an override so a
# worktree can point back at it.
DEFAULT_PROCESSED = os.environ.get(
    "SARANGI_PROCESSED_DIR", os.path.join(REPO_ROOT, "training", "data", "processed"))
FEATURE_DIR = os.path.join(REPO_ROOT, "training", "data", "ddsp_features")
CKPT_DIR = os.path.join(REPO_ROOT, "model", "sarangi_ddsp")


# --------------------------------------------------------------------------- #
# Feature extraction (cached)
# --------------------------------------------------------------------------- #
def extract_loudness(y: np.ndarray, sr: int, block_size: int, n_fft: int = 2048):
    """Per-frame A-weighted loudness in dB (roughly -80..0)."""
    import librosa
    S = librosa.stft(y, n_fft=n_fft, hop_length=block_size, center=True)
    power_db = librosa.amplitude_to_db(np.abs(S), ref=1.0)          # [freq, frames]
    # Floor the A-weighting (its DC bin is -inf) so the per-frame mean is finite.
    a_weight = np.maximum(librosa.A_weighting(
        librosa.fft_frequencies(sr=sr, n_fft=n_fft)), -80.0)
    return (power_db + a_weight[:, None]).mean(axis=0)              # [frames]


def extract_f0(y: np.ndarray, sr: int, block_size: int, device: str,
               model: str = "tiny"):
    """CREPE f0 (Hz) + periodicity, unvoiced frames interpolated for a clean osc.

    ``tiny`` on MPS is ~20x realtime with essentially the same f0 as ``full`` for
    a monophonic bowed instrument — a good speed/accuracy trade for this dataset.
    """
    import torchcrepe
    audio = torch.tensor(y, dtype=torch.float32, device=device)[None]
    f0, per = torchcrepe.predict(
        audio, sr, hop_length=block_size, fmin=50.0, fmax=1000.0,
        model=model, batch_size=1024, device=device, return_periodicity=True)
    f0 = f0[0].cpu().numpy()
    per = per[0].cpu().numpy()
    # Interpolate low-confidence (unvoiced/silent) frames so the harmonic
    # oscillator never sees garbage f0; amplitude there is gated by loudness.
    voiced = per > 0.5
    if voiced.any():
        idx = np.arange(len(f0))
        f0 = np.interp(idx, idx[voiced], f0[voiced])
    else:
        f0 = np.full_like(f0, 220.0)
    return f0.astype(np.float32)


def build_features(processed_dir: str, cfg: DDSPConfig, device: str) -> list[str]:
    os.makedirs(FEATURE_DIR, exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(processed_dir, "*.wav")))
    if not wavs:
        sys.exit(f"[features] no wavs in {processed_dir} — run preprocessing first.")
    import soundfile as sf
    paths = []
    for w in wavs:
        name = os.path.splitext(os.path.basename(w))[0]
        out = os.path.join(FEATURE_DIR, name + ".npz")
        paths.append(out)
        if os.path.exists(out):
            print(f"[features] cached {name}")
            continue
        y, sr = sf.read(w, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != cfg.sampling_rate:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sampling_rate)
            sr = cfg.sampling_rate
        n_frames = len(y) // cfg.block_size
        y = y[: n_frames * cfg.block_size]
        print(f"[features] extracting {name} ({n_frames} frames, "
              f"{len(y)/sr:.1f}s) ...")
        loud = extract_loudness(y, sr, cfg.block_size)[:n_frames]
        f0 = extract_f0(y, sr, cfg.block_size, device)[:n_frames]
        m = min(len(loud), len(f0), n_frames)
        np.savez(out, audio=y[: m * cfg.block_size],
                 f0=f0[:m], loud_db=loud[:m].astype(np.float32))
    return paths


def load_dataset(paths: list[str], cfg: DDSPConfig):
    files = [dict(np.load(p)) for p in paths]
    all_db = np.concatenate([f["loud_db"] for f in files])
    # Robust min-max normalization of loudness to [0,1] (shared by inference).
    cfg.loudness_db_lo = float(np.percentile(all_db, 5))
    cfg.loudness_db_hi = float(np.percentile(all_db, 95))
    span = max(cfg.loudness_db_hi - cfg.loudness_db_lo, 1e-3)
    for f in files:
        f["loud"] = np.clip((f["loud_db"] - cfg.loudness_db_lo) / span, 0.0, 1.0
                            ).astype(np.float32)
    return files


def sample_batch(files, cfg, win_frames: int, batch: int, rng: np.random.Generator,
                 device: str):
    B_audio, B_f0, B_loud = [], [], []
    usable = [f for f in files if len(f["f0"]) > win_frames]
    for _ in range(batch):
        f = usable[rng.integers(len(usable))]
        F = len(f["f0"])
        s = int(rng.integers(0, F - win_frames))
        a0 = s * cfg.block_size
        a1 = (s + win_frames) * cfg.block_size
        B_audio.append(f["audio"][a0:a1])
        B_f0.append(f["f0"][s:s + win_frames])
        B_loud.append(f["loud"][s:s + win_frames])
    audio = torch.tensor(np.stack(B_audio), dtype=torch.float32, device=device)
    f0 = torch.tensor(np.stack(B_f0), dtype=torch.float32, device=device)[..., None]
    loud = torch.tensor(np.stack(B_loud), dtype=torch.float32, device=device)[..., None]
    return audio, f0, loud


# --------------------------------------------------------------------------- #
# Multi-scale STFT reconstruction loss
# --------------------------------------------------------------------------- #
_SCALES = [2048, 1024, 512, 256, 128, 64]


def _stft_mag(y: torch.Tensor, n_fft: int) -> torch.Tensor:
    win = torch.hann_window(n_fft, device=y.device)
    S = torch.stft(y, n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft,
                   window=win, center=True, return_complex=True)
    return S.abs()


def spectral_loss(y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
    total = y.new_zeros(())
    for s in _SCALES:
        Sy, Syh = _stft_mag(y, s), _stft_mag(y_hat, s)
        lin = (Sy - Syh).abs().mean()
        log = (torch.log(Sy + 1e-7) - torch.log(Syh + 1e-7)).abs().mean()
        total = total + lin + log
    return total


# --------------------------------------------------------------------------- #
# Checkpoint I/O (atomic; full resumable state)
# --------------------------------------------------------------------------- #
def save_checkpoint(model, opt, step, cfg, loss_ema, rng):
    os.makedirs(CKPT_DIR, exist_ok=True)
    cfg.to_json(os.path.join(CKPT_DIR, CKPT_CONFIG))
    state = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": step,
        "loss_ema": loss_ema,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": rng.bit_generator.state,
        "config": cfg.__dict__,
    }
    tmp = os.path.join(CKPT_DIR, CKPT_WEIGHTS + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, os.path.join(CKPT_DIR, CKPT_WEIGHTS))


def try_resume(model, opt, cfg, rng):
    path = os.path.join(CKPT_DIR, CKPT_WEIGHTS)
    if not os.path.exists(path):
        return 0, None
    state = torch.load(path, map_location="cpu")
    # Resume-size guard: a checkpoint only loads into an identically-shaped
    # graph. If the architecture knobs differ from this run's config, bail with
    # a clear message instead of a cryptic state_dict shape error.
    ck = state["config"]
    arch = ("hidden_size", "n_harmonic", "n_bands", "gru_layers",
            "block_size", "sampling_rate")
    mismatch = {k: (ck.get(k), getattr(cfg, k))
                for k in arch if ck.get(k) != getattr(cfg, k)}
    if mismatch:
        detail = ", ".join(f"{k}: ckpt={a} vs requested={b}"
                           for k, (a, b) in mismatch.items())
        sys.exit(
            f"[resume] checkpoint architecture differs from requested config "
            f"({detail}).\n"
            f"         A bigger/smaller model cannot resume from this "
            f"checkpoint — pass --fresh to train it from scratch (the old "
            f"checkpoint stays on disk / is backed up by its git tag).")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    rng.bit_generator.state = state["numpy_rng"]
    for k in ("loudness_db_lo", "loudness_db_hi"):
        setattr(cfg, k, state["config"][k])
    return int(state["step"]), state.get("loss_ema")


# --------------------------------------------------------------------------- #
def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        try:  # verify FFT ops (noise branch) actually run on MPS
            x = torch.randn(2, 8, device="mps")
            torch.fft.irfft(torch.fft.rfft(x))
            return "mps"
        except Exception as e:  # noqa: BLE001
            print(f"[device] MPS present but FFT failed ({e}); using CPU.")
    return "cpu"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=20000, help="Target total steps.")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--window-s", type=float, default=2.0, help="Train window seconds.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-every", type=int, default=250, help="Checkpoint interval.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--processed-dir", default=DEFAULT_PROCESSED)
    p.add_argument("--fresh", action="store_true", help="Ignore any checkpoint.")
    # Model-size knobs (defaults match DDSPConfig — the shipped 80k model).
    # A bigger model (e.g. --hidden-size 512 --gru-layers 2) needs --fresh;
    # the resume guard refuses to load a differently-shaped checkpoint.
    p.add_argument("--hidden-size", type=int, default=DDSPConfig.hidden_size,
                   help="GRU/MLP hidden width (bigger = more capacity).")
    p.add_argument("--gru-layers", type=int, default=DDSPConfig.gru_layers,
                   help="Number of stacked GRU layers.")
    p.add_argument("--n-harmonic", type=int, default=DDSPConfig.n_harmonic,
                   help="Additive-synth harmonic count.")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    cfg = DDSPConfig(hidden_size=args.hidden_size, gru_layers=args.gru_layers,
                     n_harmonic=args.n_harmonic)
    win_frames = int(round(args.window_s * cfg.sampling_rate / cfg.block_size))
    print(f"== torch-ddsp training | device={device} | window={win_frames} frames "
          f"| batch={args.batch} | target={args.steps} steps ==")

    # 1. features (cached) + dataset — crepe 'tiny' runs happily on MPS too.
    paths = build_features(args.processed_dir, cfg, device)
    files = load_dataset(paths, cfg)
    total_min = sum(len(f["f0"]) for f in files) * cfg.block_size / cfg.sampling_rate / 60
    print(f"== dataset: {len(files)} clip(s), {total_min:.1f} min, "
          f"loudness dB [{cfg.loudness_db_lo:.1f}, {cfg.loudness_db_hi:.1f}] ==")

    # 2. model + optimizer, resume if possible
    model = DDSPSynth(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    start_step, loss_ema = (0, None)
    if not args.fresh:
        start_step, loss_ema = try_resume(model, opt, cfg, rng)
        if start_step:
            print(f"== resumed from checkpoint at step {start_step} "
                  f"(loss_ema={loss_ema:.3f}) ==")
    if start_step >= args.steps:
        print(f"== already at/{start_step} >= target {args.steps}; nothing to do. "
              f"Raise --steps to train more. ==")
        return 0

    # 3. train loop
    model.train()
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        audio, f0, loud = sample_batch(files, cfg, win_frames, args.batch, rng, device)
        y_hat = model(f0, loud)
        n = min(y_hat.shape[-1], audio.shape[-1])
        loss = spectral_loss(audio[..., :n], y_hat[..., :n])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        lv = float(loss.detach().cpu())
        loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
        if step % args.log_every == 0:
            rate = (step - start_step) / max(time.time() - t0, 1e-6)
            eta = (args.steps - step) / max(rate, 1e-6) / 60
            print(f"  step {step:6d}/{args.steps}  loss={lv:7.3f}  "
                  f"ema={loss_ema:7.3f}  {rate:4.1f} it/s  eta {eta:5.1f} min",
                  flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_checkpoint(model, opt, step, cfg, loss_ema, rng)
            print(f"  [ckpt] saved at step {step} -> {CKPT_DIR}", flush=True)

    print(f"== done. final loss_ema={loss_ema:.3f}. checkpoint at {CKPT_DIR} ==")
    print("== next: python scripts/run_training_cycle.py --skip-preprocess "
          "(auto-detects ddsp) ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
