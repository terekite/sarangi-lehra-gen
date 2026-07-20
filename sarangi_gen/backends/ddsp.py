"""DDSP neural backend — runs the trained PyTorch sarangi timbre model.

This wraps a DDSP autoencoder that resynthesizes sarangi audio from an
(f0, loudness) contour at 16 kHz. The model is a compact PyTorch
harmonic-plus-filtered-noise synth (see ``_ddsp_torch.py``), trained locally by
``training/train_torch_ddsp.py``. PyTorch is a heavy, backend-only dependency,
so this module:

  * imports cleanly with only numpy installed (``import sarangi_gen.synthesize``
    must never pull in torch), performing all heavy imports lazily; and
  * fails with a clear, actionable error when asked to synthesize before a
    checkpoint has been trained or without torch installed.

The checkpoint dir (default ``model/sarangi_ddsp/``) holds ``config.json`` +
``ddsp_torch.pt``. Output is mono float32 of length exactly
``params.n_synth_overhang`` at 16 kHz, matching the sine backend's contract.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..model import Contour, RenderParams

if TYPE_CHECKING:
    import numpy as np

# Default location a training run writes its exported model to.
_DEFAULT_CHECKPOINT = "model/sarangi_ddsp/"

# Below this normalized loudness a frame is treated as a rest: the output is
# gated to silence so rests are exactly quiet and the loop seam stays clean
# (mirrors the sine backend's voiced-gating).
_SILENCE_GATE = 0.04


class DdspBackend:
    """Runs DDSP inference from a trained PyTorch checkpoint."""

    def __init__(self, checkpoint: str | None = None) -> None:
        # Store only; no torch import here so construction is always cheap and
        # dependency-free.
        self.checkpoint = checkpoint if checkpoint is not None else _DEFAULT_CHECKPOINT
        self._model = None   # cached (model, cfg) after first load
        self._cfg = None

    # ------------------------------------------------------------------ #
    def synthesize(self, contour: Contour, params: RenderParams) -> "np.ndarray":
        """Resynthesize audio from the contour via the trained DDSP model.

        Flow: downsample the per-sample contour to the model's frame rate, run
        the harmonic+noise synth, gate rests to silence, and coerce to a mono
        float32 array of length ``params.n_synth_overhang`` at 16 kHz. Noise is
        seeded from ``params.seed`` so identical params render identically
        (cache-key idempotency).
        """
        self._require_ready()
        import numpy as np
        import torch

        model, cfg = self._load_model()
        block = cfg.block_size
        n = int(params.n_synth_overhang)

        f0 = np.asarray(contour.f0_hz, dtype=np.float32)
        loud = np.asarray(contour.loudness, dtype=np.float32)
        f0 = self._fit(f0, n)
        loud = self._fit(loud, n)

        # --- per-sample -> per-frame control signals ---
        n_frames = max(1, n // block)
        m = n_frames * block
        f0_f = f0[:m].reshape(n_frames, block)
        loud_f = loud[:m].reshape(n_frames, block)
        # f0: mean over voiced samples in the block; hold last pitch through rests
        # so the oscillator never sees 0 Hz (amplitude is gated separately).
        voiced = f0_f > 0.0
        with np.errstate(invalid="ignore"):
            f0_frame = np.where(voiced.any(1),
                                (f0_f * voiced).sum(1) / np.maximum(voiced.sum(1), 1),
                                np.nan)
        f0_frame = self._forward_fill(f0_frame, default=220.0)
        loud_frame = loud_f.mean(1).astype(np.float32)

        # --- run the model (deterministic per seed) ---
        torch.manual_seed(int(params.seed) & 0x7FFFFFFF)
        pitch = torch.tensor(f0_frame, dtype=torch.float32)[None, :, None]
        loudness = torch.tensor(loud_frame, dtype=torch.float32)[None, :, None]
        with torch.no_grad():
            audio = model(pitch, loudness)[0].cpu().numpy().astype(np.float32)

        # --- silence gate at sample rate (rests exactly silent) ---
        gate = np.clip(loud[: len(audio)] / _SILENCE_GATE, 0.0, 1.0).astype(np.float32)
        audio = audio[: len(gate)] * gate

        # --- normalize peak and coerce to the exact contract length ---
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1e-6:
            audio = (audio / peak) * 0.9
        out = self._fit(audio, n)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def _load_model(self):
        if self._model is None:
            from ._ddsp_torch import load_synth
            self._model, self._cfg = load_synth(self.checkpoint, device="cpu")
        return self._model, self._cfg

    def _require_ready(self) -> None:
        """Raise a clear error if the checkpoint or deps are unavailable."""
        cfg_path = os.path.join(self.checkpoint, "config.json")
        wts_path = os.path.join(self.checkpoint, "ddsp_torch.pt")
        if not (os.path.exists(cfg_path) and os.path.exists(wts_path)):
            raise FileNotFoundError(
                f"DDSP checkpoint not found at {self.checkpoint!r} (need "
                "config.json + ddsp_torch.pt); train it with "
                "training/train_torch_ddsp.py or use backend='sine'."
            )
        try:
            self._lazy_deps()
        except ImportError as exc:
            raise RuntimeError(
                "DDSP backend requires 'torch', which is not installed; "
                "`pip install torch` to run neural synthesis or use backend='sine'."
            ) from exc

    @staticmethod
    def _lazy_deps():
        """Import the heavy deps on demand. Kept out of module import time."""
        import torch  # noqa: F401
        return (torch,)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fit(arr: "np.ndarray", n: int) -> "np.ndarray":
        """Coerce a 1-D array to length ``n`` (pad with zeros / truncate)."""
        import numpy as np
        arr = np.ravel(arr)
        if arr.shape[0] == n:
            return arr
        out = np.zeros(n, dtype=arr.dtype)
        m = min(arr.shape[0], n)
        out[:m] = arr[:m]
        return out

    @staticmethod
    def _forward_fill(a: "np.ndarray", default: float) -> "np.ndarray":
        """Replace NaNs by the last valid value (rests hold the prior pitch)."""
        import numpy as np
        a = a.astype(np.float32)
        last = default
        for i in range(len(a)):
            if np.isnan(a[i]):
                a[i] = last
            else:
                last = a[i]
        return a
