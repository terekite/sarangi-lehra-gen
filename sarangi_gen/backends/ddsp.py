"""DDSP neural backend — loader + interface only.

This wraps a trained DDSP autoencoder that resynthesizes sarangi audio from an
(f0, loudness) contour at 16 kHz. The trained checkpoint does NOT exist yet, and
TensorFlow / ddsp are heavy optional dependencies, so this module must:

  * import cleanly with only numpy installed (``import sarangi_gen.synthesize``
    must never pull in tensorflow), and
  * fail with a clear, actionable error when asked to synthesize before a
    checkpoint has been trained.

All heavy imports are performed lazily *inside* methods. A later agent fills in
the inference body sketched in ``synthesize``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..model import Contour, RenderParams

if TYPE_CHECKING:
    import numpy as np

# Default location a training run is expected to write its exported model to.
_DEFAULT_CHECKPOINT = "model/sarangi_ddsp/"


class DdspBackend:
    """Runs DDSP inference from a trained checkpoint (when one is available)."""

    def __init__(self, checkpoint: str | None = None) -> None:
        # Store only; no TF/ddsp import here so construction is always cheap and
        # dependency-free.
        self.checkpoint = checkpoint if checkpoint is not None else _DEFAULT_CHECKPOINT
        self._model = None  # lazily built inference model / cache

    # ------------------------------------------------------------------ #
    def synthesize(self, contour: Contour, params: RenderParams) -> "np.ndarray":
        """Resynthesize audio from the contour via the trained DDSP model.

        Intended flow (to be implemented once a checkpoint exists):

          1. lazily ``import tensorflow as tf`` and ``import ddsp`` / gin config;
          2. build the autoencoder from the exported operative config and
             restore weights from ``self.checkpoint``;
          3. assemble the conditioning dict at 16 kHz —
             ``{"f0_hz": contour.f0_hz, "loudness_db": to_db(contour.loudness)}``
             framed to the model's frame rate;
          4. run the model, take the synthesized audio tensor, and coerce it to a
             mono float32 array of length ``params.n_synth_overhang`` at 16 kHz.

        Until then, fail loudly with an actionable message.
        """
        self._require_ready()
        # --- unreachable today; sketch of the real inference call ---------- #
        # tf, ddsp = self._lazy_deps()
        # model = self._load_model(tf, ddsp)
        # inputs = self._build_conditioning(contour, params)
        # audio = model(inputs, training=False)["audio_synth"]
        # return self._finalize(audio, params.n_synth_overhang)
        raise RuntimeError("DDSP backend reached an unreachable state")

    # ------------------------------------------------------------------ #
    def _require_ready(self) -> None:
        """Raise a clear error if the checkpoint or deps are unavailable."""
        if not self.checkpoint or not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"DDSP checkpoint not found at {self.checkpoint!r}; train it "
                "offline (see training/) or use backend='sine'."
            )
        try:
            self._lazy_deps()
        except ImportError as exc:
            raise RuntimeError(
                "DDSP backend requires 'tensorflow' and 'ddsp', which are not "
                "installed; install them to run neural synthesis or use "
                "backend='sine'."
            ) from exc

    @staticmethod
    def _lazy_deps():
        """Import the heavy deps on demand. Kept out of module import time."""
        import tensorflow as tf  # noqa: F401  (imported for availability check)
        import ddsp  # noqa: F401

        return tf, ddsp
