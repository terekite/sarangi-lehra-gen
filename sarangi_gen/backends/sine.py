"""Deterministic, numpy-only sine backend.

This is the always-available fallback: it makes the whole pipeline runnable and
testable without any trained model, and it powers the taal-lock verification
(the loop must be phase-continuous and drift-free). Correctness over realism.

Design
------
A single continuous-phase oscillator follows ``contour.f0_hz``. Rather than
building each note as an independent sine (which clicks at every pitch change),
we integrate instantaneous frequency into an absolute phase track::

    phase[n] = cumsum(2*pi * f0[n] / sr)

Because phase is accumulated, a change in f0 only changes the *slope* of the
phase, never its value, so the waveform is C0-continuous across pitch changes.
During rests (f0 <= 0 or NaN) the instantaneous frequency is forced to 0, so the
phase simply *holds* — when the next note resumes it continues from the frozen
phase and there is no click. Rests are silenced by zeroing amplitude, not phase.

A few weak, fixed harmonics are mixed in for a slightly less pure tone. Weights
are normalized so the summed peak stays under 1.0; everything is deterministic
(no randomness), so two identical inputs give byte-identical output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..model import Contour, RenderParams

if TYPE_CHECKING:
    pass

# Fixed harmonic amplitudes (fundamental + two weak overtones). Normalized below
# so that sum(|weights|) == _PEAK_HEADROOM, bounding the worst-case peak.
_HARMONIC_WEIGHTS = np.array([1.0, 0.12, 0.05], dtype=np.float64)
_PEAK_HEADROOM = 0.95


class SineBackend:
    """Continuous-phase additive-sine synthesizer (numpy only)."""

    def __init__(self) -> None:
        w = _HARMONIC_WEIGHTS.astype(np.float64)
        self._weights = w * (_PEAK_HEADROOM / float(w.sum()))

    def synthesize(self, contour: Contour, params: RenderParams) -> "np.ndarray":
        n = int(params.n_synth_overhang)
        sr = float(contour.sample_rate)

        f0 = self._fit(np.asarray(contour.f0_hz, dtype=np.float64), n)
        loud = self._fit(np.asarray(contour.loudness, dtype=np.float64), n)

        # Voiced where f0 is finite and strictly positive.
        voiced = np.isfinite(f0) & (f0 > 0.0)

        # Instantaneous frequency: 0 during rests so phase holds (no click on
        # resume). Non-finite f0 also maps to 0 to keep the integral clean.
        f_eff = np.where(voiced, f0, 0.0)
        phase = np.cumsum(2.0 * np.pi * f_eff / sr)

        # Additive harmonics on the shared continuous phase.
        sig = np.zeros(n, dtype=np.float64)
        for k, w in enumerate(self._weights, start=1):
            sig += w * np.sin(k * phase)

        # Amplitude envelope = loudness, gated to voiced samples so rests are
        # exactly silent.
        amp = np.where(voiced & np.isfinite(loud), np.clip(loud, 0.0, 1.0), 0.0)

        out = amp * sig
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(out, -1.0, 1.0, out=out)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _fit(arr: "np.ndarray", n: int) -> "np.ndarray":
        """Coerce a 1-D array to length ``n`` (pad with zeros / truncate).

        Contours are contracted to be exactly ``n_synth_overhang`` long; this is
        purely defensive so a mismatched input can never raise or run short.
        """
        arr = np.ravel(arr)
        if arr.shape[0] == n:
            return arr
        out = np.zeros(n, dtype=np.float64)
        m = min(arr.shape[0], n)
        out[:m] = arr[:m]
        return out
