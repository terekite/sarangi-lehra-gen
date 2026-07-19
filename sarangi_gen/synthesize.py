"""Contour -> mono audio at 16 kHz, via a pluggable backend.

Two backends are available:

  * ``"sine"``  — deterministic, numpy-only, always works (the fallback that
    keeps the pipeline runnable/testable and powers taal-lock verification);
  * ``"ddsp"``  — neural resynthesis from a trained checkpoint (interface only
    today; imports TensorFlow/ddsp lazily so this module always imports).

Backends conform to the :class:`Backend` protocol. Output is mono float32 of
length exactly ``params.n_synth_overhang`` with values roughly in [-1, 1].
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .backends.ddsp import DdspBackend
from .backends.sine import SineBackend
from .model import Contour, RenderParams

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Backend", "get_backend", "synthesize"]


@runtime_checkable
class Backend(Protocol):
    """A synthesis backend: contour + params -> mono float32 audio @16 kHz."""

    def synthesize(self, contour: Contour, params: RenderParams) -> "np.ndarray":
        ...


def get_backend(name: str, *, checkpoint: str | None = None) -> Backend:
    """Resolve a backend by name.

    ``name`` must be one of ``{"sine", "ddsp"}``. ``checkpoint`` is only used by
    the ddsp backend (defaults to ``model/sarangi_ddsp/`` when None).
    """
    if name == "sine":
        return SineBackend()
    if name == "ddsp":
        return DdspBackend(checkpoint)
    raise ValueError(f"unknown backend {name!r}; expected 'sine' or 'ddsp'")


def synthesize(
    contour: Contour, params: RenderParams, *, backend: str = "sine"
) -> "np.ndarray":
    """Synthesize ``contour`` to mono float32 audio using ``backend``.

    The result has length exactly ``params.n_synth_overhang`` at 16 kHz.
    """
    return get_backend(backend).synthesize(contour, params)
