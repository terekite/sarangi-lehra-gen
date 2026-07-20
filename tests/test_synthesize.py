"""Tests for the synthesize stage and its sine/ddsp backends.

The sine backend is the always-available fallback, so it gets the bulk of the
coverage: shape/dtype/finiteness, click-free continuous phase, silent rests,
and determinism. The ddsp backend is checked only for its interface contract
(clear error without a checkpoint) and — crucially — that importing the module
does not require TensorFlow.
"""

from __future__ import annotations

import numpy as np
import pytest

from sarangi_gen.model import Contour, RenderParams
from sarangi_gen.synthesize import get_backend, synthesize


def _params() -> RenderParams:
    # bpm/taal/avartans drive n_synth_overhang; keep it modest but realistic.
    return RenderParams(
        nagma_text="x",
        bpm=200.0,
        sa="C",
        avartans=1,
        seed=7,
    )


def _contour(params: RenderParams):
    """A hand-built contour: A4 for the first third, up to a higher pitch for
    the second third, then a rest (f0 <= 0) for the final third."""
    n = params.n_synth_overhang
    third = n // 3

    f0 = np.zeros(n, dtype=np.float32)
    loud = np.zeros(n, dtype=np.float32)

    f0[:third] = 440.0                 # A4
    f0[third : 2 * third] = 587.33     # ~D5 (a clear pitch change)
    f0[2 * third :] = 0.0              # rest

    loud[: 2 * third] = 0.8            # voiced regions loud
    loud[2 * third :] = 0.0            # rest silent

    # Sprinkle a NaN in the rest region too — must be treated as unvoiced.
    f0[2 * third + 5] = np.nan
    return Contour(f0_hz=f0, loudness=loud), third


def test_sine_shape_dtype_and_range():
    params = _params()
    contour, _ = _contour(params)

    out = synthesize(contour, params, backend="sine")

    assert out.dtype == np.float32
    assert out.shape == (params.n_synth_overhang,)
    assert np.all(np.isfinite(out))
    assert np.abs(out).max() <= 1.0 + 1e-4


def test_sine_continuous_phase_no_clicks():
    """Within voiced regions (including across the pitch change) there should be
    no large adjacent-sample jump — a proxy for click-free continuous phase."""
    params = _params()
    contour, third = _contour(params)

    out = synthesize(contour, params, backend="sine")

    # Examine the whole voiced span [0, 2*third), which contains the internal
    # 440 -> 587 Hz change. A pure phase discontinuity there would show up as a
    # jump on the order of the amplitude (~0.8); continuous phase keeps every
    # step well below that.
    voiced = out[: 2 * third]
    max_step = np.abs(np.diff(voiced)).max()
    assert max_step < 0.4


def test_sine_silent_rest():
    params = _params()
    contour, third = _contour(params)

    out = synthesize(contour, params, backend="sine")

    rest = out[2 * third :]
    assert np.allclose(rest, 0.0, atol=1e-6)


def test_sine_deterministic():
    params = _params()
    contour, _ = _contour(params)

    a = synthesize(contour, params, backend="sine")
    b = synthesize(contour, params, backend="sine")
    assert np.array_equal(a, b)


def test_ddsp_import_is_lazy_and_errors_clearly():
    # Importing the synthesize module (done at top of file) must not pull the
    # heavy, backend-only dep (torch) — the base package stays light.
    import sys

    assert "torch" not in sys.modules

    # With no checkpoint at the given path, the backend must fail loudly and
    # actionably (before any torch import), not silently.
    backend = get_backend("ddsp", checkpoint="model/__does_not_exist__/")
    with pytest.raises((FileNotFoundError, RuntimeError)) as exc:
        backend.synthesize(*_contour_for_ddsp())
    assert "sine" in str(exc.value).lower() or "checkpoint" in str(exc.value).lower()


def test_ddsp_synthesizes_when_trained():
    # Positive path: when torch + a trained checkpoint are present, the ddsp
    # backend returns finite mono audio of the exact contract length, and is
    # deterministic per seed (cache-key idempotency). Skipped in a torch-less
    # env or before a model has been trained.
    import os

    pytest.importorskip("torch")
    if not os.path.exists("model/sarangi_ddsp/ddsp_torch.pt"):
        pytest.skip("no trained ddsp checkpoint")

    contour, params = _contour_for_ddsp()
    a = synthesize(contour, params, backend="ddsp")
    b = synthesize(contour, params, backend="ddsp")
    assert a.shape == (params.n_synth_overhang,)
    assert a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.max(np.abs(a)) > 0.0          # not silence
    assert np.array_equal(a, b)             # deterministic per seed


def _contour_for_ddsp():
    params = _params()
    contour, _ = _contour(params)
    return contour, params


def test_unknown_backend_raises():
    params = _params()
    contour, _ = _contour(params)
    with pytest.raises(ValueError):
        synthesize(contour, params, backend="nope")
