"""Tests for sarangi_gen.post.master — the exact-length seamless loop gate.

Covers: output frame count / samplerate / channels / subtype; seam continuity
(the equal-power wrap makes the loop clickless); the short-input zero-pad path;
and byte-for-byte determinism across two runs on identical input.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from sarangi_gen.model import RenderParams, SAMPLE_RATE_SYNTH
from sarangi_gen.post import master


def _params(**kw) -> RenderParams:
    """A small hand-built RenderParams (short loop keeps tests fast)."""
    base = dict(nagma_text="x", bpm=160.0, sa="C", avartans=1)
    base.update(kw)
    return RenderParams(**base)


def _decaying_sine(params: RenderParams, freq_hz: float = 220.0) -> np.ndarray:
    """A decaying sine of length n_synth_overhang at the 16 kHz synth rate.

    Uses an integer number of cycles across the loop body so the tone itself is
    phase-continuous across the loop point; the overhang is its natural
    continuation, which is what the seam wrap folds back onto the head.
    """
    n = params.n_synth_overhang
    body = params.n_synth
    # snap freq to a whole number of cycles across the loop body -> continuous
    cycles = max(1, round(freq_hz * body / SAMPLE_RATE_SYNTH))
    t = np.arange(n)
    tone = np.sin(2 * np.pi * cycles * t / body)
    env = np.exp(-3.0 * t / body)  # decay across the whole buffer
    return (0.5 * tone * env).astype(np.float64)


def test_output_shape_and_format(tmp_path):
    params = _params()
    x = _decaying_sine(params)
    out = tmp_path / "loop.wav"

    written = master(x, params, str(out))

    assert written == params.loop_length_samples
    info = sf.info(str(out))
    assert info.frames == params.loop_length_samples
    assert info.samplerate == params.sample_rate_out
    assert info.channels == 2
    assert info.subtype == "PCM_16"


def test_seam_is_clickless(tmp_path):
    """The wrap-around discontinuity must be in line with interior deltas."""
    params = _params()
    x = _decaying_sine(params)
    out = tmp_path / "loop.wav"
    master(x, params, str(out))

    data, _ = sf.read(str(out), always_2d=True)
    assert data.shape[0] == params.loop_length_samples

    # interior sample-to-sample deltas (per channel)
    interior = np.abs(np.diff(data, axis=0))
    # the seam: last frame wrapping to the first frame of the next repeat
    seam = np.abs(data[0] - data[-1])

    # a clickless loop: the seam jump is no worse than a typical loud interior
    # transition. Compare to a high percentile (not the max, which can be a
    # lone spike) so the assertion is meaningful.
    ref = np.percentile(interior, 99.9, axis=0)
    assert np.all(seam <= 5.0 * ref + 1e-6), (seam, ref)


def test_short_input_zero_pads(tmp_path):
    params = _params()
    # a buffer far shorter than the loop body -> must zero-pad to exactly n
    short = 0.3 * np.sin(np.linspace(0, 20 * np.pi, 500))
    out = tmp_path / "short.wav"

    written = master(short, params, str(out))

    assert written == params.loop_length_samples
    info = sf.info(str(out))
    assert info.frames == params.loop_length_samples
    assert info.channels == 2


def test_determinism(tmp_path):
    params = _params()
    x = _decaying_sine(params)
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"

    master(x, params, str(a))
    master(x.copy(), params, str(b))

    da, sra = sf.read(str(a), always_2d=True)
    db, srb = sf.read(str(b), always_2d=True)
    assert sra == srb
    assert np.array_equal(da, db)
    # and the raw bytes match too
    assert a.read_bytes() == b.read_bytes()


def test_flac_output(tmp_path):
    params = _params(format="flac")
    x = _decaying_sine(params)
    out = tmp_path / "loop.flac"

    master(x, params, str(out))

    info = sf.info(str(out))
    assert info.format == "FLAC"
    assert info.channels == 2
    assert info.subtype == "PCM_16"
    assert info.frames == params.loop_length_samples
