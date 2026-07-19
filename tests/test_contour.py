"""Tests for the sarangi contour stage (events -> f0[] + loudness[])."""

from __future__ import annotations

import numpy as np
import pytest

from sarangi_gen.contour import render_contour
from sarangi_gen.model import SAMPLE_RATE_SYNTH, Event, RenderParams

SR = SAMPLE_RATE_SYNTH


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _params(bpm: float = 120.0, avartans: int = 2, seed: int = 7) -> RenderParams:
    return RenderParams(
        nagma_text="# synthetic\nS R G m",
        bpm=bpm,
        sa="C",
        avartans=avartans,
        seed=seed,
    )


# A per-matra ascending/undulating pattern (16 matras) with plenty of distinct
# consecutive pitches, so meend glides are exercised.
_PATTERN = [0, 2, 4, 5, 7, 5, 4, 2, 0, 2, 4, 5, 7, 9, 11, 12]


def _lehra_events(params: RenderParams, sa_midi: int = 60) -> list[Event]:
    """One note per matra across all avartans; structural on vibhag boundaries."""
    matra_dur = params.matra_dur_s
    events: list[Event] = []
    for a in range(params.avartans):
        for m in range(16):
            gidx = a * 16 + m
            events.append(
                Event(
                    start_s=gidx * matra_dur,
                    dur_s=matra_dur,
                    midi=sa_midi + _PATTERN[m],
                    velocity=90,
                    matra=m,
                    avartan=a,
                    structural=(m % 4 == 0),
                    swell=0.0,
                )
            )
    return events


def test_shape_dtype_and_ranges():
    params = _params()
    events = _lehra_events(params)
    c = render_contour(events, params)

    assert c.f0_hz.dtype == np.float32
    assert c.loudness.dtype == np.float32
    assert c.f0_hz.shape == (params.n_synth_overhang,)
    assert c.loudness.shape == (params.n_synth_overhang,)
    assert np.all(np.isfinite(c.f0_hz))
    assert np.all(np.isfinite(c.loudness))
    assert np.all(c.f0_hz >= 0.0)
    assert np.all(c.loudness >= 0.0)
    assert np.all(c.loudness <= 1.0 + 1e-4)


def test_taal_lock_structural_onsets_hit_exact_pitch():
    params = _params()
    events = _lehra_events(params)
    c = render_contour(events, params)

    for e in events:
        if not e.structural:
            continue
        idx = int(round(e.start_s * SR))
        expected = _midi_to_hz(e.midi)
        assert abs(float(c.f0_hz[idx]) - expected) < 0.6, (
            f"structural onset at matra {e.matra}/avartan {e.avartan} "
            f"missed pitch: {c.f0_hz[idx]} vs {expected}"
        )


def test_glide_present_between_distinct_pitches():
    params = _params()
    events = _lehra_events(params)
    c = render_contour(events, params)

    # Find a contiguous pair of distinct-pitch events and inspect the samples
    # just before the second onset: a pure step would contain no strictly
    # intermediate f0 values, a meend glide does.
    found_pair = False
    for a, b in zip(events, events[1:]):
        if a.midi == b.midi:
            continue
        found_pair = True
        s0 = int(round(b.start_s * SR))
        lo = min(_midi_to_hz(a.midi), _midi_to_hz(b.midi))
        hi = max(_midi_to_hz(a.midi), _midi_to_hz(b.midi))
        window = c.f0_hz[max(0, s0 - 200):s0]
        strictly_between = np.sum((window > lo + 1e-3) & (window < hi - 1e-3))
        if strictly_between > 0:
            break
    else:
        pytest.fail("no glide (intermediate f0) found before any onset")
    assert found_pair


def test_determinism():
    params = _params()
    events = _lehra_events(params)
    c1 = render_contour(events, params)
    c2 = render_contour(events, params)
    assert np.array_equal(c1.f0_hz, c2.f0_hz)
    assert np.array_equal(c1.loudness, c2.loudness)


def test_rest_gap_is_unvoiced():
    params = _params()
    events = _lehra_events(params)
    # Drop the event on matra 5 of avartan 0, leaving a rest there.
    gap_start = 5 * params.matra_dur_s
    events = [e for e in events if not (e.avartan == 0 and e.matra == 5)]
    c = render_contour(events, params)

    # Sample the middle of the now-empty matra cell.
    idx = int(round((gap_start + 0.5 * params.matra_dur_s) * SR))
    assert c.f0_hz[idx] == 0.0
    assert c.loudness[idx] == 0.0


def test_overhang_continues_tail():
    params = _params()
    events = _lehra_events(params)
    c = render_contour(events, params)
    # The loop body ends voiced (last matra sounds to the seam), so the overhang
    # must not go abruptly silent.
    assert c.f0_hz[params.n_synth] > 0.0
    assert c.f0_hz[params.n_synth_overhang - 1] > 0.0
    # ...and the tail should decay (release), not hold flat at full level.
    assert c.loudness[params.n_synth] > c.loudness[params.n_synth_overhang - 1]


def test_empty_events():
    params = _params()
    c = render_contour([], params)
    assert c.f0_hz.shape == (params.n_synth_overhang,)
    assert np.all(c.f0_hz == 0.0)
    assert np.all(c.loudness == 0.0)
