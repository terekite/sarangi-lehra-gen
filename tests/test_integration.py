"""End-to-end integration tests for the full render pipeline.

Exercises the wired pipeline on the proposed-teentaal lehra with the
deterministic `sine` backend (no TensorFlow needed):

    parse_nagma -> build_events -> render_contour -> synthesize -> master

Covers the render-pipeline correctness gates (plan §10):
  * loop-length + WAV format (§10.2)
  * cache-key output naming (§7)
  * taal lock: structural onsets land on exact sample positions, no drift
    over `avartans` (§10.4)
  * determinism: two renders are byte-identical (§6)
  * cache-key golden value (§10.5)
"""

from __future__ import annotations

import hashlib

import soundfile as sf

from sarangi_gen.cache_key import cache_key, output_filename
from sarangi_gen.contour import render_contour
from sarangi_gen.model import RenderParams
from sarangi_gen.parse import build_events, parse_nagma
from sarangi_gen.post import master
from sarangi_gen.synthesize import synthesize

GOLDEN = "634b772f8d8a9b0c94d73f02"


def _render(params: RenderParams, out_path) -> int:
    doc = parse_nagma(params.nagma_text, taal=params.taal)
    events = build_events(doc, params)
    contour = render_contour(events, params)
    mono = synthesize(contour, params, backend="sine")
    return master(mono, params, str(out_path))


def test_full_pipeline_writes_valid_loop(golden_params, tmp_path):
    out = tmp_path / output_filename(golden_params)
    frames = _render(golden_params, out)

    assert frames == golden_params.loop_length_samples
    assert out.exists()

    info = sf.info(str(out))
    assert info.samplerate == 44100
    assert info.channels == 2
    assert info.subtype == "PCM_16"
    assert info.frames == golden_params.loop_length_samples


def test_output_filename_is_cache_key(golden_params):
    assert output_filename(golden_params) == f"{cache_key(golden_params)}.wav"
    assert output_filename(golden_params) == f"{GOLDEN}.wav"


def test_cache_key_golden(golden_params):
    assert cache_key(golden_params) == GOLDEN


def test_determinism_byte_identical(golden_params, tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    _render(golden_params, a)
    _render(golden_params, b)
    ha = hashlib.sha256(a.read_bytes()).hexdigest()
    hb = hashlib.sha256(b.read_bytes()).hexdigest()
    assert ha == hb, "two identical renders must be byte-identical"


def _assert_taal_lock(params: RenderParams) -> int:
    """Every structural onset sits on an exact multiple of matra_dur_s, with no
    accumulated drift across avartans. Returns the count of structural events."""
    doc = parse_nagma(params.nagma_text, taal=params.taal)
    events = build_events(doc, params)

    sr = params.sample_rate_out
    matra_dur_s = params.matra_dur_s
    matra_samples = matra_dur_s * sr
    matra_count = params.matra_count

    structural = [e for e in events if e.structural]
    assert structural, "expected at least some structural onsets"

    for e in structural:
        # Global matra index this onset must land on (no jitter on structural).
        global_matra = e.avartan * matra_count + e.matra
        expected_samples = global_matra * matra_samples
        actual_samples = e.start_s * sr

        # start_s is built in seconds and never accumulated in samples, so the
        # onset must sit on the exact grid position to sub-sample precision.
        assert abs(actual_samples - expected_samples) < 1e-3, (
            f"structural onset drift: avartan={e.avartan} matra={e.matra} "
            f"actual={actual_samples} expected={expected_samples}"
        )
        # And it corresponds to an integer number of matras from sam.
        ratio = e.start_s / matra_dur_s
        assert abs(ratio - round(ratio)) < 1e-6
        assert round(ratio) == global_matra

    return len(structural)


def test_taal_lock_no_drift_4_avartans(proposed_teentaal_text):
    params = RenderParams(
        nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=4, seed=3,
    )
    n = _assert_taal_lock(params)
    # Sam (matra 0) of every cycle must be present as a structural onset.
    assert n >= params.avartans


def test_taal_lock_no_drift_8_avartans(proposed_teentaal_text):
    params = RenderParams(
        nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=8, seed=3,
    )
    n = _assert_taal_lock(params)
    assert n >= params.avartans


def test_taal_lock_onset_energy_at_sam(golden_params, tmp_path):
    """The rendered audio actually carries energy at each sam (loop head)."""
    out = tmp_path / "sam.wav"
    _render(golden_params, out)
    audio, sr = sf.read(str(out))
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    avartan_samples = int(round(golden_params.avartan_dur_s * sr))
    win = int(0.02 * sr)  # 20 ms window
    for a in range(golden_params.avartans):
        idx = a * avartan_samples
        seg = mono[idx:idx + win]
        rms = float((seg ** 2).mean() ** 0.5)
        assert rms > 1e-4, f"no onset energy near sam of avartan {a} (rms={rms})"
