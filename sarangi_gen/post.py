"""Loop mastering: turn a raw 16 kHz sarangi render into a gapless, exact-length
stereo loop at the output rate.

Ported faithfully from NiceNagma's ``render/src/nagma_render/master.py`` so the
output stays byte-shape compatible with its loops. The steps, in order:

  1. Resample 16 kHz -> ``params.sample_rate_out`` (polyphase, reduced integer
     ratio, deterministic).
  2. Mono -> stereo BEFORE the seam wrap and reverb, so both channels get the
     wrap and the reverb decorrelates L/R (matching master.py's channel loop).
  3. Seam wrap / trim / pad to exactly ``params.loop_length_samples``. The
     natural overhang past the loop point is mixed back onto the head with an
     equal-power fade so cycle N's tail resolves into the sam with no click.
  4. Room presence: CIRCULAR convolution reverb with a procedurally generated,
     deterministic stereo IR (length-preserving, keeps the loop exact/seamless).
  5. Loudness normalize to -16 LUFS (pyloudnorm) or peak to -1 dBFS as fallback.
  6. Write 16-bit PCM stereo WAV (or FLAC).

The output length is asserted to equal ``loop_length_samples`` exactly — the
render pipeline's primary correctness gate.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import resample_poly

from .model import RenderParams, SAMPLE_RATE_SYNTH

# Max overhang (seconds) to wrap from the tail onto the head. == OVERHANG_S.
_MAX_WRAP_S = 0.6

# Convolution room reverb: subtle "in the room with you" presence. Applied
# circularly so the loop stays exact-length and seamless. seed keeps it
# deterministic/cacheable. Set wet=0 to disable.
REVERB = {"seconds": 0.5, "wet": 0.08, "seed": 1}


def master(mono_16k: "np.ndarray", params: RenderParams, out_path: str) -> int:
    """Master ``mono_16k`` into a seamless stereo loop at ``out_path``.

    Args:
        mono_16k: mono float audio at 16 kHz, length ``params.n_synth_overhang``
            (loop body + 0.6 s overhang).
        params: render parameters; source of truth for timing/length/format.
        out_path: destination file path.

    Returns:
        The written frame count, which equals ``params.loop_length_samples``.
    """
    import soundfile as sf

    x = np.asarray(mono_16k, dtype=np.float64).reshape(-1)

    sr_out = int(params.sample_rate_out)
    n = int(params.loop_length_samples)

    # --- 1. resample 16k -> sr_out via reduced integer polyphase ratio -------
    g = math.gcd(sr_out, SAMPLE_RATE_SYNTH)
    up = sr_out // g
    down = SAMPLE_RATE_SYNTH // g
    resampled = resample_poly(x, up, down)

    # --- 2. mono -> stereo BEFORE wrap/reverb --------------------------------
    audio = np.column_stack([resampled, resampled])  # shape (frames, 2)
    m = audio.shape[0]

    # --- 3. trim/pad to exactly n, wrapping the overhang onto the head -------
    if m < n:
        pad = np.zeros((n - m, audio.shape[1]), dtype=audio.dtype)
        body = np.concatenate([audio, pad], axis=0)
    else:
        body = audio[:n].copy()
        # Wrap the natural overhang (tail past the loop point) onto the head.
        wrap = min(m - n, int(_MAX_WRAP_S * sr_out), n)
        if wrap > 0:
            tail = audio[n:n + wrap]
            # Equal-power fade of the wrapped tail so it dies into the head.
            fade = np.cos(np.linspace(0, math.pi / 2, wrap)) ** 2
            body[:wrap] += tail * fade[:, None]

    # --- 4. room presence: circular convolution reverb, length-preserving ----
    if REVERB["wet"] > 0:
        body = _apply_reverb(body, sr_out, REVERB)

    # --- 5. loudness normalize ----------------------------------------------
    body = _normalize(body, sr_out)

    # --- 6. write 16-bit PCM stereo (WAV or FLAC) ----------------------------
    fmt = "FLAC" if str(params.format).lower() == "flac" else "WAV"
    sf.write(out_path, body, sr_out, subtype="PCM_16", format=fmt)

    # --- 7. gate -------------------------------------------------------------
    assert body.shape[0] == n, "mastered loop length must equal loop_length_samples"
    return body.shape[0]


def _room_ir(sr: int, seconds: float, seed: int):
    """Procedurally generate a small stereo room impulse response.

    A few early reflections + a decorrelated, low-passed, exponentially-decaying
    diffuse tail per channel. Deterministic given the seed, so renders stay
    byte-reproducible (cacheable).
    """
    length = int(seconds * sr)
    rng = np.random.default_rng(seed)
    ir = np.zeros((length, 2))

    # sparse early reflections (ms -> samples), slightly different per channel
    for ch in range(2):
        taps = [(0.011, 0.6), (0.019, 0.45), (0.027, 0.5), (0.038, 0.32),
                (0.053, 0.28), (0.071, 0.22)]
        for t_s, gain in taps:
            idx = int((t_s + rng.uniform(-0.002, 0.002)) * sr)
            if 0 < idx < length:
                ir[idx, ch] += gain * (1.0 + rng.uniform(-0.15, 0.15))

    # diffuse tail: decorrelated noise * exponential decay, gently low-passed
    t = np.arange(length) / sr
    decay = np.exp(-t / (seconds * 0.33))
    for ch in range(2):
        noise = rng.standard_normal(length) * decay * 0.5
        k = np.hanning(9)
        k /= k.sum()                               # mild high-freq rolloff
        ir[:, ch] += np.convolve(noise, k, mode="same")

    ir[0, :] = 1.0                                 # keep the direct sound (dry) intact
    return ir


def _apply_reverb(audio, sr: int, params: dict):
    """Add convolution reverb via CIRCULAR convolution (length-preserving).

    Circular convolution means the reverb tail from the end of the loop folds
    into its head — exactly right for a seamless loop, and it keeps the output
    length == input length so the exact-loop-length gate still holds.
    """
    n = audio.shape[0]
    ir = _room_ir(sr, params["seconds"], params["seed"])
    wet = float(params["wet"])
    out = audio.copy()
    for ch in range(audio.shape[1]):
        ir_ch = ir[:, ch % ir.shape[1]]
        irp = np.zeros(n)
        mm = min(len(ir_ch), n)
        irp[:mm] = ir_ch[:mm]
        conv = np.fft.irfft(np.fft.rfft(audio[:, ch]) * np.fft.rfft(irp), n=n)
        out[:, ch] = (1.0 - wet) * audio[:, ch] + wet * conv
    return out


def _normalize(audio, sr: int):
    """Normalize to -16 LUFS (pyloudnorm) or peak to -1 dBFS as fallback."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        if math.isfinite(loudness):
            gained = pyln.normalize.loudness(audio, loudness, -16.0)
            peak = float(np.max(np.abs(gained))) or 1.0
            if peak > 0.999:  # guard against clipping after LUFS gain
                gained *= 0.999 / peak
            return gained
    except ModuleNotFoundError:
        pass

    peak = float(np.max(np.abs(audio))) or 1.0
    target = 10 ** (-1.0 / 20.0)  # -1 dBFS
    return audio * (target / peak)
