"""Timed note events -> per-sample f0 + loudness contours at 16 kHz.

This is the sarangi-specific stage that has no NiceNagma equivalent. NiceNagma
goes events -> MIDI -> FluidSynth; here we go events -> (f0[], loudness[]) ->
DDSP-or-sine synthesis. The contour is what gives the sarangi its characteristic
*continuous bowed* voice: pitch slides between notes (meend), a subtle periodic
gamak wobble, and a soft bowed attack/release instead of a percussive strike.

Everything is derived deterministically from `(events, params)` — the same input
always produces bit-identical arrays (verification §6, "primary correctness
gate"). All timing comes from `RenderParams` (single source of truth); event
times are converted to sample indices as `round(seconds * 16000)` so the taal
grid never accumulates rounding drift.

Design summary (see `render_contour` docstring for the knobs):
  * meend   — raised-cosine f0 glide carved into the *tail* of the previous note
              so it lands exactly on the next onset; width scales with the
              interval size and the laya (faster tempo => shorter glide).
  * gamak   — a small, bounded periodic pitch+amplitude oscillation whose depth
              ramps up from zero at each onset (so onsets stay pitch-exact).
  * bowing  — per-note raised-cosine attack/release; legato notes hand their
              amplitude to the next note's attack so the bow never clicks.
  * variation — per-note expressive jitter seeded by (seed, avartan, matra); it
              only touches expressive params, never a structural onset's timing
              or its landing pitch.
  * overhang — the final OVERHANG_S past the loop body continues the last note's
              pitch and decays its loudness, giving `post` a tail to seam-fold.
"""

from __future__ import annotations

import numpy as np

from .model import (
    OVERHANG_S,
    SAMPLE_RATE_SYNTH,
    Contour,
    Event,
    RenderParams,
)

# Reference matra duration (bpm 160) used to scale expressive times by laya.
_REF_MATRA_S = 0.375


def _midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def render_contour(events: list[Event], params: RenderParams) -> Contour:
    """Render `events` into a `Contour` of length `params.n_synth_overhang`.

    Returns float32 `f0_hz` and `loudness` arrays at 16 kHz. Where a note sounds
    `f0_hz` is its pitch (with meend glide + gamak wobble); rests/gaps are 0.0
    (unvoiced). `loudness` is a bowed envelope in [0, 1] roughly tracking
    velocity/swell. The trailing OVERHANG_S region continues the final note's
    tail so the post stage can fold a seamless loop seam.
    """
    sr = SAMPLE_RATE_SYNTH
    n = params.n_synth_overhang
    n_body = params.n_synth

    f0_base = np.zeros(n, dtype=np.float64)   # pure pitch + glide, pre-gamak
    gp_mod = np.ones(n, dtype=np.float64)     # gamak pitch multiplier
    ga_mod = np.ones(n, dtype=np.float64)     # gamak amplitude multiplier
    loud = np.zeros(n, dtype=np.float64)

    if not events:
        return Contour(
            f0_hz=f0_base.astype(np.float32),
            loudness=loud.astype(np.float32),
        )

    evs = sorted(events, key=lambda e: e.start_s)

    matra_dur = params.matra_dur_s
    laya_scale = _clamp(matra_dur / _REF_MATRA_S, 0.6, 1.4)
    # legato/contiguity tolerance: notes within ~5 ms are "touching".
    tol = int(round(0.005 * sr))

    # ---- per-event precompute (samples, pitch, expressive params) ---------- #
    info: list[dict] = []
    for e in evs:
        s0 = int(round(e.start_s * sr))
        s1 = int(round((e.start_s + e.dur_s) * sr))
        # Per-note expressive RNG. Non-structural *and* structural notes may get
        # amplitude/timbre variation, but timing (s0/s1) and the onset landing
        # pitch are never touched, so the taal grid stays exact.
        rng = np.random.default_rng([int(params.seed), int(e.avartan), int(e.matra)])
        amp_jitter = 1.0 + float(rng.uniform(-0.06, 0.06))
        amp = _clamp(0.75 * (e.velocity / 127.0) * amp_jitter, 0.0, 1.0)
        gdepth_scale = float(rng.uniform(0.7, 1.3))
        gphase = float(rng.uniform(0.0, 2.0 * np.pi))
        gfreq = float(rng.uniform(5.0, 6.5))
        attack_scale = float(rng.uniform(0.8, 1.2))
        info.append(
            dict(
                e=e,
                s0=s0,
                s1=s1,
                hz=_midi_to_hz(e.midi),
                amp=amp,
                gphase=gphase,
                gfreq=gfreq,
                gdepth_p=0.008 * gdepth_scale,   # ~14 cents peak pitch wobble
                gdepth_a=0.04 * gdepth_scale,    # ~4% amplitude wobble
                gramp=max(1, int(round(0.06 * laya_scale * sr))),
                attack_scale=attack_scale,
            )
        )

    # ---- 1. base pitch fill (sustain regions); rests stay 0 --------------- #
    for it in info:
        a0, a1 = max(0, it["s0"]), min(n, it["s1"])
        if a1 > a0:
            f0_base[a0:a1] = it["hz"]

    # ---- 2. auto-meend: glide into each new distinct pitch ----------------- #
    # The glide occupies [onset - g, onset) — carved from the *previous* note's
    # tail — so f0 arrives at the new pitch exactly on the onset sample (keeps
    # structural onsets pitch-exact) while still passing through intermediate
    # values (the audible slide).
    for i in range(1, len(info)):
        cur, prev = info[i], info[i - 1]
        if cur["hz"] == prev["hz"]:
            continue  # same-pitch sustain: no glide
        if cur["s0"] - prev["s1"] > tol:
            continue  # a rest separates them: fresh bowed onset, no slide
        interval = abs(cur["e"].midi - prev["e"].midi)
        glide_s = _clamp((0.020 + 0.010 * interval) * laya_scale, 0.015, 0.080)
        g = int(round(glide_s * sr))
        g = min(g, max(0, cur["s0"] - max(prev["s0"], 0)))  # fit inside prev note
        if g < 2:
            continue
        start = cur["s0"] - g
        frac = np.arange(g) / g
        w = 0.5 * (1.0 - np.cos(np.pi * frac))       # raised cosine 0 -> 1
        vals = prev["hz"] + (cur["hz"] - prev["hz"]) * w
        a0, a1 = max(0, start), min(n, cur["s0"])
        if a1 > a0:
            off = a0 - start
            f0_base[a0:a1] = vals[off:off + (a1 - a0)]

    # ---- 3. gamak: bounded periodic wobble, zero at each onset ------------- #
    for it in info:
        a0, a1 = max(0, it["s0"]), min(n, it["s1"])
        if a1 <= a0:
            continue
        idx = np.arange(a0, a1)
        tloc = (idx - it["s0"]) / sr
        env = np.clip((idx - it["s0"]) / it["gramp"], 0.0, 1.0)  # 0 at onset -> 1
        osc = np.sin(2.0 * np.pi * it["gfreq"] * tloc + it["gphase"])
        gp_mod[a0:a1] = 1.0 + env * it["gdepth_p"] * osc
        ga_mod[a0:a1] = 1.0 + env * it["gdepth_a"] * osc

    # ---- 4. bowing envelope on loudness ------------------------------------ #
    for i, it in enumerate(info):
        a0, a1 = it["s0"], it["s1"]
        L = a1 - a0
        if L <= 0:
            continue
        legato_in = i > 0 and (it["s0"] - info[i - 1]["s1"]) <= tol
        reaches_end = it["s1"] >= n_body - tol
        legato_out = (
            i < len(info) - 1 and (info[i + 1]["s0"] - it["s1"]) <= tol
        ) or reaches_end
        start_level = info[i - 1]["amp"] if legato_in else 0.0
        amp = it["amp"]

        a = min(int(round(_clamp(0.030 * laya_scale * it["attack_scale"], 0.010, 0.060) * sr)), L)
        r = 0
        if not legato_out and L > a:
            r = min(int(round(_clamp(0.080 * laya_scale, 0.020, 0.150) * sr)), L - a)

        u = np.arange(L) / max(L - 1, 1)
        swell = _clamp(it["e"].swell, 0.0, 1.0)
        body = amp * (1.0 + 0.2 * swell * np.sin(np.pi * u))  # gentle intra-note arch
        env = body.copy()
        if a > 0:
            k = np.arange(a)
            w = 0.5 * (1.0 - np.cos(np.pi * k / a))           # bowed onset 0 -> 1
            env[:a] = start_level + (body[:a] - start_level) * w
        if r > 0:
            k = np.arange(r)
            w = 0.5 * (1.0 + np.cos(np.pi * k / r))           # release 1 -> 0
            env[L - r:] = body[L - r:] * w

        b0, b1 = max(0, a0), min(n, a1)
        if b1 > b0:
            off = b0 - a0
            loud[b0:b1] = env[off:off + (b1 - b0)]

    loud *= ga_mod  # fold gamak amplitude wobble into the bowed envelope (body)

    # ---- 5. overhang: continue the final note's tail past the loop body ---- #
    if n_body < n:
        tail = max(info, key=lambda it: it["s1"])
        if tail["s1"] >= n_body - tol and tail["hz"] > 0.0:
            idx = np.arange(n_body, n)
            tloc = (idx - tail["s0"]) / sr
            osc = np.sin(2.0 * np.pi * tail["gfreq"] * tloc + tail["gphase"])
            f0_base[n_body:n] = tail["hz"]
            gp_mod[n_body:n] = 1.0 + tail["gdepth_p"] * osc  # env fully open in tail
            m = n - n_body
            u = np.arange(m) / max(m - 1, 1)
            decay = 0.5 * (1.0 + np.cos(np.pi * u))          # 1 -> 0 release
            loud[n_body:n] = tail["amp"] * decay * (1.0 + tail["gdepth_a"] * osc)

    # ---- 6. combine + finalize -------------------------------------------- #
    f0 = f0_base * gp_mod                 # rests stay 0 (0 * mod == 0)
    np.clip(loud, 0.0, 1.0, out=loud)
    np.clip(f0, 0.0, None, out=f0)        # f0 >= 0 guarantee

    return Contour(
        f0_hz=f0.astype(np.float32),
        loudness=loud.astype(np.float32),
    )
