# Sarangi Instrument Implementation Plan — v4 (port-aligned)

**Project:** Lehra practice app — second instrument (sarangi), built in a **standalone git repo** and later ported into the NiceNagma monorepo.
**Constraints:** client-side only (no server resources at runtime), one-day build, $0 budget, YouTube training data, personal/friends use.

## Context — why this revision exists

v3 was a good standalone plan, but its I/O contract quietly diverged from the NiceNagma repo it must eventually merge into. Verified against the actual NiceNagma source (`contracts/`, `core/src/nagma_core/`, `render/`, `app/lib/`), three gaps would have made the port painful:

1. **Sargam notation conflicts.** v3 used low-octave `,`, high-octave `'`, and a `~` meend token, "space-separated, one beat per token." NiceNagma's real grammar uses low-octave `.` (mandra), high-octave `'` (taar), bare middle octave — and **`,` is matra-subdivision, not low-octave.** There is **no `~`/meend token at all.** A v3 lehra file would misparse in NiceNagma.
2. **Output format underspecified.** v3 said "WAV, loop cut at sam." NiceNagma requires **stereo, 16-bit LE PCM, 44.1 kHz, trimmed to an exact integer `loop_length_samples`** with a specific equal-power seam wrap. This exactness is NiceNagma's "primary correctness gate."
3. **No cache-key alignment.** NiceNagma addresses every loop by a 24-char SHA-256 cache key over a canonical `RenderRequest`. Without matching it, the app can't auto-find a shipped sarangi loop.

**Decisions locked with the user (2026-07-19):**
- **Reuse `proposed-teentaal`** — copy NiceNagma's `assets/nagmas/proposed-teentaal.nagma` **byte-identical** into the sarangi repo. Identical `nagma_text` ⇒ identical cache-key identity across instruments.
- **Everything client-side, no server.** The DDSP synthesis runs **offline** (Colab / dev machine) to produce loop files that **ship in the app as a pre-rendered pack**; at runtime the app only looks them up and plays them. **On-device DDSP inference is Tier-2 (deferred)** — the model can't run on-device in a one-day build.

This plan updates the *contract* (notation, CLI, output format, naming) to match NiceNagma exactly. **The ML approach and training-data strategy are unchanged.** "Don't change the implementation, make the plan match the repo."

---

## 1. Approach (unchanged)

Trained ML model: **DDSP timbre synthesis**. A small neural net learns sarangi timbre from real recordings; feed it a pitch + loudness contour generated from the notes and it synthesizes audio that follows the taal grid exactly. Free Colab GPU, ~2–4 hr training, small model. AI-music APIs and sample-per-note approaches remain rejected.

Because the DDSP model is too heavy to run on-device in v1, synthesis is an **offline batch step**. Its outputs are the shipped product. This keeps the runtime **100% client-side with no server** — consistent with NiceNagma's harmonium, which already renders fully on-device (`OfflineRenderer` in `AppDelegate.swift`).

---

## 2. Training Data — realistic sourcing (unchanged from v3)

**Reality check:** solo sarangi almost always includes tabla; truly unaccompanied material is mostly alaap. Fine — **this model class learns timbre, not rhythm.** Rhythm comes entirely from the contour generator. Slow alaap is excellent timbre data; what it underrepresents is fast-bowing *articulation*.

**Dataset recipe:**
1. **Core (~70–80%): alaap / unaccompanied sections**, cut before the tabla enters. Clean, no separation, teaches sustained tone, meend, bow noise. Tanpura bleed acceptable.
2. **Articulation supplement (~20–30%): fast gat/jhala with tabla, run through `demucs`** to strip percussion. Separation artifacts keep this a minority share. Discard smeared outputs — quality over coverage.
3. Never train on raw un-separated tabla — percussion transients poison the model.

**Tanpura bleed acceptable** (the app plays a tanpura anyway). Keep minimal: prefer quiet/close-miked recordings; spectral noise reduction using a tanpura-only section as the noise profile; cut sarangi-silent segments.

Target: 10–20 min after trimming; **16 kHz mono for training** (the model's internal rate — note the final *output* is resampled to 44.1 kHz, see §6); cover all registers and bowing speeds.

---

## 3. Standalone repo: `sarangi-lehra-gen`

Self-contained. **Contract: NiceNagma-grammar sargam text in → NiceNagma-compatible loop file out.** No app dependencies; the port is a drop-in, not a rewrite.

```
sarangi-lehra-gen/
├── README.md
├── requirements.txt
├── lehras/
│   └── proposed-teentaal.nagma   # BYTE-IDENTICAL copy of NiceNagma's file
├── sarangi_gen/
│   ├── parse.py         # sargam text → note events on taal grid (NiceNagma grammar, see §4)
│   ├── taal.py          # teentaal = 4 vibhags × 4 matras (see §4)
│   ├── contour.py       # note events → f0[] + loudness[] (auto-meend, gamak, bowing, per-avartan seed)
│   ├── synthesize.py     # contours → 16 kHz audio via trained DDSP checkpoint
│   ├── cache_key.py     # port of NiceNagma's cache-key fn (see §7) — names outputs
│   └── post.py          # resample→44.1k, stereo, reverb, normalize, EXACT loop trim + seam wrap, encode
├── model/
│   └── sarangi_ddsp/    # trained checkpoint from Colab (committed or LFS)
├── generate.py          # CLI entry point
└── out/                 # rendered loops (gitignored)
```

---

## 4. Sargam input format — **must match NiceNagma exactly**

This is the biggest v3→v4 correction. Verified against `core/src/nagma_core/sargam.py`, `parser.py`, `taal.py` and the Dart port in `app/lib/render/`. `parse.py` **must reimplement this grammar, not invent one.** The single fidelity guard: the copied `proposed-teentaal.nagma` must parse identically here and in NiceNagma.

**Swars (12, case-sensitive; lowercase = komal, uppercase = shuddha/tivra). Semitone offset above Sa:**
`S`=0 `r`=1 `R`=2 `g`=3 `G`=4 `m`=5 `M`=6 `P`=7 `d`=8 `D`=9 `n`=10 `N`=11
(`m` = shuddha Ma, `M` = tivra Ma. Sa and Pa have no komal/tivra variant.)

**Octave markers (suffix, single trailing char):**
- middle (madhya): **no marker** — `S`  (offset 0)
- upper (taar): trailing **`'`** — `S'`  (+12)
- lower (mandra): trailing **`.`** — `S.`  (−12)
  (v3's low-octave `,` was wrong — `,` is subdivision below.)

**Other tokens:**
- **`-`** = sustain (previous note continues through this slot). A nagma may **not** begin with `-`.
- **`,`** = matra subdivision: splits one matra cell into **up to 4** equal slots (chaugun), e.g. `S,r,g,m`. (NiceNagma's prose docs say "max 2" but the *code* enforces max 4 — follow the code.) Slots may themselves be sustains (`S,-,g,m`).
- Vibhag separator = **newline OR `|`** (equivalent; both normalized).
- Within a vibhag, matras are **whitespace-separated**.
- Lines starting with `#` are comments; blank lines ignored.
- **No meend/glide/slur/tie token exists.** Meend is *not* an input symbol.

**Meend belongs in `contour.py`, not the grammar.** To stay port-faithful, treat meend/gamak the way NiceNagma treats legato — a **rendering behavior**, auto-inferred by the contour generator (glide the f0 between consecutive distinct pitches; parameterize glide time/curve by interval and laya), **not** a text token. This keeps the input grammar byte-compatible while still giving sarangi its characteristic slides.

**Taal structure (`teentaal`, the only taal):** 4 vibhags × 4 matras = **16 matras**. Claps: sam=matra 0, taali=4, khaali=8, taali=12 (0-based). `parse.py` must validate exact vibhag count (4) and matras-per-vibhag (4), matching NiceNagma's parser errors.

**Canonical example (`proposed-teentaal.nagma`, copy verbatim):**
```
# User-proposed lehra, Teentaal (16 matras), transcribed from staff notation.
# Sa = C (C4). Half note = 1 matra; two quarter notes = a split matra (',').
# Divisions on matra 4 (B3,C4) and matra 12 (D3,C3). '.' = mandra (lower octave).
#
#  matra:  1   2   3    4
#          C4  C4  C4   B3,C4
S S S N.,S
#          Eb4 D4  B3   C4
g R N. S
#          F#3 G3  Eb3  D3,C3
M. P. g. R.,S.
#          Eb3 F#3 G3   B3
g. M. P. N.
```

---

## 5. CLI — align field names/enums with NiceNagma's `RenderRequest`

`generate.py` flags map 1:1 to `contracts/render-request.schema.json` so a port needs no argument translation:

```
python generate.py \
    --lehra lehras/proposed-teentaal.nagma \
    --taal teentaal \          # enum: teentaal (only value)
    --bpm 160 \                # was --tempo. RenderRequest range 40–240
    --sa C# \                  # enum: C C# D D# E F F# G G# A A# B (sharps)
    --avartans 4 \             # was --seconds. RenderRequest 1–8, default 4
    --seed 3 \                 # RenderRequest default 0
    --instrument sarangi \     # NEW enum value to add on port (see §8)
    --format wav \             # enum: wav | flac
    [--out out/<file>]         # optional; defaults to cache-key name (§7)
```

Changes vs v3: `--tempo`→**`--bpm`**, `--seconds`→**`--avartans`** (integer avartans is what the loop-length gate needs; keep an optional `--seconds` that resolves to `round(seconds / avartan_dur_s)` avartans for convenience). Add `--instrument sarangi`. Sa restricted to the 12 sharp names in the schema enum (the parser also accepts flats `Db/Eb/…`, but ship sharps to match the enum). Sa→MIDI base is **C4 = 60** (`sa_to_midi`).

---

## 6. Output format — **byte-shape compatible with NiceNagma loops**

`post.py` must emit exactly what NiceNagma's renderers produce (verified in `render/master.py` and the Swift `writeWav`):

- **Container/codec:** WAV, **PCM 16-bit, little-endian, stereo (2 ch)**. (FLAC allowed via `--format flac`.) DDSP synthesizes 16 kHz mono → **resample to 44.1 kHz and duplicate/pan to stereo** in `post.py`.
- **Sample rate:** **44100** (NiceNagma default; schema also permits 48000 — default to 44100).
- **Exact loop length (the correctness gate):**
  ```
  matra_dur_s        = 60.0 / bpm
  avartan_dur_s      = 16 * matra_dur_s        # matra_count = 16 for teentaal
  loop_length_s      = avartans * avartan_dur_s
  loop_length_samples = round(loop_length_s * 44100)
  ```
  The final buffer **must be exactly `loop_length_samples` frames** — trim if longer, zero-pad if shorter. Assert it (NiceNagma's `LocalRenderer` throws if `frames != loopLengthSamples`).
- **Seam wrap (bowed/sustained → use the harmonium/lehra preset, NOT the pluck preset):** synthesize a short overhang past `loop_length_samples` (~0.6 s of release/reverb tail), then wrap it onto the head with an **equal-power `cos²` crossfade** (`fade = cos(linspace(0, π/2, wrap))²`), matching `render/master.py`'s `_MAX_WRAP_S = 0.6` path. Sarangi is bowed/continuous, so do **not** use the circular full-overhang fold reserved for plucked/drone instruments (tanpura).
- **Two hard invariants to preserve** (NiceNagma's reason to exist):
  1. **Machine-perfect laya** — matra boundaries and sam land on mathematically exact sample positions (`matra_dur_s` exact); micro-timing jitter only on **non-structural** notes. Verify with a sine-render taal-lock test.
  2. **Per-avartan variation** — render 1–8 avartans with a different derived seed per cycle (base `--seed` + avartan index) so the loop isn't rubber-stamped, mirroring NiceNagma's super-loop.

---

## 7. Output naming = NiceNagma cache key (enables true drop-in)

For the app to **auto-find a shipped sarangi loop with zero wiring**, name each output file by NiceNagma's own cache key. Port the algorithm exactly (`render/service/app.py::_cache_key` ⇔ `app/lib/services/loop_renderer.dart::cacheKey`):

- **SHA-256 of a canonical JSON object, first 24 hex chars.**
- Object keys **alphabetical**, **compact separators** (`,`/`:`), so Python `json.dumps(..., sort_keys=True, separators=(",",":"))` == Dart `jsonEncode`:
  ```json
  {"avartans":4,"bpm":160.0,"instrument":"sarangi","laya":null,
   "nagma":"<proposed-teentaal.nagma text, .strip()ed, byte-identical incl. comments>",
   "sa":"C#","seed":3,"taal":"teentaal"}
  ```
- **`bpm` is a float** on both sides (`160` → `160.0`) — format identically.
- **`laya`** = `null` (auto-select from bpm: ≤85 vilambit, ≤160 madhya, else drut). Include the key even when null.
- **`nagma`** = the raw file text, trimmed of leading/trailing whitespace only (comments retained — they're part of the hash).
- `schema_version` and `format` are **excluded** from the key.

Output filename: `out/<cache_key>.wav`. `cache_key.py` computes it; `--out` overrides for manual inspection.

**Render matrix (v1):** `proposed-teentaal` × your practice `--sa` values (start 1–2) × 2–3 `--bpm` × 2–3 `--seed`. Because the app requests loops by this exact key tuple, **render the same tuples the app will request.** Adding keys/tempos later is a re-run, not new code (transposition = shifting the f0 contour before synthesis, artifact-free).

---

## 8. Porting into NiceNagma (what the merge actually touches)

When the sarangi repo is done, the port is small and mechanical:

1. **Contracts:** add `"sarangi"` to the `instrument` enum in `contracts/render-request.schema.json` **and** `contracts/expressive-score.schema.json`. (Only schema change; call it out per the "contracts frozen" rule.)
2. **Ship the pack:** drop the `out/<cache_key>.wav` files in as a bundled asset dir (or on-demand download), analogous to `assets/soundfonts/` and `SoundfontProvider`'s download seam. Sarangi has **no `.sf2`** — it does not go through `LocalRenderer`'s soundfont path.
3. **New renderer:** add `SarangiPackRenderer implements LoopRenderer` (`app/lib/services/`). Its `getLoop(req)` computes `req.cacheKey()` and returns the bundled/downloaded `<cacheKey>.wav` (or throws "loop not in pack" if the matrix didn't cover it). `healthy()` = pack present. This mirrors the existing `LoopRenderer` seam so nothing downstream changes.
4. **Selection:** extend `_defaultRenderer()` / `Config.renderer` so `instrument == 'sarangi'` routes to `SarangiPackRenderer` while harmonium keeps `LocalRenderer`. (Instrument-picker UI is out of scope for this plan.)
5. **Cache dir reuse:** the pack renderer writes/looks up the same `${appSupport}/render-cache/<cacheKey>.wav` convention, so switching instruments never re-renders a cached loop.

**No server is introduced or contacted** — the HTTP `HttpRenderClient` path stays unused for sarangi. Tier-2 (later): export a TFLite/Core ML DDSP model + port `contour.py` to Dart/Swift for true on-device sarangi of arbitrary user input — at which point `SarangiPackRenderer` is replaced by a real on-device `LoopRenderer`.

---

## 9. One-Day Schedule (revised)

| Hours | Task |
|---|---|
| 0–1 | Collect YouTube audio (`yt-dlp`); trim; noise-reduce tanpura where possible |
| 1–2 | Preprocess (16 kHz mono); launch DDSP timbre-transfer training on free Colab GPU |
| 2–5 | Training runs unattended. In parallel: repo skeleton; **copy `proposed-teentaal.nagma` byte-identical**; write `parse.py` + `taal.py` to NiceNagma grammar (§4); write `contour.py` (auto-meend); **sine-wave taal-lock test** (matra boundaries land on exact samples) |
| 5–6 | Wire checkpoint into `synthesize.py`; first real renders; tune glide/dynamics by ear |
| 6–7 | `post.py`: resample→44.1k stereo 16-bit, reverb, normalize, **exact `loop_length_samples` trim + equal-power seam**; `cache_key.py` naming; batch render the matrix (§7) |
| 7–8 | Verify a rendered loop against NiceNagma's format + tabla clock: exact frame count, seamless loop, cache-key name resolves. (If it were being dropped in today, it slots into `SarangiPackRenderer` unchanged.) |

**Early risk check (first 30 min of hr 1–2):** Magenta's DDSP repos were archived in 2024 — expect dependency pinning in the Colab; community forks / the `ddsp` pip package are the fallback.
**Same-day fallback if model quality is poor by hr 5:** temporarily loop the best tempo-consistent YouTube solo passage (still run it through `post.py` for exact length + seam + cache-key name) so the instrument ships today; keep improving the model after.

---

## 10. Verification (end-to-end, before calling it done)

Run in the standalone repo — these prove port-readiness without touching NiceNagma:
1. **Grammar parity:** `parse.py` on `proposed-teentaal.nagma` yields 16 matras with the exact note/octave/subdivision structure the NiceNagma parser produces (spot-check `N.,S`→[N mandra, S madhya]; `R.,S.`→[R mandra, S mandra]).
2. **Loop-length gate:** for each rendered file, assert `frames == round(avartans*16*(60/bpm)*44100)` and `sample_rate==44100`, `channels==2`, `PCM_16`. (Mirror NiceNagma's `scripts/assert_loop_length.py`.)
3. **Seam:** loop the WAV ~20× and listen for a click at sam; confirm equal-power wrap, no discontinuity.
4. **Taal lock:** render a sine-substituted contour; confirm matra onsets sit on exact sample indices (no drift over 8 avartans).
5. **Cache-key parity:** compute the filename key in `cache_key.py` and independently in a tiny Dart/Python snippet using NiceNagma's exact fn; assert the 24-char strings match for the same `RenderRequest`.
6. **Drop-in dry-run:** place one output in a mock `render-cache/` dir named `<cacheKey>.wav`; confirm a NiceNagma-style `cacheKey()` for the same request resolves to that file.

---

## 11. Open Items
1. Which `--sa` values / `--bpm` / `--seed` tuples for the first batch (defines the render matrix in §7 — and exactly which loops the app can serve).
2. Commit the model checkpoint directly vs Git LFS (checkpoints are usually small enough to commit plainly).
3. Confirm sharps-only `--sa` is acceptable (schema enum) vs also emitting flat-named duplicates.
4. Tier-2 (later): TFLite/Core ML export + Dart/Swift `contour.py` port for on-device arbitrary-input sarangi, replacing the pre-rendered pack.
