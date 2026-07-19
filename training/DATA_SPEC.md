# Sarangi Training Data — Sourcing Spec

**Read this first. Start collecting audio the moment you finish it.**

You are gathering raw audio to teach a **DDSP timbre-transfer** model what a sarangi
*sounds like*. The model does **not** learn melody, rhythm, or taal — that all comes
from the runtime contour generator. It only learns **timbre**: bow noise, the reedy
buzz of sympathetic strings, how the tone changes across registers and bow speeds.

Because of that, curation is deliberately minimal. **There is NO note labelling, NO
transcription, NO alignment.** You sort clips into exactly **two buckets** and nothing
more. The preprocessing script does the rest.

---

## Target (the finish line)

- **~10–20 minutes of usable audio total**, measured *after* trimming silence.
- Cover **all registers** (low/mandra, middle/madhya, high/taar) and **all bowing
  speeds** (slow sustained meend → fast gat/jhala articulation).
- Source quality: prefer the **most lossless source you can get** (WAV/FLAC via
  `yt-dlp`). CREPE pitch tracking and demucs separation both degrade on lossy,
  reverb-drenched, or multi-instrument audio.
- 10 minutes of *clean, varied* sarangi beats 30 minutes of muddy, repetitive audio.
  Quality over quantity.

---

## The two buckets

Create clips and drop each file into exactly one of these directories:

```
training/data/raw/
├── solo/         ← ~70–80% of total. GOLD DATA.
└── with_tabla/   ← ~20–30% of total. Articulation supplement.
```

### `raw/solo/` — clean, unaccompanied sarangi (the core)

This is the **most valuable** data and should be the majority (~70–80%).

**What goes here:**
- **Alap / alaap** sections (the free-rhythm opening of a raga performance) — before
  the tabla enters. This is where sarangi is most exposed and sustained.
- Slow, sustained bowing; **meend-rich** passages (the long expressive glides between
  notes); jod/vistaar.
- Solo practice recordings, riyaaz, unaccompanied demonstrations.

**Tanpura bleed is ACCEPTABLE and expected.** The app plays a tanpura drone at runtime
anyway, so a soft tanpura behind the sarangi does not hurt the model. Prefer
**quiet / close-miked** recordings where the sarangi dominates. If tanpura is loud,
the optional noise-reduction step (see preprocess) can help.

### `raw/with_tabla/` — sarangi + tabla (the articulation supplement)

Keep this a **minority** (~20–30%). Its only job is to teach **fast-bowing
articulation** that slow alap underrepresents.

**What goes here:**
- **Fast gat / jhala** sections where the sarangi plays quick, articulate phrases,
  usually accompanied by tabla.

**These files will be automatically run through `demucs`** during preprocessing to strip
the tabla percussion, keeping the melodic ("other") stem. Separation is imperfect:
**smeared, watery, or artifact-heavy results get discarded** — that is why this bucket
stays a minority. Do not put anything here that you could instead find as clean solo.

---

## What to AVOID (poisons the model)

- **Raw, un-separated tabla in `solo/`.** Percussion transients wreck a timbre model.
  Tabla-accompanied audio goes in `with_tabla/` ONLY (so it gets separated) — never in
  `solo/`.
- **Heavy reverb / hall ambience.** Blurs the timbre and confuses CREPE f0 tracking.
- **Multiple simultaneous sarangis** (ensemble / jugalbandi with two bowed instruments).
  The model needs one clear voice.
- **Harmonium-dominant or vocal-dominant sections.** In many recordings a sarangi
  *accompanies* a singer — those sections are mostly voice. Skip them; you want the
  sarangi in front.
- **DJ / remix / "lofi" / fusion uploads.** Added beats, sidechain, effects, and
  pitch-shifting are all poison.
- **Clipping / distortion / phone-recorded** low-fi uploads.

When in doubt: if *you* cannot clearly hear the sarangi as the main voice, the model
can't learn from it either.

---

## Sourcing: `yt-dlp` (lossless-preferred)

Prefer `yt-dlp` extracting to **WAV or FLAC** over any "youtube-to-mp3" website — you
get a lossless (or best-available) source, which measurably helps CREPE and demucs.

**Install:**
```bash
pip install yt-dlp
# ffmpeg is required for extraction:  macOS: brew install ffmpeg
```

**Recommended command (WAV, best audio):**
```bash
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "%(title)s.%(ext)s" "<YOUTUBE_URL>"
```

**FLAC (smaller, still lossless):**
```bash
yt-dlp -x --audio-format flac --audio-quality 0 \
  -o "%(title)s.%(ext)s" "<YOUTUBE_URL>"
```

**MP3 is acceptable** if it's easier or the only option — DDSP will still train. Just
prefer lossless when you have the choice:
```bash
yt-dlp -x --audio-format mp3 --audio-quality 0 -o "%(title)s.%(ext)s" "<URL>"
```

**Trimming to just the section you want** (avoids downloading full hour-long concerts).
This needs `ffmpeg` present; downloads only `START`→`END`:
```bash
# grab 2:30 to 7:00 of a video (the alap), as WAV
yt-dlp -x --audio-format wav \
  --download-sections "*2:30-7:00" \
  -o "ramnarayan_darbari_alap.%(ext)s" "<URL>"
```

You do **not** have to trim perfectly here — `preprocess.py` trims leading/trailing
silence and drops sarangi-silent gaps automatically. Rough section cuts are enough.

---

## Naming convention

Free-form, but a consistent, descriptive name helps you keep track. Suggested:

```
artist_raga_section.wav
```

Examples:
```
solo/ramnarayan_darbari_alap.wav
solo/sultankhan_bhairavi_alap.flac
solo/dhrubaghosh_marwa_vistaar.wav
with_tabla/kamalsabri_desh_gat.wav
with_tabla/arunanarayan_jog_jhala.flac
```

**No transcription, note names, timestamps, or label files are needed.** The directory
(`solo/` vs `with_tabla/`) is the only metadata the pipeline uses.

---

## Where to look — candidate artists (search hints)

These are **search starting points**, not asserted URLs. Search YouTube for the artist
plus terms like `sarangi alap`, `sarangi solo`, `raag ... sarangi`, `riyaaz`.

**Sarangi masters (solo / alap-rich — best for `solo/`):**
- **Ram Narayan** — the canonical solo-sarangi maestro; huge amount of unaccompanied alap.
- **Sultan Khan**
- **Dhruba Ghosh**
- **Aruna Narayan** (daughter of Ram Narayan; solo recitals)
- **Kamal Sabri** (also fast/fusion — use his classical solo sections)
- **Sabir Khan** (sarangi)
- **Ustad Nathu Khan**, **Ustad Ghulam Sabir**, **Pandit Ramesh Mishra**
- **Murad Ali Khan**, **Sarwar Hussain Khan**

**For `with_tabla/` (fast gat/jhala articulation):**
- Any of the above in a **gat** section with tabla accompaniment.
- Nishat Khan-era / classical concert accompaniment sarangi playing fast passages.

Search terms that surface good material:
`sarangi alap`, `sarangi solo full`, `raag <name> sarangi`, `sarangi riyaaz`,
`unaccompanied sarangi`, `sarangi meend`.

---

## Quick checklist before you hand off to `preprocess.py`

- [ ] ~10–20 min total after your rough trims.
- [ ] Majority in `solo/`, a minority in `with_tabla/`.
- [ ] Registers covered: some low, plenty of middle, some high.
- [ ] Bowing speeds covered: mostly slow/sustained, some fast articulation.
- [ ] No un-separated tabla in `solo/`. No remixes, no heavy reverb, no vocal-lead cuts.
- [ ] Sourced as WAV/FLAC where possible.

Then run:
```bash
python training/preprocess.py --help
```
and follow `training/README.md`.
