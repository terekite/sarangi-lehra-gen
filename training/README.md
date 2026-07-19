# Sarangi DDSP — Offline Training

This directory holds the **offline** pipeline that trains the sarangi timbre model.
It is completely independent of the runtime code in `sarangi_gen/`. Nothing here ships
in the app — the app only plays pre-rendered loop files. See the project plan
(`../sarangi-lehra-gen-plan.md`, §2 and §9) for the full rationale.

## What this produces

A trained **DDSP** (Differentiable Digital Signal Processing) autoencoder checkpoint that
has learned **sarangi timbre** — and nothing else. It does not know melody, rhythm, or
taal. At runtime, `sarangi_gen/contour.py` generates an f0 + loudness contour on the taal
grid and `synthesize.py` feeds that contour through this checkpoint to synthesize audio.

DDSP is **self-supervised**: it extracts f0 (via CREPE) and loudness from raw audio and
learns to reconstruct the sound. **There is no note-level labelling or transcription** —
the only human curation is sorting clips into two buckets (see `DATA_SPEC.md`).

## End-to-end flow

```
  YouTube audio                training/DATA_SPEC.md  (READ FIRST)
       │  yt-dlp -x --audio-format wav
       ▼
  training/data/raw/{solo, with_tabla}/*.wav
       │  python training/preprocess.py         (demucs → trim → denoise → 16 kHz mono)
       ▼
  training/data/processed/*.wav
       │  ddsp_prepare_tfrecord                 (CREPE f0 + loudness; --tfrecord flag)
       ▼
  training/data/tfrecord/sarangi.tfrecord*
       │  training/train_ddsp.ipynb  (Colab, free GPU, ~2–4 hr: ddsp_run)
       ▼
  model/sarangi_ddsp/                           (exported checkpoint — committed / LFS)
       │
       ▼
  runtime: sarangi_gen/synthesize.py loads the checkpoint (built by another agent)
```

## Files

| File | Purpose |
|------|---------|
| `DATA_SPEC.md` | **Read first.** How to source and bucket YouTube audio. |
| `preprocess.py` | CLI: raw audio → 16 kHz mono processed clips (+ optional TFRecord). |
| `train_ddsp.ipynb` | Colab notebook: pin deps → train → export checkpoint. |
| `requirements-train.txt` | Training/preprocessing deps (Colab/dev only). |
| `data/raw/{solo,with_tabla}/` | You drop sourced audio here. |
| `data/processed/` | `preprocess.py` output. |
| `data/tfrecord/` | DDSP TFRecord dataset. |

## Step-by-step

### 1. Source data
Follow **`DATA_SPEC.md`**. Target ~10–20 min after trimming, ~70–80% clean solo,
~20–30% fast-with-tabla. No labelling needed.

### 2. Preprocess (dev machine or Colab)
```bash
pip install -r training/requirements-train.txt
python training/preprocess.py                 # both buckets, defaults
python training/preprocess.py --denoise --noise-clip refs/tanpura_only.wav
python training/preprocess.py --tfrecord       # also build the TFRecord
python training/preprocess.py --help           # all options
```
`with_tabla/` files auto-run through `demucs`; `solo/` files skip it. Everything is
trimmed, gated, resampled to **16 kHz mono**, and written to `data/processed/`.

### 3. Train (Colab)
Open `train_ddsp.ipynb` in Google Colab (free GPU). It:
- pins the DDSP deps (the Magenta repo was **archived in 2024** — the notebook documents
  the version pin and the community-fork fallback),
- uploads / mounts `data/processed/` (or the prebuilt TFRecord),
- runs `ddsp_run` autoencoder / timbre-transfer training (~2–4 hr on a free GPU),
- exports the checkpoint to `model/sarangi_ddsp/`.

### 4. Wire in the checkpoint
Commit `model/sarangi_ddsp/` (plain commit if small, else Git LFS — see plan §11). The
runtime `ddsp` backend in `sarangi_gen/synthesize.py` (built by a different agent) loads
it from that path.

## Same-day fallback (if model quality is poor)

Per plan §9: if by hour ~5 the DDSP output is not usable, **do not block the ship**.
Take the single best tempo-consistent **solo** passage you sourced and loop it through the
runtime `sarangi_gen/post.py` — that still gives you exact `loop_length_samples`, the
equal-power seam wrap, and the correct cache-key filename, so the instrument ships today.
Keep improving the model afterward and swap the loops in later (it is a re-render, not new
code).

## Notes / gotchas

- **16 kHz mono is the training rate**, not the output rate. The runtime resamples to
  44.1 kHz stereo in `post.py` — that is out of scope here.
- **DDSP + TF version pinning is the top risk.** If Colab's resolver fights you, pin
  TensorFlow *first*, then `ddsp`, then restart the runtime. See the notebook and
  `requirements-train.txt` for the known-good combo and fork fallback.
- Heavy imports (demucs/ddsp/librosa) are lazy in `preprocess.py`, so `--help` works
  before you install anything.
