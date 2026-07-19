# RUN_CYCLE — agent runbook for training & rendering sarangi

**You are a fresh agent with no prior context. This file is everything you need.**
Follow it top to bottom. Repo root: `/Users/omarali/PycharmProjects/sarangi-lehra-gen`.

Your job: take the sarangi training audio the user has dropped in, (re)train the
DDSP timbre model, render the lehra render-matrix with the trained model, and
put the results on the user's Desktop so they can verify by listening.

Run this **the first time data exists** and **every time the user adds more data**.

---

## What this project is (30-second version)

`sarangi-lehra-gen` generates pre-rendered sarangi *lehra* loops (a repeating
melodic cycle for taal practice). The pipeline `parse → contour → synthesize →
post` is already built and tested. Synthesis has two backends:

- `sine` — deterministic placeholder, always works, used before a model exists.
- `ddsp` — the real sarangi timbre, loaded from a checkpoint at
  `model/sarangi_ddsp/`. **Producing that checkpoint is the training cycle.**

DDSP timbre transfer is **self-supervised — no note labels.** Curation is just
two folders: `training/data/raw/solo/` and `training/data/raw/with_tabla/`
(see `training/DATA_SPEC.md`).

---

## The one command that does the local half

```bash
cd /Users/omarali/PycharmProjects/sarangi-lehra-gen
python scripts/run_training_cycle.py
```

This script:
1. Runs `training/preprocess.py` on the raw audio (demucs strip on `with_tabla`,
   silence trim, 16 kHz mono → `training/data/processed/`, optional TFRecord).
2. Looks for a trained checkpoint at `model/sarangi_ddsp/`.
   - **Found** → renders the matrix with the **ddsp** backend (real sarangi).
   - **Not found** → renders **sine previews** and prints the training steps below.
3. Renders the render-matrix (Sa × bpm × seed) into `out/`.
4. Copies every render to **`~/Desktop/sarangi-renders/`** with readable names and
   a `MANIFEST.txt`.

So the local, deterministic work is fully automated. The **GPU training itself
runs on Colab** (this machine has no GPU/TensorFlow) — that's the manual step in
the middle, described next.

---

## Full cycle, step by step

### 1. Preprocess (local, automated)
```bash
python scripts/run_training_cycle.py --skip-... # (just run it; step 1 happens first)
# or run preprocess alone:
python training/preprocess.py --tfrecord
```
If deps are missing: `pip install -r training/requirements-train.txt` (heavy —
demucs, tensorflow, ddsp, crepe; Colab/dev only, never shipped).
Output: cleaned 16 kHz mono clips in `training/data/processed/`, and if
`--tfrecord`, a TFRecord in `training/data/tfrecord/`.

### 2. Train on Colab (manual, ~2–4 hr on free GPU)
Open `training/train_ddsp.ipynb` in Google Colab (GPU runtime).
- Upload/mount `training/data/processed/` (or the TFRecord).
- Run the cells: it pins DDSP deps (note: Magenta's DDSP was archived in 2024 —
  the notebook documents the version pin / fallback), trains the autoencoder,
  and exports the checkpoint.
- **Download the exported checkpoint and place it at
  `/Users/omarali/PycharmProjects/sarangi-lehra-gen/model/sarangi_ddsp/`**
  (the folder should contain the TF checkpoint files + `operative_config-*.gin`).

The runtime `ddsp` backend (`sarangi_gen/backends/ddsp.py`) loads exactly this
folder. Its inference flow is sketched there; if it raises "checkpoint not found"
the export didn't land in that folder.

### 3. Render + verify (local, automated)
```bash
python scripts/run_training_cycle.py            # now auto-detects the checkpoint → ddsp
```
Renders the matrix with the trained model and copies to
`~/Desktop/sarangi-renders/`. Open that folder and listen.

### 4. Adding MORE data later
Drop new clips into `training/data/raw/{solo,with_tabla}/`, then repeat from
step 1. Re-training refines the checkpoint; re-rendering refreshes the Desktop
folder. Renders are named by cache key, so unchanged tuples are idempotent.

---

## Verifying a render is good

- **Format gate:** `python scripts/assert_loop_length.py --bpm 160 --avartans 4 --sr 44100 out/<key>.wav`
  → must PASS (frames exact, stereo, PCM_16). Loop-length is the primary
  correctness gate.
- **Seamless loop:** loop a file ~20× and listen at the sam (cycle boundary) —
  there must be no click/pop. The equal-power seam wrap in `post.py` guarantees
  this for a correct render.
- **Timbre:** it should sound like a bowed sarangi with audible meend (slides
  between notes), not a pure tone (that would mean it fell back to `sine`).
- **Full test suite still green:** `python -m pytest -q` → all pass.

---

## Useful flags on the driver

```bash
python scripts/run_training_cycle.py --backend sine     # force previews
python scripts/run_training_cycle.py --backend ddsp     # force real (errors if no ckpt)
python scripts/run_training_cycle.py --skip-preprocess  # data unchanged, just re-render
python scripts/run_training_cycle.py --sa C,C# --bpm 80,160 --seed 0,3 --avartans 4
python scripts/run_training_cycle.py --no-copy          # don't touch the Desktop folder
```

Default render matrix (plan §7): **Sa {C, C#} × bpm {80, 160} × seed {0, 3}** = 8 loops.

---

## If model quality is poor (same-day fallback)

Pick the best tempo-consistent solo passage from your source audio and run it
through `post.py` (exact length + seam + cache-key name) so a usable loop ships
while you keep improving the model. See `sarangi-lehra-gen-plan.md` §9.
