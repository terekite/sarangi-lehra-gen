# sarangi-lehra-gen

Offline generator for pre-rendered **sarangi lehra** loops, built to drop into
the NiceNagma app as a bundled loop pack. NiceNagma-grammar sargam text in →
NiceNagma-compatible loop file out (stereo 16-bit PCM, 44.1 kHz, trimmed to an
exact `loop_length_samples`, seamless at sam, named by NiceNagma's cache key).

The synthesis model (DDSP timbre transfer for sarangi) is trained **offline**;
its rendered outputs are the shipped product. At runtime nothing runs
server-side. Until the model is trained, a deterministic **`sine`** backend makes
the whole pipeline runnable and testable.

## Pipeline

```
nagma text ──parse──▶ Events ──contour──▶ f0 + loudness ──synthesize──▶ mono 16k
                                                                            │
   named <cache_key>.wav ◀──post── stereo 44.1k, exact loop, seam wrap ◀────┘
```

| Module | Role |
|---|---|
| `sarangi_gen/model.py` | Shared contract (dataclasses, `RenderParams`, timing) |
| `sarangi_gen/sargam.py`, `taal.py` | Pitch/tonic mapping, Teentaal (ports of NiceNagma) |
| `sarangi_gen/parse.py` | sargam text → `NagmaDoc` → timed `Event`s |
| `sarangi_gen/contour.py` | Events → f0/loudness (auto-meend, bowing, gamak) |
| `sarangi_gen/synthesize.py` | contour → mono 16 kHz audio (`sine` / `ddsp` backends) |
| `sarangi_gen/post.py` | resample, stereo, exact-length trim, seam wrap, normalize, encode |
| `sarangi_gen/cache_key.py` | NiceNagma-identical output naming |
| `generate.py` | CLI entry point |
| `training/` | Offline DDSP data-prep + Colab training scaffolding |

## Quickstart

```bash
pip install -r requirements.txt
pytest
python generate.py --lehra lehras/proposed-teentaal.nagma \
    --bpm 160 --sa C# --avartans 4 --seed 3       # writes out/<cache_key>.wav
```

## Training data

DDSP timbre transfer is **self-supervised — no note-level labeling**. See
`training/DATA_SPEC.md` for exactly where to drop sourced audio (`solo/` vs
`with_tabla/`) and what to avoid.

## Porting into NiceNagma

Loops are named by NiceNagma's cache key, so the port is mechanical: add
`"sarangi"` to the two contract enums, ship `out/*.wav` as a bundled pack, and
add a `SarangiPackRenderer` that returns `<cacheKey>.wav`. See
`sarangi-lehra-gen-plan.md` §8.
