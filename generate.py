#!/usr/bin/env python3
"""CLI entry point for the sarangi-lehra-gen pipeline.

Wires the frozen `sarangi_gen` stages into a single command:

    read lehra text -> RenderParams -> parse_nagma -> build_events
        -> render_contour -> synthesize(backend=...) -> master(...)

Flags map 1:1 onto NiceNagma's RenderRequest (see
`contracts/render-request.schema.json`). Output files are named by NiceNagma's
own cache key (`sarangi_gen.cache_key.output_filename`) so the app can auto-find
a shipped loop with zero wiring.

Batch mode (`--batch` / comma-separated `--sa`/`--bpm`/`--seed`) renders the
cross-product of the render matrix (plan §7). A file whose cache-key name already
exists on disk is skipped, so re-running the matrix is cheap and idempotent.

Runnable as `python generate.py --lehra ... --bpm ... --sa ...` and exposes
`main()`.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

# Make `import sarangi_gen` work when run as a script from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sarangi_gen.cache_key import cache_key, output_filename
from sarangi_gen.contour import render_contour
from sarangi_gen.model import RenderParams
from sarangi_gen.parse import build_events, parse_nagma
from sarangi_gen.post import master
from sarangi_gen.synthesize import synthesize

# Sharp-name Sa enum from render-request.schema.json (sharps only).
SA_CHOICES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Default render matrix (plan §7) used by `--batch` with no explicit values.
DEFAULT_MATRIX_SA = ["C", "C#"]
DEFAULT_MATRIX_BPM = [80.0, 160.0]
DEFAULT_MATRIX_SEED = [0, 3]


def _build_params(
    nagma_text: str,
    *,
    bpm: float,
    sa: str,
    taal: str,
    avartans: int,
    seed: int,
    instrument: str,
    fmt: str,
) -> RenderParams:
    return RenderParams(
        nagma_text=nagma_text,
        bpm=float(bpm),
        sa=sa,
        avartans=int(avartans),
        seed=int(seed),
        instrument=instrument,
        taal=taal,
        format=fmt,
    )


def render_one(
    params: RenderParams,
    *,
    backend: str,
    out_path: Path,
    skip_existing: bool = False,
    verbose: bool = True,
) -> int | None:
    """Run the full pipeline for `params`, writing to `out_path`.

    Returns the written frame count, or None if the file already existed and
    `skip_existing` is set. Asserts the written frame count equals
    `params.loop_length_samples` (the primary correctness gate).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and out_path.exists():
        if verbose:
            print(f"skip (exists): {out_path.name}")
        return None

    doc = parse_nagma(params.nagma_text, taal=params.taal)
    events = build_events(doc, params)
    contour = render_contour(events, params)
    mono = synthesize(contour, params, backend=backend)
    frames = master(mono, params, str(out_path))

    expected = params.loop_length_samples
    assert frames == expected, (
        f"loop-length gate failed: wrote {frames} frames, "
        f"expected {expected} (== loop_length_samples)"
    )

    if verbose:
        key = cache_key(params)
        print(f"wrote: {out_path}")
        print(f"  cache_key : {key}")
        print(f"  bpm/sa    : {params.bpm} / {params.sa}")
        print(f"  avartans  : {params.avartans}  seed={params.seed}  backend={backend}")
        print(f"  frames    : {frames}  (== loop_length_samples)")
    return frames


def _resolve_out(args_out: str | None, params: RenderParams) -> Path:
    if args_out:
        return Path(args_out)
    return Path("out") / output_filename(params)


def _parse_list(value: str, caster):
    return [caster(v.strip()) for v in value.split(",") if v.strip()]


def _resolve_avartans_from_seconds(seconds: float, params0: RenderParams) -> int:
    """Convert a convenience --seconds into an integer avartan count."""
    avartan_dur_s = params0.avartan_dur_s
    return max(1, round(seconds / avartan_dur_s))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate.py",
        description="Render a sarangi lehra loop (NiceNagma-compatible output).",
    )
    parser.add_argument("--lehra", required=True, type=Path,
                        help="Path to the .nagma lehra text file (required).")
    parser.add_argument("--taal", default="teentaal",
                        help="Taal name (default: teentaal).")
    parser.add_argument("--bpm", default=None,
                        help="Beats per minute, 40..240 (float; default 160). "
                             "Comma-separated in --batch mode.")
    parser.add_argument("--sa", default=None,
                        help="Sa tonic key, sharps enum C..B (required unless "
                             "--batch). Comma-separated in --batch mode.")
    parser.add_argument("--avartans", type=int, default=4,
                        help="Number of avartans, 1..8 (default: 4).")
    parser.add_argument("--seed", default=None,
                        help="Base seed (default: 0). Comma-separated in --batch.")
    parser.add_argument("--instrument", default="sarangi",
                        help="Instrument name (default: sarangi).")
    parser.add_argument("--format", dest="fmt", choices=["wav", "flac"],
                        default="wav", help="Output container (default: wav).")
    parser.add_argument("--backend", choices=["sine", "ddsp"], default="sine",
                        help="Synthesis backend (default: sine).")
    parser.add_argument("--out", default=None,
                        help="Output path (single render only). "
                             "Default: out/<cache_key>.<ext>.")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Convenience: resolve --avartans from a target "
                             "duration in seconds (overrides --avartans, warns).")
    parser.add_argument("--batch", action="store_true",
                        help="Render the cross-product of comma-separated "
                             "--sa/--bpm/--seed (defaults to the §7 matrix).")

    args = parser.parse_args(argv)

    lehra_path: Path = args.lehra
    if not lehra_path.exists():
        parser.error(f"lehra file not found: {lehra_path}")
    nagma_text = lehra_path.read_text()

    # --- Batch mode: render the matrix cross-product ------------------------ #
    if args.batch:
        sa_vals = _parse_list(args.sa, str) if args.sa else DEFAULT_MATRIX_SA
        bpm_vals = _parse_list(args.bpm, float) if args.bpm else DEFAULT_MATRIX_BPM
        seed_vals = _parse_list(args.seed, int) if args.seed else DEFAULT_MATRIX_SEED
        # Guard the schema Sa enum for each value.
        for sa in sa_vals:
            if sa not in SA_CHOICES:
                parser.error(f"--sa {sa!r} not in schema enum {SA_CHOICES}")

        combos = list(itertools.product(sa_vals, bpm_vals, seed_vals))
        print(f"batch: {len(combos)} render(s) "
              f"[sa={sa_vals} bpm={bpm_vals} seed={seed_vals}]")
        rendered = skipped = 0
        for sa, bpm, seed in combos:
            params = _build_params(
                nagma_text, bpm=bpm, sa=sa, taal=args.taal,
                avartans=args.avartans, seed=seed,
                instrument=args.instrument, fmt=args.fmt,
            )
            out_path = Path("out") / output_filename(params)
            result = render_one(
                params, backend=args.backend, out_path=out_path,
                skip_existing=True, verbose=False,
            )
            if result is None:
                skipped += 1
                print(f"  skip {out_path.name}  (sa={sa} bpm={bpm} seed={seed})")
            else:
                rendered += 1
                print(f"  ok   {out_path.name}  (sa={sa} bpm={bpm} seed={seed})")
        print(f"batch done: {rendered} rendered, {skipped} skipped.")
        return 0

    # --- Single render ------------------------------------------------------ #
    if args.sa is None:
        parser.error("--sa is required (unless --batch).")
    if args.sa not in SA_CHOICES:
        parser.error(f"--sa {args.sa!r} not in schema enum {SA_CHOICES}")

    bpm = _parse_list(args.bpm, float)[0] if args.bpm else 160.0
    seed = _parse_list(args.seed, int)[0] if args.seed else 0
    avartans = args.avartans

    if args.seconds is not None:
        probe = _build_params(
            nagma_text, bpm=bpm, sa=args.sa, taal=args.taal,
            avartans=avartans, seed=seed,
            instrument=args.instrument, fmt=args.fmt,
        )
        avartans = _resolve_avartans_from_seconds(args.seconds, probe)
        print(f"warning: --seconds {args.seconds} resolved to "
              f"--avartans {avartans} (avartan_dur_s={probe.avartan_dur_s:.4f}s)")

    params = _build_params(
        nagma_text, bpm=bpm, sa=args.sa, taal=args.taal,
        avartans=avartans, seed=seed,
        instrument=args.instrument, fmt=args.fmt,
    )
    out_path = _resolve_out(args.out, params)
    render_one(params, backend=args.backend, out_path=out_path, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
