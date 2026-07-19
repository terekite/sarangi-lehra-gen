#!/usr/bin/env python3
"""Loop-length + format gate (verification §10.2).

Asserts, for each WAV/FLAC file, the invariants NiceNagma's LocalRenderer
enforces on a shipped loop:

  * frames    == round(avartans * 16 * (60/bpm) * samplerate)
  * channels  == 2
  * subtype   == PCM_16
  * samplerate in {44100, 48000}

The expected frame count cannot be recovered from the audio file alone, so the
render parameters are supplied on the command line:

    python scripts/assert_loop_length.py --bpm 160 --avartans 4 --sr 44100 FILE...

(`--matra-count` defaults to 16 for teentaal.) Exits non-zero if any file fails;
prints a clear per-file PASS/FAIL line.
"""

from __future__ import annotations

import argparse
import sys

import soundfile as sf

VALID_SAMPLERATES = {44100, 48000}


def expected_frames(bpm: float, avartans: int, sr: int, matra_count: int) -> int:
    """The exact frame count = round(avartans * matra_count * (60/bpm) * sr)."""
    matra_dur_s = 60.0 / bpm
    loop_length_s = avartans * matra_count * matra_dur_s
    return round(loop_length_s * sr)


def check_file(path: str, bpm: float, avartans: int, sr: int,
               matra_count: int) -> tuple[bool, str]:
    try:
        info = sf.info(path)
    except Exception as exc:  # noqa: BLE001 - report any read failure as FAIL
        return False, f"could not read file ({exc})"

    want = expected_frames(bpm, avartans, sr, matra_count)
    problems: list[str] = []

    if info.frames != want:
        problems.append(f"frames {info.frames} != expected {want}")
    if info.channels != 2:
        problems.append(f"channels {info.channels} != 2")
    if info.subtype != "PCM_16":
        problems.append(f"subtype {info.subtype} != PCM_16")
    if info.samplerate not in VALID_SAMPLERATES:
        problems.append(f"samplerate {info.samplerate} not in {sorted(VALID_SAMPLERATES)}")
    if info.samplerate != sr:
        problems.append(f"samplerate {info.samplerate} != --sr {sr}")

    if problems:
        return False, "; ".join(problems)
    return True, (
        f"frames={info.frames} sr={info.samplerate} "
        f"ch={info.channels} subtype={info.subtype}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assert loop length + WAV/FLAC format invariants."
    )
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--avartans", type=int, required=True)
    parser.add_argument("--sr", type=int, required=True,
                        help="Output sample rate (44100 or 48000).")
    parser.add_argument("--matra-count", type=int, default=16,
                        help="Matras per avartan (default: 16 for teentaal).")
    parser.add_argument("files", nargs="+", help="WAV/FLAC files to check.")
    args = parser.parse_args(argv)

    all_ok = True
    for path in args.files:
        ok, detail = check_file(path, args.bpm, args.avartans, args.sr,
                                args.matra_count)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {path}: {detail}")
        all_ok = all_ok and ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
