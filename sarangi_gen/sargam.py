"""Sargam <-> pitch mapping and Sa (tonic) key parsing.

Ported verbatim from NiceNagma `core/src/nagma_core/sargam.py` so a later merge
into the monorepo is a move, not a rewrite. Pure data + functions, zero I/O.
Semitone offsets are relative to Sa (the tonic), so the same NagmaDoc transposes
to any key by changing `sa_midi`.
"""

from __future__ import annotations

# Semitone offset of each swar above Sa. Lowercase = komal (flat), uppercase =
# shuddha/tivra. Covers all twelve chromatic degrees.
SWAR_SEMITONES: dict[str, int] = {
    "S": 0,   # Shadja
    "r": 1,   # komal Re
    "R": 2,   # shuddha Re
    "g": 3,   # komal Ga
    "G": 4,   # shuddha Ga
    "m": 5,   # shuddha Ma
    "M": 6,   # tivra Ma
    "P": 7,   # Pancham
    "d": 8,   # komal Dha
    "D": 9,   # shuddha Dha
    "n": 10,  # komal Ni
    "N": 11,  # shuddha Ni
}

VALID_SWARS = frozenset(SWAR_SEMITONES)

OCTAVE_OFFSET: dict[str, int] = {
    "mandra": -12,
    "madhya": 0,
    "taar": 12,
}

# Text-grammar octave markers appended to a swar token.
OCTAVE_MARKERS: dict[str, str] = {
    "'": "taar",   # S'  -> upper octave
    ".": "mandra",  # S.  -> lower octave
}

# Sa key name -> semitone above C. Sa is placed in MIDI octave 4 (C4 = 60),
# which puts a lehra in a comfortable madhya-saptak range.
_KEY_SEMITONE: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11,
}

# MIDI note of C in the octave Sa lives in (C4 = 60).
_SA_BASE_MIDI = 60


def sa_to_midi(key: str) -> int:
    """Map a Sa key name ('C#', 'D', ...) to its MIDI note number."""
    k = key.strip()
    if k not in _KEY_SEMITONE:
        raise ValueError(
            f"Unknown Sa key {key!r}. Expected one of: "
            + ", ".join(sorted(set(_KEY_SEMITONE)))
        )
    return _SA_BASE_MIDI + _KEY_SEMITONE[k]


def note_to_midi(swar: str, octave: str, sa_midi: int) -> int:
    """Resolve a (swar, octave) pair to an absolute MIDI note for a given Sa."""
    if swar not in SWAR_SEMITONES:
        raise ValueError(f"Unknown swar {swar!r}")
    if octave not in OCTAVE_OFFSET:
        raise ValueError(f"Unknown octave {octave!r}")
    return sa_midi + SWAR_SEMITONES[swar] + OCTAVE_OFFSET[octave]
