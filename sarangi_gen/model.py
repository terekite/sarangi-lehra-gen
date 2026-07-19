"""THE shared contract. Every module in `sarangi_gen` imports from here and
nothing else in the package (data flows between stages through the CLI, not via
sibling imports). Keep the ported dataclasses in lockstep with NiceNagma's
`core/src/nagma_core/models.py` so the eventual monorepo merge is a move.

Timing note: `RenderParams` is the single source of truth for all durations.
Everything derives from `matra_dur_s = 60 / bpm`, computed in seconds and never
accumulated in samples, which keeps the taal grid drift-free (verification §10.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .sargam import sa_to_midi
from .taal import get_taal

if TYPE_CHECKING:  # numpy only needed at runtime by contour/synth/post
    import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SAMPLE_RATE_SYNTH = 16000     # contour + synthesis internal rate (DDSP native)
SAMPLE_RATE_OUT = 44100       # shipped WAV rate (schema also allows 48000)
OVERHANG_S = 0.6              # == master._MAX_WRAP_S; render this much past loop end
SCHEMA_VERSION = "1.0.0"


# --------------------------------------------------------------------------- #
# NagmaDoc  (ported verbatim from nagma_core.models)
# --------------------------------------------------------------------------- #
@dataclass
class Note:
    kind: str                      # 'swar' | 'sustain'
    swar: str | None = None        # required when kind == 'swar'
    octave: str = "madhya"

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "sustain":
            return {"kind": "sustain"}
        return {"kind": "swar", "swar": self.swar, "octave": self.octave}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Note":
        return cls(kind=d["kind"], swar=d.get("swar"), octave=d.get("octave", "madhya"))


@dataclass
class Matra:
    index: int
    notes: list[Note]

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "notes": [n.to_dict() for n in self.notes]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Matra":
        return cls(index=d["index"], notes=[Note.from_dict(n) for n in d["notes"]])


@dataclass
class NagmaDoc:
    taal: str
    matras: list[Matra]
    raag: str = "bhairavi"
    name: str = ""
    source_text: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "taal": self.taal,
            "raag": self.raag,
            "name": self.name,
            "source_text": self.source_text,
            "matras": [m.to_dict() for m in self.matras],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NagmaDoc":
        return cls(
            taal=d["taal"],
            matras=[Matra.from_dict(m) for m in d["matras"]],
            raag=d.get("raag", "bhairavi"),
            name=d.get("name", ""),
            source_text=d.get("source_text", ""),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# --------------------------------------------------------------------------- #
# Event  (ported verbatim from nagma_core.models; matches expressive-score.schema)
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    start_s: float
    dur_s: float
    midi: int
    velocity: int
    matra: int
    avartan: int
    structural: bool     # True == on a matra boundary; no micro-timing jitter
    swell: float = 0.0   # intra-note swell depth (0..1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s, "dur_s": self.dur_s, "midi": self.midi,
            "velocity": self.velocity, "matra": self.matra, "avartan": self.avartan,
            "structural": self.structural, "swell": self.swell,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Event":
        return cls(
            start_s=d["start_s"], dur_s=d["dur_s"], midi=d["midi"],
            velocity=d["velocity"], matra=d["matra"], avartan=d["avartan"],
            structural=d["structural"], swell=d.get("swell", 0.0),
        )


# --------------------------------------------------------------------------- #
# RenderParams  (sarangi-specific; field names/enums match render-request.schema)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RenderParams:
    nagma_text: str                 # RAW file text (un-parsed; drives the cache key)
    bpm: float                      # MUST be float (cache-key formatting) — 40..240
    sa: str                         # sharps enum "C".."B"
    avartans: int = 4               # 1..8
    seed: int = 0
    instrument: str = "sarangi"
    taal: str = "teentaal"
    laya: str | None = None         # hashed as-is (null) even if internally reduced
    format: str = "wav"             # 'wav' | 'flac' (excluded from cache key)
    sample_rate_out: int = SAMPLE_RATE_OUT

    # ---- derived timing (single source of truth) ----
    @property
    def sa_midi(self) -> int:
        return sa_to_midi(self.sa)

    @property
    def matra_count(self) -> int:
        return get_taal(self.taal).matra_count      # 16 for teentaal

    @property
    def matra_dur_s(self) -> float:
        return 60.0 / self.bpm

    @property
    def avartan_dur_s(self) -> float:
        return self.matra_count * self.matra_dur_s

    @property
    def loop_length_s(self) -> float:
        return self.avartans * self.avartan_dur_s

    @property
    def loop_length_samples(self) -> int:
        """THE correctness gate — final frame count at the OUTPUT rate."""
        return round(self.loop_length_s * self.sample_rate_out)

    @property
    def n_synth(self) -> int:
        """Loop body length at the 16 kHz synthesis rate."""
        return round(self.loop_length_s * SAMPLE_RATE_SYNTH)

    @property
    def n_synth_overhang(self) -> int:
        """Body + seam overhang, at 16 kHz — the length contour/synth must produce."""
        return self.n_synth + round(OVERHANG_S * SAMPLE_RATE_SYNTH)


# --------------------------------------------------------------------------- #
# Contour  (sarangi-specific; the parse->synth intermediate)
# --------------------------------------------------------------------------- #
@dataclass
class Contour:
    f0_hz: "np.ndarray"        # float32 [n_synth_overhang], Hz (0/NaN => unvoiced/rest)
    loudness: "np.ndarray"     # float32 [n_synth_overhang], 0..1
    sample_rate: int = SAMPLE_RATE_SYNTH
