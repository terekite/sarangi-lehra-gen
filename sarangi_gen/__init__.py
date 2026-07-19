"""sarangi_gen — offline generator for pre-rendered sarangi lehra loops.

Public surface is the shared contract in `model.py`. Pipeline modules
(`parse`, `contour`, `synthesize`, `post`, `cache_key`) import from `model`
only; they are wired together by `generate.py`, not by importing each other.
"""

from .model import (
    OVERHANG_S,
    SAMPLE_RATE_OUT,
    SAMPLE_RATE_SYNTH,
    SCHEMA_VERSION,
    Contour,
    Event,
    Matra,
    NagmaDoc,
    Note,
    RenderParams,
)

__all__ = [
    "OVERHANG_S",
    "SAMPLE_RATE_OUT",
    "SAMPLE_RATE_SYNTH",
    "SCHEMA_VERSION",
    "Contour",
    "Event",
    "Matra",
    "NagmaDoc",
    "Note",
    "RenderParams",
]
