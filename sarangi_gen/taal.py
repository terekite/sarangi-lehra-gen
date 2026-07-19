"""Taal definitions.

Ported verbatim from NiceNagma `core/src/nagma_core/taal.py`. Ships Teentaal
only, but the structure is taal-agnostic. A taal is a sequence of vibhags
(measures); each vibhag has a matra count and a clap type. Sam is matra 0.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vibhag:
    """One section of a taal cycle."""
    length: int          # number of matras in this vibhag
    clap: str            # 'taali', 'khaali', or 'sam' (sam is the first taali)


@dataclass(frozen=True)
class Taal:
    name: str
    vibhags: tuple[Vibhag, ...]

    @property
    def matra_count(self) -> int:
        return sum(v.length for v in self.vibhags)

    @property
    def vibhag_lengths(self) -> tuple[int, ...]:
        return tuple(v.length for v in self.vibhags)

    def matra_marks(self) -> list[str]:
        """Per-matra structural label: 'sam' | 'taali' | 'khaali' | 'plain'.

        The first matra of each vibhag carries its clap; the rest are 'plain'.
        Matra 0 is always 'sam'.
        """
        marks: list[str] = []
        for v in self.vibhags:
            marks.append(v.clap)
            marks.extend("plain" for _ in range(v.length - 1))
        return marks


# Teentaal: 16 matras, 4 vibhags of 4 (0-based: sam=0, taali=4, khaali=8, taali=12).
TEENTAAL = Taal(
    name="teentaal",
    vibhags=(
        Vibhag(length=4, clap="sam"),
        Vibhag(length=4, clap="taali"),
        Vibhag(length=4, clap="khaali"),
        Vibhag(length=4, clap="taali"),
    ),
)

TAALS: dict[str, Taal] = {t.name: t for t in (TEENTAAL,)}


def get_taal(name: str) -> Taal:
    try:
        return TAALS[name]
    except KeyError:
        raise ValueError(
            f"Unknown taal {name!r}. Available: {', '.join(sorted(TAALS))}"
        ) from None
