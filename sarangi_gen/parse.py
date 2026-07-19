"""Sargam text-grammar parser and event compiler.

Two public entry points, both faithful ports of NiceNagma
(`core/src/nagma_core/{parser,reduce,compiler,performance}.py`):

  * ``parse_nagma`` : raw text -> validated :class:`NagmaDoc` (grammar v1).
  * ``build_events``: NagmaDoc + :class:`RenderParams` -> timed :class:`Event`
    super-loop, i.e. NiceNagma's ``realize()`` audio path
    ``compile_score(reduce_doc(doc, laya_for_bpm(bpm)), params)``.

Timing invariants (verification §10.4): every duration derives from
``matra_dur_s = 60/bpm`` computed in SECONDS and never accumulated in samples,
so structural onsets (matra boundaries / sam) land on mathematically exact
multiples of ``matra_dur_s``. Only non-structural onsets are ever perturbed.
Everything is deterministic given ``(doc, params)``.

Grammar (v1)
------------
- Vibhags are separated by a newline OR ``|`` (equivalent). Within a vibhag,
  matras are whitespace-separated.
- A matra subdivides into up to 4 equal slots (chaugun) via ``,``. Slots may be
  sustains.
- A note is a swar symbol optionally followed by an octave marker:
  ``'`` -> taar (upper), ``.`` -> mandra (lower), none -> madhya.
- ``-`` is a sustain (holds the previous note); a nagma may not begin with one.
- Lines beginning with ``#`` are comments; blank lines are ignored.
"""

from __future__ import annotations

import random

from .model import Event, Matra, NagmaDoc, Note, RenderParams
from .sargam import OCTAVE_MARKERS, VALID_SWARS, note_to_midi
from .taal import Taal, get_taal


class NagmaParseError(ValueError):
    """Raised on any grammar or structural validation failure.

    The message is user-facing and should read cleanly in an inline validation
    UI (points at the offending token / vibhag / matra).
    """


# --------------------------------------------------------------------------- #
# parse_nagma  (port of nagma_core.parser)
# --------------------------------------------------------------------------- #
def _parse_note(token: str, *, vibhag_no: int, matra_no: int) -> Note:
    tok = token.strip()
    if not tok:
        raise NagmaParseError(
            f"Empty note in vibhag {vibhag_no}, matra {matra_no}."
        )
    if tok == "-":
        return Note(kind="sustain")

    octave = "madhya"
    core = tok
    marker = tok[-1]
    if marker in OCTAVE_MARKERS:
        octave = OCTAVE_MARKERS[marker]
        core = tok[:-1]

    if core not in VALID_SWARS:
        raise NagmaParseError(
            f"Unknown swar {tok!r} in vibhag {vibhag_no}, matra {matra_no}. "
            f"Valid swars: S r R g G m M P d D n N "
            f"(add ' for taar / . for mandra)."
        )
    return Note(kind="swar", swar=core, octave=octave)


def _parse_matra(cell: str, *, vibhag_no: int, matra_no: int) -> list[Note]:
    # A matra may be subdivided into up to 4 equal slots (chaugun) via ',' — this
    # is the density a slow (vilambit) lehra needs. '-' sustains a slot, which
    # covers uneven rhythms (e.g. 'S,-,g,m'). Faster layas are derived by reducing
    # this authored density, so authoring is done at max density.
    parts = [p for p in cell.split(",")]
    if len(parts) > 4:
        raise NagmaParseError(
            f"Matra {matra_no} in vibhag {vibhag_no} has {len(parts)} notes; "
            f"a matra allows at most 4 subdivisions (split with ',')."
        )
    return [
        _parse_note(p, vibhag_no=vibhag_no, matra_no=matra_no) for p in parts
    ]


def parse_nagma(text: str, *, taal: str = "teentaal") -> NagmaDoc:
    """Parse text-grammar into a validated NagmaDoc for the given taal."""
    taal_def = get_taal(taal)

    # Strip comments/blank lines, then split into vibhags on newlines AND '|'.
    cleaned_lines = [
        ln for ln in text.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    joined = "\n".join(cleaned_lines)
    vibhag_strs = [
        seg.strip()
        for seg in joined.replace("\n", "|").split("|")
        if seg.strip()
    ]

    expected_vibhags = len(taal_def.vibhags)
    if len(vibhag_strs) != expected_vibhags:
        raise NagmaParseError(
            f"{taal_def.name} has {expected_vibhags} vibhags but got "
            f"{len(vibhag_strs)}. Separate vibhags with a newline or '|'."
        )

    matras: list[Matra] = []
    matra_index = 0
    for vi, (vstr, vib) in enumerate(zip(vibhag_strs, taal_def.vibhags), start=1):
        cells = vstr.split()
        if len(cells) != vib.length:
            raise NagmaParseError(
                f"Vibhag {vi} should have {vib.length} matras but got "
                f"{len(cells)}: {vstr!r}."
            )
        for mi, cell in enumerate(cells, start=1):
            notes = _parse_matra(cell, vibhag_no=vi, matra_no=mi)
            matras.append(Matra(index=matra_index, notes=notes))
            matra_index += 1

    # A leading sustain has nothing to hold, which is almost always a typo.
    first = matras[0].notes[0]
    if first.kind == "sustain":
        raise NagmaParseError(
            "The nagma cannot begin with a sustain '-' (nothing to hold at sam)."
        )

    return NagmaDoc(taal=taal_def.name, matras=matras, source_text=text)


# --------------------------------------------------------------------------- #
# Laya reduction  (port of nagma_core.reduce)
# --------------------------------------------------------------------------- #
# Laya bands by BPM. Soft, tala-dependent thresholds; laya is hashed null in the
# cache key elsewhere — this only changes event density.
VILAMBIT_MAX_BPM = 85
MADHYA_MAX_BPM = 160


def laya_for_bpm(bpm: float) -> str:
    if bpm <= VILAMBIT_MAX_BPM:
        return "vilambit"
    if bpm <= MADHYA_MAX_BPM:
        return "madhya"
    return "drut"


def _copy(note: Note) -> Note:
    return Note(kind=note.kind, swar=note.swar, octave=note.octave)


def _sustain() -> Note:
    return Note(kind="sustain")


def _pitch(note: Note):
    return (note.swar, note.octave) if note.kind == "swar" else None


def _reduce_madhya(doc: NagmaDoc) -> list[Matra]:
    out = []
    for m in doc.matras:
        n = len(m.notes)
        keep = {0, n // 2}  # downbeat + midpoint (caps struck slots at 2)
        notes = [_copy(m.notes[i]) if i in keep else _sustain() for i in range(n)]
        out.append(Matra(index=m.index, notes=notes))
    return out


def _reduce_drut(doc: NagmaDoc, taal: Taal) -> list[Matra]:
    marks = taal.matra_marks()
    out = []
    last_struck = None
    for m in doc.matras:
        slot0 = m.notes[0]
        mark = marks[m.index]
        pitch = _pitch(slot0)
        # Protect the taal outline; elsewhere omit a plain matra that just repeats.
        strike = slot0.kind == "swar" and not (
            mark == "plain" and pitch is not None and pitch == last_struck
        )
        if strike:
            out.append(Matra(index=m.index, notes=[_copy(slot0)]))  # whole-matra note
            last_struck = pitch
        else:
            out.append(Matra(index=m.index, notes=[_sustain()]))    # held through
    return out


def reduce_doc(doc: NagmaDoc, laya: str) -> NagmaDoc:
    """Return a NagmaDoc realized for `laya` ('vilambit'|'madhya'|'drut')."""
    if laya == "vilambit":
        matras = [
            Matra(index=m.index, notes=[_copy(n) for n in m.notes])
            for m in doc.matras
        ]
    else:
        taal = get_taal(doc.taal)
        if laya == "madhya":
            matras = _reduce_madhya(doc)
        elif laya == "drut":
            matras = _reduce_drut(doc, taal)
        else:
            raise ValueError(f"unknown laya {laya!r}")
    return NagmaDoc(
        taal=doc.taal,
        matras=matras,
        raag=doc.raag,
        name=doc.name,
        source_text=doc.source_text,
    )


# --------------------------------------------------------------------------- #
# Performance model  (port of nagma_core.performance) — pure, deterministic.
# --------------------------------------------------------------------------- #
JITTER_S = 0.012            # max abs micro-timing jitter on a non-structural note
LEGATO_OVERLAP_S = 0.045    # overlap added to every note's sounding duration
REARTICULATION_GAP_S = 0.05  # min silence before the SAME pitch sounds again
MIN_NOTE_S = 0.06           # floor on a note's sounding duration after clamping
BASE_VELOCITY = 82
VELOCITY_NOISE = 4
SWELL_MAX_DEPTH = 0.20
SWELL_FULL_AT_S = 0.8
_VELOCITY_MIN = 24
_VELOCITY_MAX = 122

# Grace notes (kan swar): a light neighbour-swar flicked in before a main note.
GRACE_DENSITY = 0.12
GRACE_DENSITY_BY_LAYA = {"vilambit": 0.14, "madhya": 0.09, "drut": 0.03}
GRACE_DUR_S = 0.07
GRACE_LEGATO_S = 0.03
GRACE_MIN_MAIN_S = 0.30
GRACE_VEL_SCALE = 0.68


def _avartan_rng(base_seed: int, avartan: int) -> random.Random:
    """Deterministic RNG unique to (base_seed, avartan) so each cycle varies."""
    return random.Random((base_seed * 2654435761) ^ (avartan * 40503) ^ 0x9E3779B9)


def _grace_rng(base_seed: int, avartan: int) -> random.Random:
    """RNG for ornament placement — a separate stream from timing/dynamics so
    adding grace notes doesn't perturb the existing per-note feel."""
    return random.Random((base_seed * 2246822519) ^ (avartan * 3266489917) ^ 0x85EBCA77)


def _timing_jitter(rng: random.Random, structural: bool) -> float:
    if structural:
        return 0.0
    return rng.uniform(-JITTER_S, JITTER_S)


def _base_velocity(matra: int, mark: str, matra_count: int) -> int:
    """Taal-shaped velocity before per-avartan noise."""
    vel = float(BASE_VELOCITY)
    vel += 10.0 * (matra / matra_count)  # gentle swell toward sam
    if mark == "sam":
        vel += 22.0
    elif mark == "taali":
        vel += 10.0
    elif mark == "khaali":
        vel -= 12.0
    khaali_start = matra_count // 2
    if matra == khaali_start + 1:
        vel -= 4.0
    return int(round(vel))


def _apply_velocity_noise(rng: random.Random, velocity: int) -> int:
    v = velocity + rng.randint(-VELOCITY_NOISE, VELOCITY_NOISE)
    return max(_VELOCITY_MIN, min(_VELOCITY_MAX, v))


def _kan_pitch(midi: int, pitch_set: list[int]) -> int | None:
    """Nearest composition pitch above the main note (upper kan), else below."""
    higher = [p for p in pitch_set if p > midi]
    if higher:
        return min(higher)
    lower = [p for p in pitch_set if p < midi]
    return max(lower) if lower else None


def _swell_depth(rng: random.Random, dur_s: float, structural: bool) -> float:
    """Per-note intra-note swell depth. Longer notes breathe more."""
    d = SWELL_MAX_DEPTH * min(1.0, dur_s / SWELL_FULL_AT_S)
    if structural:
        d *= 0.8
    d *= 0.85 + 0.3 * rng.random()
    return round(max(0.0, min(0.5, d)), 3)


# --------------------------------------------------------------------------- #
# build_events  (port of nagma_core.compiler.realize / compile_score)
# --------------------------------------------------------------------------- #
def _one_avartan_events(
    doc: NagmaDoc, matra_dur_s: float, sa_midi: int, marks: list[str]
) -> list[dict]:
    """Base (jitter-free, base-velocity) events for a single cycle.

    Sustains fold into the preceding struck note's duration. Structural onsets
    (matra boundaries, slot 0) sit on exact multiples of ``matra_dur_s``.
    """
    events: list[dict] = []
    last: dict | None = None  # most recent struck note, for sustain folding

    for matra in doc.matras:
        n = len(matra.notes)
        for slot, note in enumerate(matra.notes):
            slot_dur = matra_dur_s / n
            slot_start = matra.index * matra_dur_s + slot * slot_dur
            structural = slot == 0

            if note.kind == "sustain":
                if last is not None:
                    last["dur_s"] += slot_dur
                # A sustain with no predecessor is silence (parser forbids it at
                # sam; mid-piece it just yields a rest).
                continue

            midi = note_to_midi(note.swar, note.octave, sa_midi)
            ev = {
                "start_s": slot_start,
                "dur_s": slot_dur,
                "midi": midi,
                "matra": matra.index,
                "structural": structural,
                "mark": marks[matra.index],
            }
            events.append(ev)
            last = ev

    return events


def build_events(doc: NagmaDoc, params: RenderParams) -> list[Event]:
    """Compile a NagmaDoc into a super-loop of Events across `params.avartans`.

    Mirrors NiceNagma's harmonium realization path: reduce the authored
    (vilambit) doc to the laya implied by the BPM, then expand into a super-loop
    where each avartan carries its own RNG seed for micro-variation.
    """
    laya = laya_for_bpm(params.bpm)
    reduced = reduce_doc(doc, laya)
    grace_density = GRACE_DENSITY_BY_LAYA.get(laya, GRACE_DENSITY)

    taal_def = get_taal(doc.taal)
    marks = taal_def.matra_marks()
    matra_count = taal_def.matra_count

    matra_dur_s = params.matra_dur_s
    avartan_dur_s = params.avartan_dur_s
    sa_midi = params.sa_midi

    base = _one_avartan_events(reduced, matra_dur_s, sa_midi, marks)

    # Legato overlap, applied once on the base durations.
    for ev in base:
        ev["dur_s"] += LEGATO_OVERLAP_S

    pitch_set = sorted({ev["midi"] for ev in base})

    events: list[Event] = []
    for a in range(params.avartans):
        rng = _avartan_rng(params.seed, a)
        grng = _grace_rng(params.seed, a)  # separate stream: graces don't move feel
        cycle_offset = a * avartan_dur_s
        for ev in base:
            jitter = _timing_jitter(rng, ev["structural"])
            start = cycle_offset + ev["start_s"] + jitter
            vel = _base_velocity(ev["matra"], ev["mark"], matra_count)
            vel = _apply_velocity_noise(rng, vel)
            swell = _swell_depth(rng, ev["dur_s"], ev["structural"])
            events.append(
                Event(
                    start_s=start,
                    dur_s=ev["dur_s"],
                    midi=ev["midi"],
                    velocity=vel,
                    matra=ev["matra"],
                    avartan=a,
                    structural=ev["structural"],
                    swell=swell,
                )
            )
            # Kan swar: flick a neighbour swar in just before this note. It steals
            # time from BEFORE the exact onset, so the grid stays untouched.
            if ev["dur_s"] >= GRACE_MIN_MAIN_S and grng.random() < grace_density:
                kan = _kan_pitch(ev["midi"], pitch_set)
                gstart = start - GRACE_DUR_S
                if kan is not None and kan != ev["midi"] and gstart >= cycle_offset:
                    events.append(
                        Event(
                            start_s=gstart,
                            dur_s=GRACE_DUR_S + GRACE_LEGATO_S,
                            midi=kan,
                            velocity=max(1, int(vel * GRACE_VEL_SCALE)),
                            matra=ev["matra"],
                            avartan=a,
                            structural=False,
                            swell=0.0,
                        )
                    )

    events.sort(key=lambda e: e.start_s)

    # Same-pitch re-articulation: legato must not sustain a note into the next
    # onset of the SAME pitch (that retriggers the sampler mid-note and the
    # repeat sounds cut off). Clamp each note to end a small gap before the next
    # same-pitch onset. Different-pitch overlap (harmonium bleed) is untouched.
    by_pitch: dict[int, list[Event]] = {}
    for e in events:
        by_pitch.setdefault(e.midi, []).append(e)
    for group in by_pitch.values():
        for a, b in zip(group, group[1:]):  # already start-sorted
            latest_end = b.start_s - REARTICULATION_GAP_S
            if a.start_s + a.dur_s > latest_end:
                a.dur_s = max(MIN_NOTE_S, latest_end - a.start_s)

    return events
