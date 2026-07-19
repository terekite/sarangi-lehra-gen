"""Grammar-parity (verification §10.1) and event-timing tests for parse.py."""

from __future__ import annotations

import pytest

from sarangi_gen.model import RenderParams
from sarangi_gen.parse import NagmaParseError, build_events, parse_nagma

# A structurally-valid teentaal skeleton (4 vibhags of 4 single-note matras).
VALID_16 = """\
S S S S
S S S S
S S S S
S S S S
"""


# --------------------------------------------------------------------------- #
# parse_nagma — grammar parity
# --------------------------------------------------------------------------- #
def test_parses_16_matras_4_vibhags_of_4(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    assert doc.taal == "teentaal"
    assert len(doc.matras) == 16
    # 4 vibhags of 4 == indices 0..15, contiguous.
    assert [m.index for m in doc.matras] == list(range(16))


def test_source_text_is_raw_input(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    assert doc.source_text == proposed_teentaal_text


def test_spot_check_split_matras(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)

    # matra 3 (0-based) is 'N.,S' -> [N mandra, S madhya]
    m3 = doc.matras[3].notes
    assert [(n.kind, n.swar, n.octave) for n in m3] == [
        ("swar", "N", "mandra"),
        ("swar", "S", "madhya"),
    ]

    # matra 11 (0-based) is 'R.,S.' -> [R mandra, S mandra]
    m11 = doc.matras[11].notes
    assert [(n.kind, n.swar, n.octave) for n in m11] == [
        ("swar", "R", "mandra"),
        ("swar", "S", "mandra"),
    ]


def test_octave_markers_and_madhya_default():
    doc = parse_nagma(VALID_16)
    n = doc.matras[0].notes[0]
    assert (n.kind, n.swar, n.octave) == ("swar", "S", "madhya")

    taar = parse_nagma("S' S S S\nS S S S\nS S S S\nS S S S")
    assert taar.matras[0].notes[0].octave == "taar"


def test_subdivision_cap_four_allowed_five_raises():
    # 4 comma-separated slots is allowed (chaugun).
    ok = parse_nagma("S,S,S,S S S S\nS S S S\nS S S S\nS S S S")
    assert len(ok.matras[0].notes) == 4

    # 5 slots exceeds the enforced code limit of 4.
    with pytest.raises(NagmaParseError):
        parse_nagma("S,S,S,S,S S S S\nS S S S\nS S S S\nS S S S")


def test_sustain_slot_within_matra_ok():
    doc = parse_nagma("S,- S S S\nS S S S\nS S S S\nS S S S")
    assert [n.kind for n in doc.matras[0].notes] == ["swar", "sustain"]


def test_leading_sustain_at_sam_raises():
    with pytest.raises(NagmaParseError):
        parse_nagma("- S S S\nS S S S\nS S S S\nS S S S")


def test_wrong_vibhag_count_raises():
    with pytest.raises(NagmaParseError):
        parse_nagma("S S S S\nS S S S\nS S S S")  # only 3 vibhags


def test_wrong_matra_count_raises():
    with pytest.raises(NagmaParseError):
        parse_nagma("S S S\nS S S S\nS S S S\nS S S S")  # vibhag 1 has 3 matras


def test_unknown_swar_raises():
    with pytest.raises(NagmaParseError):
        parse_nagma("Z S S S\nS S S S\nS S S S\nS S S S")


def test_pipe_and_newline_equivalent():
    a = parse_nagma("S S S S | S S S S | S S S S | S S S S")
    b = parse_nagma(VALID_16)
    assert [m.to_dict() for m in a.matras] == [m.to_dict() for m in b.matras]


def test_comments_and_blank_lines_ignored():
    text = (
        "# header comment\n\n"
        "S S S S\n"
        "  # indented comment\n"
        "S S S S\n\n"
        "S S S S\n"
        "S S S S\n"
    )
    doc = parse_nagma(text)
    assert len(doc.matras) == 16


# --------------------------------------------------------------------------- #
# build_events — timing, structure, determinism
# --------------------------------------------------------------------------- #
def _params(text, **kw):
    base = dict(nagma_text=text, bpm=160.0, sa="C#", avartans=4, seed=3)
    base.update(kw)
    return RenderParams(**base)


def test_structural_events_land_on_exact_matra_grid(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    params = _params(proposed_teentaal_text)
    events = build_events(doc, params)

    structural = [e for e in events if e.structural]
    # Every slot-0 of every matra is a struck swar in this lehra -> one
    # structural event per matra per avartan.
    assert len(structural) == 16 * params.avartans

    md = params.matra_dur_s
    for e in structural:
        k = round(e.start_s / md)
        assert abs(e.start_s - k * md) < 1e-9
        # And it maps back to the right global matra index.
        assert k == e.avartan * params.matra_count + e.matra


def test_events_sorted_by_start(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    events = build_events(doc, _params(proposed_teentaal_text))
    starts = [e.start_s for e in events]
    assert starts == sorted(starts)


def test_avartan_indices_present(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    params = _params(proposed_teentaal_text)
    events = build_events(doc, params)
    assert {e.avartan for e in events} == set(range(params.avartans))


def test_non_structural_notes_get_jitter_structural_do_not(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    params = _params(proposed_teentaal_text)
    events = build_events(doc, params)
    md = params.matra_dur_s
    # Structural onsets are exact; non-structural ones are generally off-grid.
    off_grid = 0
    for e in events:
        if not e.structural and e.matra is not None:
            expected = e.avartan * params.matra_count * md + e.matra * md
            # slot-1 notes and graces sit away from the matra boundary anyway,
            # but a jittered onset is (almost surely) not an exact multiple.
            if abs((e.start_s / md) - round(e.start_s / md)) > 1e-9:
                off_grid += 1
    assert off_grid > 0


def test_determinism(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    params = _params(proposed_teentaal_text)
    a = build_events(doc, params)
    b = build_events(doc, params)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


def test_seed_changes_variation(proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    a = build_events(doc, _params(proposed_teentaal_text, seed=1))
    b = build_events(doc, _params(proposed_teentaal_text, seed=2))
    # Different seeds must change the non-structural micro-variation.
    assert [e.to_dict() for e in a] != [e.to_dict() for e in b]


def test_uses_golden_params_fixture(golden_params, proposed_teentaal_text):
    doc = parse_nagma(proposed_teentaal_text)
    events = build_events(doc, golden_params)
    structural = [e for e in events if e.structural]
    assert len(structural) == 16 * golden_params.avartans
