"""Cache-key parity (verification §10.5).

The golden hash is the value NiceNagma's `_cache_key` produces for this exact
RenderRequest — if this test ever changes value, the app can no longer auto-find
shipped loops. `scripts/cache_key_parity.py` (workstream E) additionally checks
against NiceNagma's live function.
"""

from __future__ import annotations

from sarangi_gen.cache_key import cache_key, output_filename
from sarangi_gen.model import RenderParams

GOLDEN = "634b772f8d8a9b0c94d73f02"


def test_golden_cache_key(golden_params):
    assert cache_key(golden_params) == GOLDEN


def test_bpm_is_float_in_key(proposed_teentaal_text):
    # bpm 160 (int) and 160.0 (float) must hash identically — both -> "160.0".
    p_int = RenderParams(nagma_text=proposed_teentaal_text, bpm=160, sa="C#", avartans=4, seed=3)
    p_flt = RenderParams(nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=4, seed=3)
    assert cache_key(p_int) == cache_key(p_flt) == GOLDEN


def test_key_is_24_hex(golden_params):
    key = cache_key(golden_params)
    assert len(key) == 24
    assert all(c in "0123456789abcdef" for c in key)


def test_format_excluded_from_key(proposed_teentaal_text):
    wav = RenderParams(nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=4, seed=3, format="wav")
    flac = RenderParams(nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=4, seed=3, format="flac")
    assert cache_key(wav) == cache_key(flac)
    assert output_filename(wav).endswith(".wav")
    assert output_filename(flac).endswith(".flac")


def test_seed_and_sa_change_key(proposed_teentaal_text):
    base = RenderParams(nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=4, seed=3)
    assert cache_key(base) != cache_key(
        RenderParams(nagma_text=proposed_teentaal_text, bpm=160.0, sa="C#", avartans=4, seed=4)
    )
    assert cache_key(base) != cache_key(
        RenderParams(nagma_text=proposed_teentaal_text, bpm=160.0, sa="D", avartans=4, seed=3)
    )
