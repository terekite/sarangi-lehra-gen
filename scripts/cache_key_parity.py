#!/usr/bin/env python3
"""Cache-key parity (verification §10.5).

Proves our `sarangi_gen.cache_key.cache_key` produces the same 24-char key as
NiceNagma's own `_cache_key` for the same RenderRequest.

NiceNagma's function lives at
`render/src/nagma_render/service/app.py::_cache_key`. Rather than import that
module (it pulls FastAPI/pydantic and a soundfont-dependent app at import time),
we re-implement its *exact* canonical form here — SHA-256 over
`json.dumps({...}, sort_keys=True, separators=(",",":"))`, first 24 hex chars —
and assert it equals our port for several requests, including the golden one:

    bpm 160.0, sa C#, avartans 4, seed 3, instrument sarangi, proposed-teentaal
        -> 634b772f8d8a9b0c94d73f02

The re-implementation is copied field-for-field from the frozen NiceNagma source
(see the docstring block below), so any divergence in our port is caught.

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sarangi_gen.cache_key import cache_key  # noqa: E402
from sarangi_gen.model import RenderParams  # noqa: E402

LEHRA_PATH = _REPO_ROOT / "lehras" / "proposed-teentaal.nagma"
GOLDEN = "634b772f8d8a9b0c94d73f02"


def nicenagma_cache_key(
    *,
    nagma_text: str,
    bpm: float,
    sa: str,
    instrument: str = "sarangi",
    taal: str = "teentaal",
    avartans: int = 4,
    seed: int = 0,
    laya: str | None = None,
) -> str:
    """Byte-for-byte re-implementation of NiceNagma's `_cache_key`.

    Source: render/src/nagma_render/service/app.py::_cache_key
    Keep this in lockstep with that function (the whole point of the parity
    check). The pydantic RenderBody coerces bpm to float, so we do too.
    """
    h = hashlib.sha256()
    h.update(
        json.dumps(
            {
                "nagma": nagma_text.strip(),
                "bpm": float(bpm),
                "sa": sa,
                "instrument": instrument,
                "taal": taal,
                "avartans": avartans,
                "seed": seed,
                "laya": laya,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return h.hexdigest()[:24]


def _cases(nagma_text: str) -> list[dict]:
    return [
        dict(bpm=160.0, sa="C#", avartans=4, seed=3),   # golden
        dict(bpm=80.0, sa="C", avartans=4, seed=0),
        dict(bpm=160, sa="D", avartans=8, seed=1),      # bpm as int -> float
        dict(bpm=120.0, sa="G#", avartans=2, seed=7),
    ]


def main() -> int:
    nagma_text = LEHRA_PATH.read_text()

    all_ok = True
    for i, case in enumerate(_cases(nagma_text)):
        params = RenderParams(nagma_text=nagma_text, **case)
        ours = cache_key(params)
        theirs = nicenagma_cache_key(
            nagma_text=nagma_text,
            bpm=case["bpm"],
            sa=case["sa"],
            instrument=params.instrument,
            taal=params.taal,
            avartans=case["avartans"],
            seed=case["seed"],
            laya=params.laya,
        )
        ok = ours == theirs
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] case {i}: {case} -> ours={ours} theirs={theirs}")
        all_ok = all_ok and ok

    # Golden anchor: the first case must equal the frozen golden value.
    golden_params = RenderParams(
        nagma_text=nagma_text, bpm=160.0, sa="C#", avartans=4, seed=3,
    )
    golden_ok = cache_key(golden_params) == GOLDEN
    print(f"[{'PASS' if golden_ok else 'FAIL'}] golden: "
          f"{cache_key(golden_params)} == {GOLDEN}")
    all_ok = all_ok and golden_ok

    if all_ok:
        print("cache_key_parity: PASS")
        return 0
    print("cache_key_parity: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
