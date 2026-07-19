"""Output naming = NiceNagma's cache key, so the app auto-finds a shipped loop
with zero wiring. Exact port of `render/src/nagma_render/service/app.py::_cache_key`
and `app/lib/services/loop_renderer.dart::cacheKey`.

SHA-256 over a canonical JSON object, first 24 hex chars. `sort_keys=True` +
compact separators make Python's output byte-identical to Dart's `jsonEncode`.
`bpm` is a float on both sides (80 -> "80.0"). `format`/`schema_version` are
excluded from the key; `laya` is hashed as-is (null even when auto-resolved).
"""

from __future__ import annotations

import hashlib
import json

from .model import RenderParams


def cache_key(params: RenderParams) -> str:
    obj = {
        "avartans": params.avartans,
        "bpm": float(params.bpm),
        "instrument": params.instrument,
        "laya": params.laya,
        "nagma": params.nagma_text.strip(),
        "sa": params.sa,
        "seed": params.seed,
        "taal": params.taal,
    }
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


def output_filename(params: RenderParams) -> str:
    ext = "flac" if params.format == "flac" else "wav"
    return f"{cache_key(params)}.{ext}"
