#!/usr/bin/env python3
"""Drop-in dry-run (verification §10.6).

Simulates how NiceNagma's SarangiPackRenderer would resolve a shipped loop:

  1. Render (or reuse) one output for a known RenderParams.
  2. Place it in a mock `render-cache/<cacheKey>.<ext>` directory (a tmp dir).
  3. Independently recompute the cacheKey for the same request (as the app's
     `req.cacheKey()` would) and confirm the file resolves at that path.

This proves the cache-key naming lets the app auto-find the loop with zero
wiring. Prints PASS/FAIL; exits non-zero on failure.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sarangi_gen.cache_key import cache_key, output_filename  # noqa: E402
from sarangi_gen.model import RenderParams  # noqa: E402

# Import the pipeline runner from generate.py (repo root).
import generate  # noqa: E402

LEHRA_PATH = _REPO_ROOT / "lehras" / "proposed-teentaal.nagma"


def main() -> int:
    nagma_text = LEHRA_PATH.read_text()
    params = RenderParams(
        nagma_text=nagma_text, bpm=160.0, sa="C#", avartans=4, seed=3,
    )
    key = cache_key(params)
    fname = output_filename(params)

    tmp = Path(tempfile.mkdtemp(prefix="dropin_dryrun_"))
    try:
        # 1. Obtain a render: reuse out/<name> if present, else render fresh.
        existing = _REPO_ROOT / "out" / fname
        source = tmp / fname
        if existing.exists():
            shutil.copyfile(existing, source)
            print(f"reusing existing render: {existing}")
        else:
            generate.render_one(
                params, backend="sine", out_path=source, verbose=False,
            )
            print(f"rendered fresh into tmp: {source.name}")

        # 2. Place it in a mock render-cache/<cacheKey>.<ext> dir.
        cache_dir = tmp / "render-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / fname
        shutil.move(str(source), str(cached_path))

        # 3. App-side resolution: recompute the key and look for the file.
        app_key = cache_key(params)
        resolved = cache_dir / f"{app_key}.{params.format}"

        ok = (
            app_key == key
            and resolved.exists()
            and resolved == cached_path
        )
        if ok:
            print(f"[PASS] cacheKey {app_key} resolves to {resolved}")
            print("dropin_dryrun: PASS")
            return 0
        print(f"[FAIL] key={key} app_key={app_key} "
              f"resolved={resolved} exists={resolved.exists()}")
        print("dropin_dryrun: FAIL")
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
