"""Shared pytest fixtures. Owned by the foundation so no workstream races on it.

Adds the repo root to sys.path (so `import sarangi_gen` works without install)
and exposes the byte-identical proposed-teentaal lehra text + a canonical
RenderParams for the golden cache-key fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEHRA_PATH = REPO_ROOT / "lehras" / "proposed-teentaal.nagma"


@pytest.fixture(scope="session")
def proposed_teentaal_text() -> str:
    return LEHRA_PATH.read_text()


@pytest.fixture(scope="session")
def golden_params(proposed_teentaal_text):
    from sarangi_gen.model import RenderParams

    return RenderParams(
        nagma_text=proposed_teentaal_text,
        bpm=160.0,
        sa="C#",
        avartans=4,
        seed=3,
    )
