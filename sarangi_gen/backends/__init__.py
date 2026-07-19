"""Pluggable synthesis backends.

Importing this package must stay dependency-light (numpy only). ``DdspBackend``
defers its TensorFlow / ddsp imports to call time, so importing it here is safe.
"""

from __future__ import annotations

from .ddsp import DdspBackend
from .sine import SineBackend

__all__ = ["SineBackend", "DdspBackend"]
