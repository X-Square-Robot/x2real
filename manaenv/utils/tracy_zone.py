"""Deferred carb.profiler zones for ManaEnv product code.

Kit/Tracy is optional: importing this module must not load ``carb``. Python
``carb.profiler.begin`` names the zone after the calling function, so this
helper uses ``begin_with_location`` and the caller-supplied zone name.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_TRACY_PROFILER: Any = None
_TRACY_PROFILER_RESOLVED = False
_TRACY_PROFILER_LOCK = threading.Lock()


def _tracy_profiler() -> Any:
    """Return carb.profiler when Kit is loaded; otherwise None."""
    global _TRACY_PROFILER, _TRACY_PROFILER_RESOLVED
    if _TRACY_PROFILER_RESOLVED:
        return _TRACY_PROFILER
    with _TRACY_PROFILER_LOCK:
        if _TRACY_PROFILER_RESOLVED:
            return _TRACY_PROFILER
        _TRACY_PROFILER_RESOLVED = True
        try:
            import carb.profiler as profiler
        except ImportError:
            _TRACY_PROFILER = None
            return None
        _TRACY_PROFILER = profiler
        return profiler


@contextmanager
def tracy_zone(name: str, *, source: str = "tracy_zone.py") -> Iterator[None]:
    """Emit a channel-1 Tracy zone when the profiler plugin is present."""
    profiler = _tracy_profiler()
    if profiler is None:
        yield
        return
    begin_with_location = getattr(profiler, "begin_with_location", None)
    if begin_with_location is None:
        yield
        return
    begin_with_location(1, name, name, source, 0)
    try:
        yield
    finally:
        profiler.end(1)
