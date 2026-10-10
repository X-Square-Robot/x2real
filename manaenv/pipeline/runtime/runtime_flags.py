"""Process-global runtime values, YAML-driven.

Set from ``pipeline.pipeline_runtime`` when the pipeline consumes its runtime
cfg and read by evaluation runtime hooks (e.g. ``stream_viewports``).
Everything runs in one process (main.py -> translator -> build -> run), so a
module-level dict is a valid channel between config parse and loop runtime.
"""

from __future__ import annotations

from typing import Any

_VALUES: dict[str, Any] = {}


def set_value(name: str, value: Any) -> None:
    """Set a structured runtime value (e.g. the stream_viewports mapping)."""
    _VALUES[name] = value


def get_value(name: str, default: Any = None) -> Any:
    """Read a structured runtime value; unset reads as ``default``."""
    return _VALUES.get(name, default)
