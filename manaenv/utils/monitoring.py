"""Standard embedded monitoring/telemetry API.

Mirrors :mod:`manaenv.utils.debug_print`: a module-level enable flag, a single
active-monitor reference, and a thin :func:`monitor_emit` wrapper that any code
can call without knowing about files, channels, or the DataCenter. The active
monitor is the DataCenter-owned ``MonitoringManager`` (the single writer of the
monitoring channel); it registers itself via :func:`set_active_monitor` during
``DataCenter.setup`` when monitoring is enabled.

Disabled by default. When off (or when no monitor is registered),
:func:`monitor_emit` is a cheap no-op, so a normal pipeline run is unaffected.
Toggle it with the ``--monitor`` CLI flag / ``MANAENV_MONITORING`` env var, which
set :data:`MONITORING_ENABLED`.
"""

from __future__ import annotations

from typing import Any

# Global switch for monitoring. Default OFF — monitoring must never affect a
# normal run unless explicitly enabled.
MONITORING_ENABLED: bool = False

# The active monitor sink. Any object exposing ``record(channel, payload, meta)``.
_active_monitor: Any | None = None


def set_monitoring_enabled(value: bool) -> None:
    """Flip the global monitoring switch at RUNTIME.

    Historically ``MONITORING_ENABLED`` was written only once at process start
    (``main.py`` from ``--monitor`` / ``MANAENV_MONITORING``). The editor service
    needs to turn monitoring on/off per rollout on a live engine, so this is the
    runtime write path. Pair it with ``MonitoringManager.apply_enabled`` (which
    also (de)registers the runner hooks + active sink) — flipping this flag alone
    only gates ``monitor_emit``; it does not attach the sink.
    """
    global MONITORING_ENABLED
    MONITORING_ENABLED = bool(value)


def set_active_monitor(monitor: Any) -> None:
    """Register the active monitor sink (called by the DataCenter manager)."""
    global _active_monitor
    _active_monitor = monitor


def clear_active_monitor() -> None:
    """Unregister the active monitor sink (called on DataCenter cleanup)."""
    global _active_monitor
    _active_monitor = None


def get_active_monitor() -> Any | None:
    return _active_monitor


def is_monitoring_enabled() -> bool:
    """Cheap guard: monitoring is on *and* a sink is registered.

    Call this before doing any monitoring-only work (e.g. computing a snapshot)
    so disabled runs pay zero cost.
    """
    return MONITORING_ENABLED and _active_monitor is not None


def monitor_emit(channel: str, payload: dict[str, Any], **meta: Any) -> None:
    """Forward one monitoring event to the active monitor; no-op when disabled.

    Best-effort and side-effect-only: it never raises into the caller (a
    monitoring failure must not break a pipeline run), mirroring ``manaprint``.

    Args:
        channel: Logical stream name, e.g. ``"stage"`` / ``"run_start"`` / ``"run_end"``.
        payload: The event body (JSON-serialisable dict).
        **meta: Flat metadata merged into the channel envelope (e.g. ``step=``, ``env_ref=``).
    """
    if not MONITORING_ENABLED:
        return
    monitor = _active_monitor
    if monitor is None:
        return
    try:
        monitor.record(channel, payload, meta)
    except Exception:
        # Monitoring is best-effort; swallow everything.
        pass
