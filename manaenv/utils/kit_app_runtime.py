"""Small runtime gates for interactive Kit application loops."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any


def close_kit_app(app: Any, *, exit_code: int = 0) -> None:
    """Finalize Python views and preserve failures through Kit fast shutdown."""
    # Tensor views can participate in Python cycles. If app.close() unloads
    # physics.tensors first, their late finalizers call into a dead extension.
    gc.collect()
    if exit_code:
        app.close(exit_code=int(exit_code))
    else:
        app.close()


def update_kit_app_once(app: Any, *, after_update: Callable[[], None] | None = None) -> bool:
    """Pump one Kit frame and return whether more application work is safe."""
    if not app.is_running():
        return False

    try:
        app.update()
        if not app.is_running():
            return False
        if after_update is not None:
            after_update()
        return app.is_running()
    except ReferenceError:
        if app.is_running():
            raise
        return False


def run_kit_app_loop(app: Any, *, after_update: Callable[[], None] | None = None) -> None:
    """Pump Kit until it stops, running optional work only after a live frame."""
    while update_kit_app_once(app, after_update=after_update):
        pass


def pump_kit_app_frames(app: Any, count: int) -> bool:
    """Pump at most *count* frames, returning ``False`` when Kit has stopped."""
    for _ in range(count):
        if not update_kit_app_once(app):
            return False
    return True


def run_kit_app_step_loop(app: Any, *, step: Callable[[], None]) -> None:
    """Run a simulation step callback until Kit stops without adding a Kit tick."""
    while app.is_running():
        try:
            step()
        except ReferenceError:
            if app.is_running():
                raise
            return
