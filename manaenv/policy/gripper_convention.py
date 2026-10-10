"""Gripper unit conversion between a policy's wire convention and sim units.

Why this exists
---------------
A policy's gripper channel and the simulator's gripper joint do not
necessarily speak the same units, and nothing in the wire protocol declares
which is in use.  Models trained on ``x2robot_dataset`` v1 / lerobot see
gripper values that the dataset builder pre-divided by the source robot's
gripper scale, so they emit ``[0, 1]`` (0 = closed, 1 = fully open).  The
simulator drives a joint in the embodiment's own native units — drive-joint
radians for EX001-6R (H: 5.717175, G: 1.89), metres for Franka's continuous
parallel jaw, and a ``[-1, +1]`` sign channel for Franka's binary mode.

Feeding a ``[0, 1]`` command straight into a radian joint reads "fully open"
as ~1 rad out of 5.7 and the gripper barely moves.  The failure is silent:
:class:`~manaenv.actions.clamped_raw_gripper.ClampedRawGripperAction` clamps
into the valid range, so the command is legal, just wrong.

Design
------
Two axes, deliberately kept independent so they add rather than multiply as
models and embodiments accumulate:

  * **The model** declares its convention once, in its ``deploy.yaml``
    ``metadata`` (``gripper_convention: normalized_01 | raw``).  This is a
    property of the training data, which the model owner knows.  It rides the
    existing one-way PolicySpace metadata handshake, so no new channel and no
    sim-side edit is needed to onboard a model.
  * **The embodiment** owns the units, exposed through its existing
    ``get_gripper_states() -> {"open": float, "closed": float}``.  The scale is
    never hardcoded here, so the web UI's H/G selection propagates for free and
    cannot disagree with what the sim actually drives.

``raw`` is the default for an undeclared convention: the PolicySpace code path
is shared with policies that predate this mechanism (cosmos3, droid) and were
already emitting sim units correctly.  A model that *declares*
``normalized_01`` against an embodiment that cannot report its gripper range
raises instead of guessing.

The conversion is symmetric and both directions matter.  The action path
(model -> sim) is the visible one.  The observation path (sim -> model) is
easier to forget and just as damaging: the model has only ever seen ``[0, 1]``
state, so handing it raw radians is outside its trained state range. That does not
show up in openloop replay — there both state and ground truth come from the
same normalized dataset — only as a degraded closed-loop success rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

CONVENTION_KEY = "gripper_convention"
LIMITS_KEY = "gripper_limits"
WIRE_INDICES_KEY = "gripper_wire_indices"
STATE_INDICES_KEY = "gripper_state_indices"

NORMALIZED_01 = "normalized_01"
RAW = "raw"

_SUPPORTED = (NORMALIZED_01, RAW)


def normalize_convention(raw: Any) -> str:
    """Validate and canonicalize a declared convention string.

    An unset/empty declaration resolves to :data:`RAW` — see the module
    docstring on why the permissive default is the safe one here.  An
    *unrecognized* value raises: a typo'd convention must not silently fall
    back to pass-through, because that is exactly the silent-wrong-scale
    failure this module exists to prevent.
    """
    if raw is None:
        return RAW
    value = str(raw).strip()
    if not value:
        return RAW
    if value not in _SUPPORTED:
        raise ValueError(
            f"Unsupported {CONVENTION_KEY}={raw!r}; expected one of {list(_SUPPORTED)}. "
            "Declare it in the policy's deploy.yaml metadata."
        )
    return value


def read_gripper_limits(embodiment: Any) -> tuple[float, float] | None:
    """Return ``(closed, open)`` in the embodiment's native gripper units.

    Reads the embodiment's existing ``get_gripper_states()`` hook rather than
    introducing a parallel accessor, so an embodiment that already works with
    interactive control and policy evaluation need no change. Returns ``None`` when
    the embodiment does not implement it (most do not yet) or reports a
    degenerate range; the caller decides whether that is fatal.
    """
    getter = getattr(embodiment, "get_gripper_states", None)
    if not callable(getter):
        return None
    states = getter()
    if not isinstance(states, dict):
        return None
    try:
        closed = float(states["closed"])
        open_ = float(states["open"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(closed) and np.isfinite(open_)):
        return None
    if abs(open_ - closed) < 1e-9:
        return None
    return closed, open_


def limits_to_metadata(limits: tuple[float, float]) -> dict[str, float]:
    closed, open_ = limits
    return {"closed": float(closed), "open": float(open_)}


def limits_from_metadata(metadata: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(metadata, dict):
        return None
    entry = metadata.get(LIMITS_KEY)
    if not isinstance(entry, dict):
        return None
    try:
        closed = float(entry["closed"])
        open_ = float(entry["open"])
    except (KeyError, TypeError, ValueError):
        return None
    if abs(open_ - closed) < 1e-9:
        return None
    return closed, open_


def to_sim_units(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Map ``[0, 1]`` policy output onto ``[closed, open]`` sim units.

    Clipping comes first and is not optional.  Flow-matching sampling noise
    plus ``disable_normalizer_clamp`` lets the model emit slightly outside
    ``[0, 1]``; scaling before clipping would amplify the excursion by the
    gripper scale instead of bounding it.
    """
    closed, open_ = limits
    frac = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return (closed + frac * (open_ - closed)).astype(np.float32)


def to_normalized_01(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Map ``[closed, open]`` sim units back onto the model's ``[0, 1]`` range."""
    closed, open_ = limits
    frac = (np.asarray(values, dtype=np.float32) - closed) / (open_ - closed)
    return np.clip(frac, 0.0, 1.0).astype(np.float32)


def read_wire_indices(metadata: dict[str, Any] | None) -> Sequence[int] | None:
    """Return the gripper column indices of the action wire, if declared."""
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get(WIRE_INDICES_KEY)
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    return [int(index) for index in raw]


def read_state_indices(metadata: dict[str, Any] | None) -> dict[str, int]:
    """Return ``{state_field_name: gripper_index}``, empty when undeclared."""
    if not isinstance(metadata, dict):
        return {}
    raw = metadata.get(STATE_INDICES_KEY)
    if not isinstance(raw, dict):
        return {}
    return {str(name): int(index) for name, index in raw.items()}


def convert_wire_actions(
    actions: np.ndarray,
    *,
    limits: tuple[float, float],
    indices: Sequence[int],
) -> np.ndarray:
    """Return a copy of ``(T, D)`` wire actions with gripper columns rescaled.

    Out-of-range indices raise rather than being skipped: a wire that is
    narrower than its declared gripper slot means the layout declaration and
    the model disagree, and continuing would leave the gripper unconverted.
    """
    arr = np.array(actions, dtype=np.float32, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"wire actions must be (T, D); got shape {arr.shape}")
    width = int(arr.shape[1])
    for index in indices:
        if not (0 <= index < width):
            raise ValueError(
                f"{WIRE_INDICES_KEY} declares gripper column {index}, "
                f"but the policy returned a {width}-wide action wire."
            )
        arr[:, index] = to_sim_units(arr[:, index], limits)
    return arr


def convert_state_field(
    values: np.ndarray,
    *,
    limits: tuple[float, float],
    index: int,
) -> np.ndarray:
    """Return a copy of a 1-D state vector with its gripper entry normalized."""
    arr = np.array(values, dtype=np.float32, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"state field must be 1-D; got shape {arr.shape}")
    if not (0 <= index < int(arr.shape[0])):
        raise ValueError(
            f"{STATE_INDICES_KEY} declares gripper index {index}, "
            f"but the simulator emitted a {int(arr.shape[0])}-wide state field."
        )
    arr[index] = float(to_normalized_01(arr[index : index + 1], limits)[0])
    return arr
