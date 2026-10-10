"""ManaEnv compatibility hooks for pinned Isaac Lab and Newton actuator APIs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from manaenv.utils.debug_print import manaprint

_PATCH_FLAG = "_manaenv_control_target_alias_compat_v1"
_CONTROL_TARGET_ALIASES = (
    ("joint_target_q", "joint_target_pos"),
    ("joint_target_qd", "joint_target_vel"),
)


def _iter_articulation_cfgs(value: Any, path: str = "", seen: set[int] | None = None) -> Iterator[tuple[str, Any]]:
    """Yield MR30 articulation configs from a composed scene config."""
    if value is None or isinstance(value, (str, bytes, int, float, bool, type)):
        return
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if (
        hasattr(value, "native_actuators")
        and isinstance(getattr(value, "actuators", None), Mapping)
        and (hasattr(value, "prim_path") or type(value).__name__.endswith("ArticulationCfg"))
    ):
        yield path, value
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_articulation_cfgs(item, child_path, seen)
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            yield from _iter_articulation_cfgs(item, child_path, seen)
        return
    if not is_dataclass(value):
        return
    for field in fields(value):
        child_path = f"{path}.{field.name}" if path else field.name
        yield from _iter_articulation_cfgs(getattr(value, field.name, None), child_path, seen)


def _has_stable_pd(cfg: Any) -> bool:
    actuators = getattr(cfg, "actuators", None)
    return isinstance(actuators, Mapping) and any(
        type(actuator_cfg).__name__ == "StablePDActuatorCfg" for actuator_cfg in actuators.values()
    )


def apply_per_articulation_newton_selection(env_cfg: Any, *, runtime_native: bool | None) -> dict[str, bool]:
    """Apply MR30 native selection while retaining legacy runtime overrides.

    With automatic StablePD selection, StablePD articulations opt in and all
    other unselected articulations opt out. An explicit runtime ``True`` keeps
    the legacy all-articulation native behavior; explicit per-articulation
    values always win.
    """
    sim_cfg = getattr(env_cfg, "sim", None)
    if not bool(getattr(sim_cfg, "use_newton_actuators", False)):
        return {}

    candidates = list(_iter_articulation_cfgs(getattr(env_cfg, "scene", None)))
    if not candidates:
        return {}

    stable_pd_present = any(_has_stable_pd(cfg) for _, cfg in candidates)
    if runtime_native is None and not stable_pd_present:
        return {}

    resolved: dict[str, bool] = {}
    for path, cfg in candidates:
        explicit = getattr(cfg, "native_actuators", None)
        if explicit is not None:
            resolved[path] = bool(explicit)
            continue
        if runtime_native is True:
            selected = True
        elif runtime_native is False:
            selected = False
        else:
            selected = _has_stable_pd(cfg)
        setattr(cfg, "native_actuators", selected)
        resolved[path] = selected
    return resolved


def _declares_attribute(cls: type, name: str) -> bool:
    if hasattr(cls, name):
        return True
    if name in getattr(cls, "__annotations__", {}):
        return True
    return name in getattr(cls, "__dataclass_fields__", {})


def _forwarding_property(target: str) -> property:
    def _get(instance: Any) -> Any:
        return getattr(instance, target)

    def _set(instance: Any, value: Any) -> None:
        setattr(instance, target, value)

    return property(_get, _set)


def install_physx_actuator_target_compat() -> bool:
    """Expose Newton 1.5 control-target names on Isaac Lab's PhysX wrapper.

    The pinned Newton actuator graph reads ``joint_target_q`` and
    ``joint_target_qd``. The pinned Isaac Lab wrapper publishes the same
    arrays as ``joint_target_pos`` and ``joint_target_vel``. Install writable
    aliases on the wrapper class without replacing either upstream object.

    Returns ``True`` only when at least one alias is installed.
    """

    from isaaclab_newton.actuators import PhysxActuatorWrapper

    wrapper_class = PhysxActuatorWrapper
    if getattr(wrapper_class, _PATCH_FLAG, False):
        return False

    missing_targets = [
        target
        for alias, target in _CONTROL_TARGET_ALIASES
        if not _declares_attribute(wrapper_class, alias) and not _declares_attribute(wrapper_class, target)
    ]
    if missing_targets:
        raise RuntimeError(
            "Newton actuator target compatibility cannot be installed: "
            f"PhysxActuatorWrapper lacks {', '.join(sorted(missing_targets))}"
        )

    installed = []
    for alias, target in _CONTROL_TARGET_ALIASES:
        if _declares_attribute(wrapper_class, alias):
            continue
        setattr(wrapper_class, alias, _forwarding_property(target))
        installed.append(f"{alias}->{target}")

    setattr(wrapper_class, _PATCH_FLAG, True)
    if installed:
        aliases = ", ".join(installed)
        manaprint(f"INFO: Installed ManaEnv Newton actuator target compatibility aliases ({aliases}).")
    return bool(installed)


__all__ = ["apply_per_articulation_newton_selection", "install_physx_actuator_target_compat"]
