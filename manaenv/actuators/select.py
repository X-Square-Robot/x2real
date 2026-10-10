"""Actuator selection shared by every manaenv embodiment.

Single home for the YAML ``actuator`` knob. Each name maps to an actuator cfg
class; ``actuator_params`` is a per-group override bag (keyed by actuator-group
name) applied uniformly to every model. A group may carry its own ``actuator``
key to override the robot-level model for just that group (e.g. StablePD arms
with implicit-PD grippers). isaaclab is imported at module load (embodiments
already pull it).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from manaenv.actuators.bias_forces import sync_stable_pd_gravity_compensation_with_articulation_cfg
from manaenv.actuators.discovery import ACTUATOR_SPECS
from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
from manaenv.pipeline.registry.global_provider import global_registry
from manaenv.utils.debug_print import manaprint

ACTUATOR_KINDS = frozenset(ACTUATOR_SPECS)

# YAML short keys accepted in an actuator_params group, mapped to the cfg field.
_KEY_ALIASES = {"joint_names": "joint_names_expr"}

# Base fields copied verbatim from the source Implicit cfg when converting to
# another cfg class. Deliberately excludes ``effort_limit_sim`` /
# ``velocity_limit_sim``: SceneCfg authors the ``_sim`` mirrors, and copying
# them would clamp a YAML-authored ``effort_limit`` back to the sim value.
_BASE_COPY_FIELDS = (
    "joint_names_expr",
    "stiffness",
    "damping",
    "effort_limit",
    "velocity_limit",
    "friction",
    "armature",
)


def _resolve_overrides(overrides: dict | None, target_cls: type) -> dict:
    """Map YAML override keys (with aliases) onto valid target_cls fields."""
    valid = {f.name for f in dataclasses.fields(target_cls)} - {"class_type"}
    out: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        field = _KEY_ALIASES.get(key, key)
        if field in valid:
            out[field] = value
        else:
            manaprint(f"WARNING: actuator_params override {key!r} is not a field of {target_cls.__name__}; dropped.")
    return out


def _convert(act_cfg: Any, target_cls: type, overrides: dict | None) -> Any:
    """Build target_cls from act_cfg: copy base fields, then apply YAML overrides."""
    field_names = {f.name for f in dataclasses.fields(target_cls)}
    kwargs = {f: getattr(act_cfg, f) for f in _BASE_COPY_FIELDS if hasattr(act_cfg, f) and f in field_names}
    # Let target_cls set its own default class_type (the actuator class); overriding it with
    # target_cls itself (the *cfg* class) makes the articulation call Cfg(cfg=...) and crash.
    kwargs.update(_resolve_overrides(overrides, target_cls))
    return target_cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class _ModelPolicy:
    """A model's deviations from the generic per-group conversion.

    Models without an entry use the generic conversion. Only post-conversion
    behavior that differs from the generic path lives here.
    """

    post: Callable[[Any], None] | None = None
    """Runs after a successful conversion."""


def _post_stable_pd(robot_cfg: Any) -> None:
    """``stablepd``: sync gravity compensation from the articulation.

    ``use_gravity_compensation`` is derived from the articulation's
    ``disable_gravity`` flag (not YAML); the main assembly path re-syncs it
    idempotently, so this is a construction-time double-check.
    """
    sync_stable_pd_gravity_compensation_with_articulation_cfg(robot_cfg)


_GENERIC_POLICY = _ModelPolicy()

# Only models that deviate from the generic per-group conversion appear here.
_MODEL_POLICIES: dict[str, _ModelPolicy] = {
    "stablepd": _ModelPolicy(post=_post_stable_pd),
}


def _normalize_params(params: dict) -> dict:
    """Flatten the model-grouped form to the per-group form.

    ``actuator_params`` accepts two shapes: per-group (``{group: overrides}``,
    optionally with an ``actuator`` key inside a group) and model-grouped
    (``{model: {group: overrides}}``). The latter is detected when every
    top-level key is a known actuator kind, then flattened by injecting the
    model as each group's ``actuator`` key.
    """
    if not params or not all(k in ACTUATOR_KINDS for k in params):
        return params
    flat: dict[str, dict] = {}
    for model, groups in params.items():
        for name, overrides in (groups or {}).items():
            if name in flat:
                raise ValueError(f"actuator group {name!r} is listed under multiple models in actuator_params.")
            flat[name] = {**(overrides or {}), "actuator": model}
    return flat


def _group_models(actuators: dict, params: dict, default: str) -> dict[str, str]:
    """Per convertible group, the model to apply: the group's own ``actuator``
    key in ``actuator_params`` when present, else the robot-level default."""
    from isaaclab.actuators import ImplicitActuatorCfg

    models: dict[str, str] = {}
    for name, act_cfg in actuators.items():
        if not isinstance(act_cfg, ImplicitActuatorCfg):
            continue
        group_model = (params.get(name) or {}).get("actuator", default)
        if group_model not in ACTUATOR_KINDS:
            raise ValueError(
                f"Unknown actuator {group_model!r} for group {name!r}; choose one of {sorted(ACTUATOR_KINDS)}."
            )
        models[name] = group_model
    return models


def _convert_actuators(robot_cfg: Any, actuators: dict, group_models: dict[str, str], params: dict) -> None:
    """Convert ``robot_cfg.actuators`` Implicit entries to each group's cfg class.

    Per group, YAML ``actuator_params[group]`` overrides are applied on top of
    the source Implicit fields. A group with no overrides falls back to the
    SceneCfg-authored values (for ``pd`` this preserves the original object
    identity).
    """
    replaced = {}
    for name, act_cfg in actuators.items():
        model = group_models.get(name)
        if model is None:
            replaced[name] = act_cfg
            continue
        target_cls = global_registry.get_impl(model)
        if target_cls is None:
            raise ValueError(f"Unknown actuator {model!r}; choose one of {sorted(ACTUATOR_KINDS)}.")
        overrides = dict(params.get(name) or {})
        overrides.pop("actuator", None)
        if target_cls is type(act_cfg):
            # Same class: keep identity when untouched, else patch in place.
            replaced[name] = (
                dataclasses.replace(act_cfg, **_resolve_overrides(overrides, target_cls)) if overrides else act_cfg
            )
        else:
            replaced[name] = _convert(act_cfg, target_cls, overrides)
    robot_cfg.actuators = replaced
    counts = {m: sum(1 for gm in group_models.values() if gm == m) for m in sorted(set(group_models.values()))}
    manaprint(f"INFO: actuator groups applied: {counts}.")


def apply_actuator(robot_cfg: Any, model: str, *, actuator_params: dict | None = None) -> None:
    """Rewrite ``robot_cfg.actuators`` to the control law named by ``model``.

    Generic flow for every model: convert each ``ImplicitActuatorCfg`` entry to
    the model's cfg class while applying per-group ``actuator_params`` overrides
    (other cfgs pass through), then run the optional ``post`` hook.

    * ``pd`` -- keep ImplicitActuatorCfg (engine/native PD); a group
      with no overrides preserves the SceneCfg cfg object unchanged.
    * ``stablepd`` -- StablePDActuatorCfg (Tan 2011 implicit); ``post`` syncs
      gravity compensation from the articulation.

    ``actuator_params`` is keyed by actuator-group name; each group's dict holds
    cfg field overrides (``stiffness``/``damping``/``joint_names``/...) plus an
    optional ``actuator`` key selecting a different model for that group alone.
    Alternatively the whole dict may be keyed by model name with nested group
    dicts (see :func:`_normalize_params`). Unknown override keys are dropped
    with a warning.
    """
    if model not in ACTUATOR_KINDS:
        raise ValueError(f"Unknown actuator {model!r}; choose one of {sorted(ACTUATOR_KINDS)}.")

    params = _normalize_params(dict(actuator_params or {}))
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return

    group_models = _group_models(actuators, params, model)
    used = sorted(set(group_models.values()))
    bootstrap_discovery(("manaenv.actuators",))

    _convert_actuators(robot_cfg, actuators, group_models, params)

    for used_model in used:
        post = _MODEL_POLICIES.get(used_model, _GENERIC_POLICY).post
        if post is not None:
            post(robot_cfg)
