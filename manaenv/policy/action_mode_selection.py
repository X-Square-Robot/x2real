"""Shared action-mode selection for policy-driven and planner-driven runs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from manaenv.policy.policy_embodiment import PolicyEmbodimentCFG
from manaenv.utils.debug_print import manaprint


def _order_policy_cfg(order: Any) -> Any:
    """Resolve the built ManaPolicyCFG from a typed order.

    Evaluation orders carry ``policy_cfg`` directly.
    carry it on the reflected ``planner`` CFG (PolicyPlanner).
    """
    cfg = getattr(order, "policy_cfg", None)
    if cfg is not None:
        return cfg
    planner = getattr(order, "planner", None)
    return getattr(planner, "policy_cfg", None)


def resolve_order_action_mode(order: Any = None) -> str | None:
    """Resolve action mode: the policy's declared mode, then its embodiment default.

    The policy (``ManaPolicyCFG.action_mode``) owns "which interface I drive";
    when it is empty the embodiment's own ``action_mode`` default applies. There
    is no order-level override — selection is a policy property, mirroring how
    ``get_prompt()`` became the sole prompt authority.
    """
    if order is None:
        return None
    policy_cfg = _order_policy_cfg(order)
    if policy_cfg is None:
        return None
    mode = getattr(policy_cfg, "action_mode", None)
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    embodiment = getattr(policy_cfg, "embodiment", None)
    mode = getattr(embodiment, "action_mode", None) if embodiment is not None else None
    return mode.strip() if isinstance(mode, str) and mode.strip() else None


def clone_embodiment_for_action_mode(cfg: PolicyEmbodimentCFG, action_mode: str) -> PolicyEmbodimentCFG:
    """Return a copy of ``cfg`` with per-mode action and observation overrides."""
    mode = str(action_mode).strip()
    cloned = deepcopy(cfg)
    if not mode:
        return cloned

    layouts = (cloned.metadata or {}).get("action_mode_layouts") or {}
    entry = layouts.get(mode) or {}
    slot_layout = entry.get("slot_layout")
    if slot_layout:
        cloned.slot_layout = {key: dict(value) for key, value in slot_layout.items()}
    if entry.get("action_dim") is not None:
        cloned.action_dim = int(entry["action_dim"])
    action_field_map = entry.get("action_field_map")
    if action_field_map is not None:
        cloned.action_field_map = dict(action_field_map)
    obs_field_map = entry.get("obs_field_map")
    if obs_field_map is not None:
        cloned.obs_field_map = dict(obs_field_map)
    obs_dims = entry.get("obs_dims")
    if obs_dims is not None:
        cloned.obs_dims = {str(key): int(value) for key, value in obs_dims.items()}
    cloned.action_mode = mode
    return cloned


def apply_order_action_mode_to_policy_cfg(order: Any) -> str | None:
    """Apply the resolved action mode (policy-declared, else embodiment default) in place.

    The mode is resolved from the order's policy CFG (``ManaPolicyCFG.action_mode``
    first, then the injected ``PolicyEmbodimentCFG.action_mode`` default). When a
    mode resolves, the embodiment is cloned with that mode's wire-contract overrides.
    """
    mode = resolve_order_action_mode(order)
    policy_cfg = _order_policy_cfg(order)
    embodiment = getattr(policy_cfg, "embodiment", None) if policy_cfg is not None else None
    if mode and isinstance(embodiment, PolicyEmbodimentCFG):
        policy_cfg.embodiment = clone_embodiment_for_action_mode(embodiment, mode)
    return mode


def resolve_pink_ik_runtime(order: Any = None) -> dict[str, Any]:
    """Resolve the evaluation policy's IK runtime knobs."""
    runtime: dict[str, Any] = {}
    if order is None:
        return runtime
    policy_cfg = _order_policy_cfg(order)
    embodiment = getattr(policy_cfg, "embodiment", None) if policy_cfg is not None else None
    metadata = getattr(embodiment, "metadata", None) if embodiment is not None else None
    if isinstance(metadata, dict) and isinstance(metadata.get("pink_ik_runtime"), dict):
        runtime.update(metadata["pink_ik_runtime"])
    order_runtime = getattr(order, "pink_ik_runtime", None)
    if isinstance(order_runtime, dict):
        runtime.update(order_runtime)
    return runtime


def apply_embodiment_action_config(
    *,
    embodiment: Any,
    action_mode: str | None,
    pink_ik_runtime: dict[str, Any] | None = None,
    context: str = "Factory",
) -> None:
    """Switch the simulator embodiment action config for an action mode."""
    if embodiment is None or not action_mode:
        return

    if action_mode == "joint_position":
        _set_action_config(embodiment, "JointActionsCfg", action_mode, context)
        return
    if action_mode == "ik_eef_delta":
        _set_action_config(embodiment, "ActionsCfg", action_mode, context)
        return
    if action_mode == "ik_eef_abs":
        selector = getattr(embodiment, "use_abs_ik_action_config", None)
        if callable(selector):
            selector()
            manaprint(f"INFO: [{context}] selected ActionsCfgAbsIK with runtime X2Robot IK assets")
            return
        _set_action_config(embodiment, "ActionsCfgAbsIK", action_mode, context)
        return
    if action_mode == "pink_ik":
        _set_pink_ik_action_config(embodiment, pink_ik_runtime or {}, context)
        return
    if action_mode == "ik_eef_abs_wholebody":
        selector = getattr(embodiment, "use_wallx_wholebody_action_config", None)
        if callable(selector):
            selector()
            manaprint(f"INFO: [{context}] selected ActionsCfgWallxWholebody with runtime X2Robot IK assets")
            return
        _set_action_config(
            embodiment,
            "ActionsCfgWallxWholebody",
            action_mode,
            context,
        )
        return
    manaprint(
        f"WARNING: [{context}] Unsupported action_mode='{action_mode}', "
        "leaving existing embodiment.action_config unchanged"
    )


def _set_action_config(embodiment: Any, cfg_attr: str, action_mode: str, context: str) -> None:
    cfg_cls = getattr(embodiment, cfg_attr, None)
    if cfg_cls is None:
        raise RuntimeError(f"Embodiment {type(embodiment).__name__} has no {cfg_attr} for action_mode='{action_mode}'.")
    embodiment.action_config = cfg_cls()
    manaprint(f"INFO: [{context}] Applied {cfg_attr} for action_mode='{action_mode}'")


def _set_pink_ik_action_config(embodiment: Any, runtime: dict[str, Any], context: str) -> None:
    # EX001-6R: Pink was replaced by X2Robot IK. Prefer X2Robot when present.
    if hasattr(embodiment, "use_x2robot_ik_action_config"):
        embodiment.use_x2robot_ik_action_config(**(runtime or {}))
        manaprint(f"INFO: [{context}] Applied X2RobotIKActionsCfg (Pink replacement)")
        return

    pink_cfg = getattr(embodiment, "PinkIKActionsCfg", None)
    if pink_cfg is None:
        raise RuntimeError(
            f"Embodiment {type(embodiment).__name__} has neither X2RobotIKActionsCfg "
            f"nor PinkIKActionsCfg for action_mode='pink_ik'."
        )

    build = getattr(embodiment, "use_pink_ik_action_config", None)
    if callable(build):
        build(**runtime)
    else:
        embodiment.action_config = pink_cfg()
    manaprint(f"INFO: [{context}] Applied PinkIKActionsCfg")
