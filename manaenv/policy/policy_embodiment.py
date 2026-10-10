"""Built-in policy-to-robot contracts for public evaluation.

The open distribution keeps one suite YAML, so policy wire layouts are derived
from the runtime embodiment class instead of loading auxiliary YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import yaml

if TYPE_CHECKING:
    from manaenv.policy.mana_policy import ManaPolicyCFG


@dataclass
class PolicyEmbodimentCFG:
    """Wire and action layout used to bind one policy to one robot."""

    name: str = ""
    action_mode: str = ""
    action_dim: int = 0
    action_field_map: Dict[str, str] = field(default_factory=dict)
    slot_layout: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    obs_field_map: Dict[str, str] = field(default_factory=dict)
    obs_dims: Dict[str, int] = field(default_factory=dict)
    state_dtype: str = "float32"
    image_encode: str = "jpeg_b64"
    image_jpeg_quality: int = 95
    instruction_key: str = "instruction"
    ee_frames: Dict[str, str] = field(default_factory=dict)
    wire_pose_reference: str = "init_ee_pose"
    metadata: Dict[str, Any] = field(default_factory=dict)


_DUAL_EE_EMBODIMENTS = {"EX0016REmbodiment", "ARXR5Embodiment"}
_SINGLE_JOINT_EMBODIMENTS = {"DroidEmbodiment", "FrankaEmbodiment"}
_EMBODIMENTS_DIR = Path(__file__).resolve().parents[1] / "defination" / "embodiments"


def _declared_contract(implementation: str, embodiment: str) -> PolicyEmbodimentCFG | None:
    """Load an explicitly published policy/embodiment contract when present."""
    path = _EMBODIMENTS_DIR / implementation / f"{embodiment}.yaml"
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Policy embodiment config must be a mapping: {path}")
    return PolicyEmbodimentCFG(
        name=str(data.get("name", path.stem)),
        action_mode=str(data.get("action_mode", "")),
        action_dim=int(data.get("action_dim", 0)),
        action_field_map=dict(data.get("action_field_map") or {}),
        slot_layout={str(key): dict(value) for key, value in (data.get("slot_layout") or {}).items()},
        obs_field_map=dict(data.get("obs_field_map") or {}),
        obs_dims={str(key): int(value) for key, value in (data.get("obs_dims") or {}).items()},
        state_dtype=str(data.get("state_dtype", "float32")),
        image_encode=str(data.get("image_encode", "jpeg_b64")),
        image_jpeg_quality=int(data.get("image_jpeg_quality", 95)),
        instruction_key=str(data.get("instruction_key", "instruction")),
        ee_frames=dict(data.get("ee_frames") or {}),
        wire_pose_reference=str(data.get("wire_pose_reference", "init_ee_pose")),
        metadata=dict(data.get("metadata") or {}),
    )


def _dual_ee_contract(name: str, *, policy_space: bool) -> PolicyEmbodimentCFG:
    metadata: Dict[str, Any] = {}
    if policy_space:
        metadata = {
            "gripper_wire_indices": [6, 13],
            "gripper_state_indices": {"follow1_pos": 6, "follow2_pos": 6},
            "action_mode_layouts": {
                "joint_position": {
                    "action_dim": 26,
                    "action_field_map": {"action[0:14]": "joint_action"},
                    "slot_layout": {"action": {"joint_pos": [0, 14]}},
                    "obs_field_map": {
                        "follow1_pos": "follow_left_joint_pos",
                        "follow2_pos": "follow_right_joint_pos",
                        "camera_front": "head_cam",
                        "camera_left": "left_wrist_cam",
                        "camera_right": "right_wrist_cam",
                    },
                    "obs_dims": {"follow1_pos": 7, "follow2_pos": 7},
                }
            },
        }
    return PolicyEmbodimentCFG(
        name=name,
        action_mode="ik_eef_abs",
        action_dim=16,
        action_field_map={
            "follow1_pos[0:7]": "arm_action",
            "follow1_pos[7:8]": "gripper_action",
            "follow2_pos[0:7]": "right_arm_action",
            "follow2_pos[7:8]": "right_gripper_action",
        },
        slot_layout={
            "follow1_pos": {"pos": [0, 3], "quat": [3, 7], "gripper": [7]},
            "follow2_pos": {"pos": [8, 11], "quat": [11, 15], "gripper": [15]},
        },
        obs_field_map={
            "follow1_pos": "follow1_pos",
            "follow2_pos": "follow2_pos",
            "camera_front": "head_cam",
            "camera_left": "left_wrist_cam",
            "camera_right": "right_wrist_cam",
        },
        obs_dims={"follow1_pos": 7, "follow2_pos": 7},
        image_encode="raw" if policy_space else "jpeg_b64",
        ee_frames={"left": "left_ee_frame", "right": "right_ee_frame"},
        metadata=metadata,
    )


def _single_joint_contract(name: str) -> PolicyEmbodimentCFG:
    return PolicyEmbodimentCFG(
        name=name,
        action_mode="joint_position",
        action_dim=8,
        obs_field_map={
            "joint_position": "arm_joint_pos",
            "gripper_position": "gripper_pos",
            "wrist": "wrist_cam_rgb",
            "exterior_1": "exterior_1_cam_rgb",
            "exterior_2": "exterior_2_cam_rgb",
        },
        obs_dims={"joint_position": 7, "gripper_position": 1},
        image_encode="raw",
    )


def load_policy_embodiment_cfg(
    *, implementation_yaml_name: str, embodiment_class_name: str
) -> PolicyEmbodimentCFG:
    """Build the public wire contract for a supported runtime embodiment."""

    implementation = str(implementation_yaml_name).strip()
    embodiment = str(embodiment_class_name).strip()
    declared = _declared_contract(implementation, embodiment)
    if declared is not None:
        return declared
    if embodiment in _DUAL_EE_EMBODIMENTS and implementation in {"wallx", "policy_space"}:
        return _dual_ee_contract(embodiment, policy_space=implementation == "policy_space")
    if embodiment in _SINGLE_JOINT_EMBODIMENTS and implementation == "policy_space":
        return _single_joint_contract(embodiment)
    raise ValueError(
        "Unsupported public policy/embodiment pair: "
        f"implementation={implementation!r}, embodiment={embodiment!r}."
    )


def bind_embodiment_to_policy_cfg(
    *, environment: Any, policy_cfg: Optional["ManaPolicyCFG"]
) -> None:
    """Inject a built-in policy embodiment contract into an evaluation CFG."""

    if policy_cfg is None:
        return
    from manaenv.policy.mana_policy import ManaPolicyCFG as _ManaPolicyCFG

    if not isinstance(policy_cfg, _ManaPolicyCFG) or policy_cfg.embodiment is not None:
        return
    implementation_cfg = policy_cfg.implementation
    if implementation_cfg is None:
        raise RuntimeError("ManaPolicyCFG.implementation is required for public evaluation.")
    implementation = getattr(type(implementation_cfg), "YAML_DISPATCH_KEY", "")
    if not implementation:
        raise RuntimeError(
            f"Policy implementation {type(implementation_cfg).__name__} has no public dispatch key."
        )
    runtime_embodiment = getattr(environment, "embodiment", None)
    if runtime_embodiment is None:
        raise RuntimeError("Evaluation environment has no runtime embodiment.")
    embodiment_cfg = load_policy_embodiment_cfg(
        implementation_yaml_name=implementation,
        embodiment_class_name=type(runtime_embodiment).__name__,
    )
    _inject_gripper_limits(embodiment_cfg, runtime_embodiment)
    policy_cfg.embodiment = embodiment_cfg


def _inject_gripper_limits(cfg: PolicyEmbodimentCFG, embodiment: Any) -> None:
    from manaenv.policy.gripper_convention import LIMITS_KEY, limits_to_metadata, read_gripper_limits

    limits = read_gripper_limits(embodiment)
    if limits is None:
        return
    cfg.metadata = dict(cfg.metadata)
    cfg.metadata[LIMITS_KEY] = limits_to_metadata(limits)


__all__ = ["PolicyEmbodimentCFG", "bind_embodiment_to_policy_cfg", "load_policy_embodiment_cfg"]
