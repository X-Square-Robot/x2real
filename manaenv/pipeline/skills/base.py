"""Atomic skill abstractions for planner-driven manipulation."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch

from manaenv.utils.debug_print import manaprint
from manaenv.utils.scene_object_utils import get_rigid_object_env_pose, get_rigid_object_scale

MOTION_KIND_LINEAR_CARTESIAN = "linear_cartesian"
MOTION_KIND_ARC = "arc"


@dataclass
class SkillCfg:
    """Base configuration for an atomic skill instance."""

    class_type: type[Skill] | None = None
    skill_type: str = ""
    arm: str = "left"
    # Human-readable noun used to author language instructions (overrides the
    # asset-name heuristic in the staged instruction builder). E.g. a pick's
    # display_name is the item ("yellow block"); a place's is the destination
    # ("yellow paper").
    display_name: str = ""


@dataclass
class SkillTarget:
    """One sparse target point emitted by a skill before motion planning."""

    pos: torch.Tensor
    quat: torch.Tensor
    gripper: float
    phase: str
    arm: str = "left"
    joint_pos: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Atomic skill base.

    A skill observes the scene and emits sparse target points. It never runs IK,
    motion generation, or low-level control directly.
    """

    Cfg: type[SkillCfg] = SkillCfg

    def __init__(self, cfg: SkillCfg):
        self.cfg = cfg
        self._phase_report_state: dict[str, Any] = {"last_phase": None, "last_log_step": -(10**9)}

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ()

    def _resolve_arm(self, observation: dict[str, Any]) -> str:
        arm = str(getattr(self.cfg, "arm", "current") or "current").strip().lower()
        if arm in ("current", "auto"):
            return str(observation.get("active_arm") or "left").strip().lower()
        if arm == "both":
            return "both"
        return arm

    def _select_nearest_arm_to_object(self, observation: dict[str, Any], target_pos: torch.Tensor) -> str:
        return self._ranked_arms_to_object(observation, target_pos)[0]

    def _ranked_arms_to_object(self, observation: dict[str, Any], object_pos: torch.Tensor) -> list[str]:

        arm_cfg = str(getattr(self.cfg, "arm", "current") or "current").strip().lower()
        if arm_cfg in ("left", "right", "both"):
            return [arm_cfg]
        if arm_cfg not in ("current", "auto"):
            raise ValueError(f"Unsupported {getattr(self.cfg, 'skill_type', 'skill')!r} arm mode: {arm_cfg!r}")

        object_pos = torch.as_tensor(object_pos, dtype=torch.float32).detach().to(device="cpu").reshape(3)
        ee_positions = self._read_arm_ee_positions(observation)
        if not ee_positions:
            return [str(observation.get("active_arm") or "left").strip().lower()]

        ranked = sorted(
            ee_positions.items(),
            key=lambda item: float(torch.linalg.norm(item[1].to(dtype=torch.float32) - object_pos).item()),
        )
        return [arm for arm, _ in ranked]

    def _read_arm_ee_quat(self, observation: dict[str, Any], arm: str) -> torch.Tensor:
        """Read one arm's EE orientation (xyzw) from the live env when available."""

        arm_key = str(arm or "").strip().lower()
        gym_env = observation.get("gym_env")
        env_idx = int(observation.get("env_idx", 0))
        cache = observation.get("ee_index_cache")
        if not isinstance(cache, dict):
            cache = {}
        if gym_env is not None and arm_key in ("left", "right"):
            from manaenv.utils.robot_state import get_arm_ee_local_pose

            _, quat = get_arm_ee_local_pose(
                gym_env=gym_env,
                arm=arm_key,
                env_idx=env_idx,
                index_cache=cache,
            )
            if quat is not None and quat.numel() >= 4:
                return quat[:4].detach().to(dtype=torch.float32)

        active = str(observation.get("active_arm") or "").strip().lower()
        current_quat = observation.get("current_ee_quat")
        if arm_key == active and current_quat is not None:
            return torch.as_tensor(current_quat, dtype=torch.float32).reshape(4)
        raise ValueError(
            f"Could not resolve EE quaternion for arm={arm_key!r}; "
            "observation must include gym_env or matching current_ee_quat."
        )

    @staticmethod
    def _read_arm_ee_positions(observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Read both EE positions in env-local coordinates when a live env is available."""

        gym_env = observation.get("gym_env")
        env_idx = int(observation.get("env_idx", 0))
        cache = observation.get("ee_index_cache")
        if not isinstance(cache, dict):
            cache = {}

        positions: dict[str, torch.Tensor] = {}
        if gym_env is not None:
            from manaenv.utils.robot_state import get_arm_ee_local_pose

            for arm in ("left", "right"):
                pos, _ = get_arm_ee_local_pose(
                    gym_env=gym_env,
                    arm=arm,
                    env_idx=env_idx,
                    index_cache=cache,
                )
                if pos is not None and pos.numel() >= 3:
                    positions[arm] = pos[:3].detach().to(dtype=torch.float32, device="cpu")

        if not positions:
            active = str(observation.get("active_arm") or "").strip().lower()
            current = observation.get("current_ee_pos")
            if active in ("left", "right") and current is not None:
                positions[active] = torch.as_tensor(current, dtype=torch.float32).detach().to(device="cpu").reshape(3)
        return positions

    @staticmethod
    def _read_selected_arm_ee_pose(
        observation: dict[str, Any],
        arm: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the live EE pose for the arm selected by a skill."""

        current_pos = torch.as_tensor(observation["current_ee_pos"], dtype=torch.float32, device=device).reshape(3)
        current_quat = torch.as_tensor(observation["current_ee_quat"], dtype=torch.float32, device=device).reshape(4)
        arm_key = str(arm or "").strip().lower()
        if arm_key not in ("left", "right"):
            return current_pos, current_quat

        gym_env = observation.get("gym_env")
        if gym_env is None:
            return current_pos, current_quat

        from manaenv.utils.robot_state import get_arm_ee_local_pose

        env_idx = int(observation.get("env_idx", 0))
        cache = observation.get("ee_index_cache")
        if not isinstance(cache, dict):
            cache = {}
        live_pos, live_quat = get_arm_ee_local_pose(
            gym_env=gym_env,
            arm=arm_key,
            env_idx=env_idx,
            index_cache=cache,
        )
        if live_pos is None or live_quat is None:
            return current_pos, current_quat
        return (
            live_pos[:3].detach().to(device=device, dtype=torch.float32),
            live_quat[:4].detach().to(device=device, dtype=torch.float32),
        )

    @abstractmethod
    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        """Collect skill-specific scene state from the runtime environment."""

    @abstractmethod
    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        """Return ordered sparse targets for the current atomic skill."""

    @staticmethod
    def read_object_pose_and_scale(
        env: Any,
        object_name: str,
        env_idx: int,
        env_origin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float]]:
        pose = get_rigid_object_env_pose(env, object_name, env_idx, env_origin=env_origin)
        if pose is None:
            raise RuntimeError(
                f"Skill '{getattr(env, 'task_name', '?')}' could not read pose for object '{object_name}'."
            )
        pos_env, quat = pose
        scale = get_rigid_object_scale(env, object_name)
        return pos_env, quat, scale

    @staticmethod
    def scale_object_local_points(
        local_pos: torch.Tensor,
        obj_scale: tuple[float, float, float],
    ) -> torch.Tensor:

        sx, sy, sz = (float(obj_scale[0]), float(obj_scale[1]), float(obj_scale[2]))
        if not (abs(sx - sy) < 1e-6 and abs(sy - sz) < 1e-6):
            warnings.warn(
                f"Anisotropic object scale {obj_scale} applied to mesh-local points; "
                "rotation fields of mesh-local targets are not corrected.",
                stacklevel=2,
            )
        scale_tensor = torch.as_tensor((sx, sy, sz), dtype=local_pos.dtype, device=local_pos.device)
        return local_pos * scale_tensor

    def report_phase(
        self,
        env: Any,
        env_idx: int,
        phase: str,
        phase_step: int,
        *,
        namespace: str | None = None,
        extras: dict[str, Any] | None = None,
        log_interval: int = 30,
    ) -> None:
        if env is None:
            return
        ns = (namespace or getattr(self.cfg, "skill_type", "skill") or "skill").strip().lower() or "skill"

        extras_dict = getattr(env, "extras", None)
        if isinstance(extras_dict, dict):
            num_envs = int(getattr(env, "num_envs", 1))
            phase_key = f"{ns}_phase"
            step_key = f"{ns}_phase_step"
            phase_arr = extras_dict.get(phase_key)
            if not isinstance(phase_arr, list) or len(phase_arr) != num_envs:
                phase_arr = ["idle"] * num_envs
                extras_dict[phase_key] = phase_arr
            step_arr = extras_dict.get(step_key)
            if not isinstance(step_arr, list) or len(step_arr) != num_envs:
                step_arr = [0] * num_envs
                extras_dict[step_key] = step_arr
            if 0 <= int(env_idx) < num_envs:
                phase_arr[int(env_idx)] = str(phase)
                step_arr[int(env_idx)] = int(phase_step)

        cur_phase = str(phase)
        last_phase = self._phase_report_state.get("last_phase")
        last_log_step = int(self._phase_report_state.get("last_log_step", -(10**9)))
        edge = last_phase != cur_phase
        heartbeat = log_interval > 0 and (int(phase_step) - last_log_step) >= int(log_interval)
        if log_interval > 0 and (edge or heartbeat):
            extras_str = ""
            if extras:
                extras_str = " " + " ".join(f"{k}={v}" for k, v in extras.items())
            tag = "EDGE" if edge else "HB"
            manaprint(f"[{ns}_phase env={int(env_idx)} {tag}] phase={cur_phase} step={int(phase_step)}{extras_str}")
            self._phase_report_state["last_phase"] = cur_phase
            self._phase_report_state["last_log_step"] = int(phase_step)

    def expand_to_waypoints(
        self,
        targets: list[SkillTarget],
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        settings: Any,
    ) -> list[dict[str, Any]]:

        if not targets:
            return []

        arms = {str(target.arm) for target in targets}
        if len(arms) != 1:
            raise NotImplementedError("Default expand_to_waypoints only supports single-arm target sequences.")

        normalized = [self._normalize_target(target) for target in targets]
        phases = [target.phase for target in normalized]
        if phases == ["pre_grasp", "grasp", "lift"] and len(normalized) == 3:
            pre_target, grasp_target, lift_target = normalized
            close_gripper = float(lift_target.gripper)
            return [
                {
                    "pos": current_pos.detach().clone().to(dtype=torch.float32),
                    "quat": current_quat.detach().clone().to(dtype=torch.float32),
                    "gripper": float(pre_target.gripper),
                    "phase": "start",
                },
                self._target_to_waypoint(pre_target, phase="approach"),
                self._target_to_waypoint(grasp_target, phase="descend"),
                {
                    "pos": grasp_target.pos,
                    "quat": grasp_target.quat,
                    "gripper": close_gripper,
                    "phase": "close",
                },
                self._target_to_waypoint(lift_target, phase="lift"),
            ]

        return [
            {
                "pos": current_pos.detach().clone().to(dtype=torch.float32),
                "quat": current_quat.detach().clone().to(dtype=torch.float32),
                "gripper": float(normalized[0].gripper),
                "phase": "start",
            },
            *[self._target_to_waypoint(target, phase=target.phase) for target in normalized],
        ]

    @staticmethod
    def _normalize_target(target: SkillTarget) -> SkillTarget:
        return SkillTarget(
            pos=torch.as_tensor(target.pos, dtype=torch.float32).reshape(3),
            quat=torch.as_tensor(target.quat, dtype=torch.float32).reshape(4),
            gripper=float(target.gripper),
            phase=str(target.phase),
            arm=str(target.arm),
            joint_pos=(
                torch.as_tensor(target.joint_pos, dtype=torch.float32).reshape(-1)
                if target.joint_pos is not None
                else None
            ),
            extras=dict(target.extras),
        )

    @staticmethod
    def _target_to_waypoint(target: SkillTarget, *, phase: str) -> dict[str, Any]:
        waypoint = {
            "pos": target.pos,
            "quat": target.quat,
            "gripper": float(target.gripper),
            "phase": phase,
        }
        if target.joint_pos is not None:
            waypoint["joint_pos"] = target.joint_pos
        waypoint.update(target.extras)
        return waypoint

    @staticmethod
    def build_return_home_target(
        start_pose: Any,
        *,
        arm: str,
        gripper: float,
        time_dilation_factor: float | None,
        device: torch.device | None = None,
        dual_ee_abs: bool = False,
    ) -> SkillTarget:
        """Build a ``return_home`` target restoring one arm to its episode-start pose."""
        pos, quat = start_pose
        pos_t = torch.as_tensor(pos, dtype=torch.float32)
        quat_t = torch.as_tensor(quat, dtype=torch.float32)
        if device is not None:
            pos_t = pos_t.to(device=device)
            quat_t = quat_t.to(device=device)
        extras: dict[str, Any] = {}
        if dual_ee_abs:
            extras["dual_ee_abs"] = True
        if time_dilation_factor is not None:
            extras["time_dilation_factor"] = float(time_dilation_factor)
        return SkillTarget(
            pos=pos_t.reshape(3),
            quat=quat_t.reshape(4),
            gripper=float(gripper),
            phase="return_home",
            arm=arm,
            extras=extras,
        )


class BimanualSkill(ABC):

    @abstractmethod
    def plan_targets_per_arm(
        self,
        observation: dict[str, Any],
        evaluator: Any | None,
    ) -> dict[str, list[SkillTarget]]:
        raise NotImplementedError(
            "BimanualSkill is a forward-compatibility placeholder; no concrete implementation yet."
        )


_SKILL_REGISTRY: dict[str, type[Skill]] = {}


def register_skill(name: str, cls: type[Skill]) -> None:
    """Register a skill implementation under a stable string alias."""

    alias = str(name).strip().lower()
    if not alias:
        raise ValueError("Skill registration name cannot be empty.")
    _SKILL_REGISTRY[alias] = cls


def get_skill_class(name: str) -> type[Skill]:
    """Resolve a registered skill class."""

    alias = str(name).strip().lower()
    cls = _SKILL_REGISTRY.get(alias)
    if cls is None:
        available = ", ".join(sorted(_SKILL_REGISTRY))
        raise KeyError(f"Unknown skill '{name}'. Available: {available}")
    return cls


def create_skill(name: str, params: dict[str, Any] | None = None) -> Skill:
    """Instantiate a registered skill from raw config parameters."""

    cls = get_skill_class(name)
    cfg_cls = getattr(cls, "Cfg", SkillCfg)
    raw = dict(params or {})
    raw.setdefault("skill_type", str(name).strip().lower())
    raw.setdefault("class_type", cls)
    try:
        cfg = cfg_cls(**raw)
    except TypeError:
        fields = getattr(cfg_cls, "__dataclass_fields__", {})
        filtered = {key: value for key, value in raw.items() if key in fields}
        cfg = cfg_cls(**filtered)
    if getattr(cfg, "class_type", None) is None:
        cfg.class_type = cls
    if not getattr(cfg, "skill_type", ""):
        cfg.skill_type = str(name).strip().lower()
    return cls(cfg)


__all__ = [
    "BimanualSkill",
    "Skill",
    "SkillCfg",
    "SkillTarget",
    "create_skill",
    "get_skill_class",
    "register_skill",
]
