"""``twist`` skill: grip a knob and rotate it about its own axis (revolute).

cuRobo plans **approach + grasp + close** only. Post-grasp motion is joint-space:

- **twist**: last arm joint only (knob rotation).
- **retreat**: non-last joints → saved pre_grasp snapshot; last joint unchanged; gripper open.
- **flip**: last joint ±180° (limit-aware).
- **regrasp**: non-last joints → saved grasp snapshot; last joint unchanged; gripper close.
- Repeat until the sampled temperature target is reached, then **release**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.pipeline.skills.twist_runtime.twist_goal import read_pinned_twist_goal
from manaenv.utils.debug_print import manaprint
from manaenv.utils.scene_object_utils import get_articulation_joint_value
from manaenv.utils.tensor_utils import as_torch

from .articulation_base import ArticulationGripSkill, ArticulationGripSkillCfg
from .base import SkillTarget, register_skill
from .twist_runtime.wrist_limit import (
    JointLimitSpec,
    flip_joint_deg,
    joint_deg_to_rad,
    joint_rad_to_deg,
    max_travel_deg,
    near_limit_for_direction,
    resolve_joint_limit_spec,
)

_TWIST_JOINT_THRESHOLD = 0.03


@dataclass
class TwistSkillCfg(ArticulationGripSkillCfg):
    skill_type: str = "twist"
    joint_type: str = "revolute"
    # Embodiment-owned last-joint travel envelope (degrees). Unset = single arc.
    last_joint_limit_min_deg: float | None = None
    last_joint_limit_max_deg: float | None = None
    limit_margin_deg: float = 10.0
    wrist_flip_deg: float = 180.0
    # Coupling from knob-joint delta sign to last-joint command sign.
    # EX001: -1 — temp↑ (knob joint more negative) ⇒ last joint +Δ.
    last_joint_twist_sign: float = -1.0
    twist_joint_steps: int = 40
    retreat_joint_steps: int = 25
    flip_joint_steps: int = 30
    regrasp_joint_steps: int = 25
    release_joint_steps: int = 15
    max_limit_flip_cycles: int = 8
    knob_reached_tolerance_rad: float = 0.052
    twist_joint_position_threshold: float = _TWIST_JOINT_THRESHOLD
    gripper_open_fraction: float = 0.25
    arm_joint_prefix: str | None = None


class TwistSkill(ArticulationGripSkill):
    """Grip a knob and twist it about its own axis (in-place rotation)."""

    Cfg = TwistSkillCfg

    def __init__(self, cfg: TwistSkillCfg):
        super().__init__(cfg)
        self._dynamic_arc_sweep: float | None = None
        self._limit_spec: JointLimitSpec | None = resolve_joint_limit_spec(cfg)
        self._twist_phase: str = "approach"
        self._saved_arm: str = ""
        self._pre_grasp_joint_q: torch.Tensor | None = None
        self._grasp_joint_q: torch.Tensor | None = None
        self._flip_cycles: int = 0
        self._goal_logged: bool = False

    def reset_episode(self) -> None:
        """Reset per-episode twist phase state (called from planner ``reset``)."""
        self._dynamic_arc_sweep = None
        self._twist_phase = "approach"
        self._saved_arm = ""
        self._pre_grasp_joint_q = None
        self._grasp_joint_q = None
        self._flip_cycles = 0
        self._goal_logged = False

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("turnable", "object")

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        observation = super().observe(env, obs, int(env_idx))
        self._limit_spec = resolve_joint_limit_spec(self.cfg)

        goal = read_pinned_twist_goal(env, int(env_idx))
        if goal is None:
            return observation

        articulation_name = self.cfg.articulation_name or self.cfg.object_name
        current = get_articulation_joint_value(
            env,
            articulation_name,
            int(env_idx),
            joint_name=goal.joint_name,
        )
        if current is None:
            return observation

        from manaenv.assets.objects.object_library import KnobA

        target_joint = KnobA.display_value_to_joint_angle(goal.target_temperature)
        full_sweep = float(target_joint - float(current))
        self._dynamic_arc_sweep = full_sweep
        if not self._goal_logged:
            mode = "limit_aware" if self._limit_spec is not None else "joint_only"
            manaprint(
                f"[twist] env={int(env_idx)} initial={goal.initial_temperature} "
                f"target={goal.target_temperature} "
                f"knob_delta_deg={math.degrees(full_sweep):+.1f} mode={mode}"
            )
            self._goal_logged = True
        observation["twist_arc_sweep_rad"] = float(full_sweep)
        observation["twist_limit_aware"] = self._limit_spec is not None
        observation["gym_env"] = env
        observation["env_idx"] = int(env_idx)
        return observation

    def requests_segment_replan(self, observation: dict[str, Any]) -> bool:
        """Advance the post-grasp phase machine; return True to replan in-skill."""
        if self._twist_phase == "done":
            return False

        if self._twist_phase == "approach":
            self._saved_arm = self._normalize_arm(self._saved_arm, observation)
            if self._grasp_joint_q is None:
                self._grasp_joint_q = self._read_arm_joint_q(observation)
            self._twist_phase = "twist"
            manaprint(f"[twist] grasp done arm={self._saved_arm!r}, start joint twist")
            return True

        if self._twist_phase == "twist":
            if self._knob_at_target(observation):
                self._twist_phase = "release"
                manaprint("[twist] knob at target, releasing")
                return True
            remaining = self._knob_remaining_rad(observation)
            if remaining is not None and abs(remaining) <= float(self.cfg.knob_reached_tolerance_rad):
                self._twist_phase = "release"
                return True
            direction = self._last_joint_twist_direction(observation)
            last_deg = self._last_joint_deg(observation)
            if (
                self._limit_spec is not None
                and direction != 0
                and last_deg is not None
                and near_limit_for_direction(last_deg, direction, self._limit_spec)
            ):
                self._flip_cycles += 1
                if self._flip_cycles > int(self.cfg.max_limit_flip_cycles):
                    manaprint("[twist] max flip cycles exceeded, releasing")
                    self._twist_phase = "release"
                    return True
                self._twist_phase = "retreat"
                manaprint(
                    f"[twist] last joint near limit ({last_deg:+.1f} deg), " f"retreat+flip cycle={self._flip_cycles}"
                )
                return True
            if remaining is not None and abs(remaining) > float(self.cfg.knob_reached_tolerance_rad):
                self._twist_phase = "twist"
                return True
            self._twist_phase = "release"
            return True

        if self._twist_phase == "retreat":
            self._twist_phase = "flip"
            return True
        if self._twist_phase == "flip":
            self._twist_phase = "regrasp"
            manaprint("[twist] wrist flipped, descend non-last joints to grasp + close")
            return True
        if self._twist_phase == "regrasp":
            self._twist_phase = "twist"
            manaprint(f"[twist] re-grasp done arm={self._saved_arm!r}, resume joint twist")
            return True
        if self._twist_phase == "release":
            self._twist_phase = "done"
            return False

        return False

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        if self._twist_phase == "done":
            return []

        if self._twist_phase == "approach":
            return self._plan_approach_grasp(observation, evaluator)
        if self._twist_phase == "twist":
            return self._plan_joint_twist_segment(observation)
        if self._twist_phase == "retreat":
            return self._plan_joint_retreat(observation)
        if self._twist_phase == "flip":
            return self._plan_joint_flip(observation)
        if self._twist_phase == "regrasp":
            return self._plan_joint_regrasp(observation)
        if self._twist_phase == "release":
            return self._plan_joint_release(observation)
        return []

    def expand_to_waypoints(
        self,
        targets: list[SkillTarget],
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        settings: Any,
    ) -> list[dict[str, Any]]:
        del settings
        if not targets:
            return []
        if self._targets_are_joint_only(targets):
            return [self._target_to_waypoint(self._normalize_target(t), phase=t.phase) for t in targets]
        return self._expand_approach_grasp(targets, current_pos, current_quat)

    @staticmethod
    def _targets_are_joint_only(targets: list[SkillTarget]) -> bool:
        for target in targets:
            if target.joint_pos is None:
                return False
        return bool(targets)

    def _expand_approach_grasp(
        self,
        targets: list[SkillTarget],
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
    ) -> list[dict[str, Any]]:
        normalized = [self._normalize_target(t) for t in targets]
        if len(normalized) < 2:
            pre_t = normalized[0]
            return [
                {
                    "pos": current_pos.detach().clone().to(dtype=torch.float32),
                    "quat": current_quat.detach().clone().to(dtype=torch.float32),
                    "gripper": self._gripper_open_cmd(),
                    "phase": "start",
                },
                self._target_to_waypoint(pre_t, phase="approach"),
            ]
        pre_t, grasp_t = normalized[0], normalized[1]
        gripper_open = self._gripper_open_cmd()
        gripper_close = float(self.cfg.gripper_close)
        return [
            {
                "pos": current_pos.detach().clone().to(dtype=torch.float32),
                "quat": current_quat.detach().clone().to(dtype=torch.float32),
                "gripper": gripper_open,
                "phase": "start",
            },
            self._target_to_waypoint(pre_t, phase="approach"),
            self._target_to_waypoint(grasp_t, phase="descend"),
            {"pos": grasp_t.pos, "quat": grasp_t.quat, "gripper": gripper_close, "phase": "close"},
        ]

    def _normalize_arm(self, arm: Any, observation: dict[str, Any]) -> str:
        raw = arm or observation.get("active_arm") or self.cfg.arm or "left"
        name = str(raw).strip().lower()
        if name in ("right",):
            return "right"
        return "left"

    def _infer_selected_arm(self, observation: dict[str, Any], grasp_targets: list[SkillTarget]) -> str:
        for target in grasp_targets:
            arm = str(getattr(target, "arm", None) or "").strip().lower()
            if arm in ("left", "right"):
                return arm
        obj_pose = observation.get("object_pose_env")
        if obj_pose is not None:
            obj_pos, _ = obj_pose
            return self._normalize_arm(
                self._select_nearest_arm_to_object(observation, obj_pos),
                observation,
            )
        return self._normalize_arm(None, observation)

    def _gripper_open_cmd(self) -> float:
        """Partial open (fraction of embodiment-resolved full open)."""
        open_val = float(self.cfg.gripper_open)
        frac = float(getattr(self.cfg, "gripper_open_fraction", 1.0) or 1.0)
        if 0.0 < frac < 1.0:
            return open_val * frac
        return open_val

    def _joint_threshold_for_phase(self, phase: str) -> float:
        del phase
        return float(self.cfg.twist_joint_position_threshold)

    def _begin_limit_retreat_cycle(
        self,
        observation: dict[str, Any],
        *,
        last_deg: float,
        direction: int,
        reason: str,
    ) -> list[SkillTarget]:
        self._flip_cycles += 1
        if self._flip_cycles > int(self.cfg.max_limit_flip_cycles):
            manaprint("[twist] max flip cycles exceeded at limit boundary, releasing")
            self._twist_phase = "release"
            return self._plan_joint_release(observation)
        manaprint(f"[twist] {reason} ({last_deg:+.1f} deg dir={direction}), " f"retreat+flip cycle={self._flip_cycles}")
        self._twist_phase = "retreat"
        return self._plan_joint_retreat(observation)

    def on_waypoint_reached(
        self,
        phase: str,
        observation: dict[str, Any],
        *,
        next_phase: str = "",
    ) -> None:
        """Capture joint snapshots at approach/grasp boundaries during cuRobo approach."""
        if self._twist_phase != "approach":
            return
        joint_q = self._read_arm_joint_q(observation)
        if joint_q is None:
            return
        if phase == "approach" and next_phase == "descend":
            self._pre_grasp_joint_q = joint_q.detach().clone()
            manaprint(
                f"[twist] saved pre_grasp joints " f"(last={joint_rad_to_deg(float(joint_q[-1].item())):+.1f} deg)"
            )
        elif phase == "close":
            self._grasp_joint_q = joint_q.detach().clone()

    def _plan_approach_grasp(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        if self._pre_grasp_joint_q is None:
            self._pre_grasp_joint_q = self._read_arm_joint_q(observation)
        targets = super().plan_targets(observation, evaluator)
        if not targets:
            return []
        grasp_targets = [t for t in targets if t.phase in ("pre_grasp", "grasp")]
        self._saved_arm = self._infer_selected_arm(observation, grasp_targets)
        if self._pre_grasp_joint_q is None:
            self._pre_grasp_joint_q = self._read_arm_joint_q(observation)
        return grasp_targets

    def _plan_joint_twist_segment(self, observation: dict[str, Any]) -> list[SkillTarget]:
        current_q = self._read_arm_joint_q(observation)
        if current_q is None:
            return []
        last_idx = current_q.numel() - 1
        direction = self._last_joint_twist_direction(observation)
        if direction == 0:
            self._twist_phase = "release"
            return self._plan_joint_release(observation)

        last_deg = joint_rad_to_deg(float(current_q[last_idx].item()))
        remaining = self._knob_remaining_rad(observation)
        knob_needed_deg = abs(math.degrees(remaining or 0.0))
        if self._limit_spec is not None:
            room_deg = max_travel_deg(last_deg, direction, self._limit_spec)
            margin_deg = float(self._limit_spec.margin_deg)
            # Within the margin band (e.g. 10 deg left): stop twisting and flip now.
            if room_deg <= margin_deg or near_limit_for_direction(last_deg, direction, self._limit_spec):
                return self._begin_limit_retreat_cycle(
                    observation,
                    last_deg=last_deg,
                    direction=direction,
                    reason=f"last-joint within {margin_deg:.0f} deg of soft limit",
                )
            use_deg = min(room_deg, knob_needed_deg)
        else:
            use_deg = knob_needed_deg
            if use_deg < 0.5:
                self._twist_phase = "release"
                return self._plan_joint_release(observation)

        end_q = current_q.clone()
        end_q[last_idx] = current_q[last_idx] + joint_deg_to_rad(direction * use_deg)
        manaprint(
            f"[twist] joint twist segment last_joint {last_deg:+.1f} -> "
            f"{joint_rad_to_deg(float(end_q[last_idx].item())):+.1f} deg "
            f"(delta={use_deg:.1f}, knob_remaining_deg={knob_needed_deg:+.1f}, dir={direction:+d})"
        )
        return self._joint_interp_targets(
            observation,
            start_q=current_q,
            end_q=end_q,
            steps=int(self.cfg.twist_joint_steps),
            phase="twist_slide",
            gripper=float(self.cfg.gripper_close),
        )

    def _plan_joint_retreat(self, observation: dict[str, Any]) -> list[SkillTarget]:
        current_q = self._read_arm_joint_q(observation)
        pre_q = self._pre_grasp_joint_q
        if pre_q is None:
            pre_q = self._read_arm_joint_q(observation)
        if current_q is None or pre_q is None:
            manaprint("WARNING: [twist] retreat skipped: missing joint snapshot")
            self._twist_phase = "flip"
            return self._plan_joint_flip(observation)
        end_q = self._joint_q_preserve_last(pre_q, current_q)
        last_deg = joint_rad_to_deg(float(current_q[-1].item()))
        manaprint(
            f"[twist] retreat non-last joints to pre_grasp " f"(last joint stays {last_deg:+.1f} deg, gripper open)"
        )
        return self._joint_interp_targets(
            observation,
            start_q=current_q,
            end_q=end_q,
            steps=int(self.cfg.retreat_joint_steps),
            phase="twist_retreat",
            gripper=self._gripper_open_cmd(),
            gripper_start=float(self.cfg.gripper_close),
        )

    def _plan_joint_flip(self, observation: dict[str, Any]) -> list[SkillTarget]:
        if self._limit_spec is None:
            self._twist_phase = "twist"
            return self._plan_joint_twist_segment(observation)
        current_q = self._read_arm_joint_q(observation)
        if current_q is None:
            return []
        last_idx = current_q.numel() - 1
        last_deg = joint_rad_to_deg(float(current_q[last_idx].item()))
        target_deg = flip_joint_deg(last_deg, self._limit_spec)
        end_q = current_q.clone()
        end_q[last_idx] = joint_deg_to_rad(target_deg)
        manaprint(f"[twist] flip last joint {last_deg:+.1f} -> {target_deg:+.1f} deg")
        return self._joint_interp_targets(
            observation,
            start_q=current_q,
            end_q=end_q,
            steps=int(self.cfg.flip_joint_steps),
            phase="twist_flip",
            gripper=self._gripper_open_cmd(),
        )

    def _plan_joint_regrasp(self, observation: dict[str, Any]) -> list[SkillTarget]:
        current_q = self._read_arm_joint_q(observation)
        grasp_q = self._grasp_joint_q
        if current_q is None or grasp_q is None:
            manaprint("WARNING: [twist] regrasp skipped: missing grasp joint snapshot")
            self._twist_phase = "twist"
            return self._plan_joint_twist_segment(observation)
        end_q = self._joint_q_preserve_last(grasp_q, current_q)
        last_deg = joint_rad_to_deg(float(current_q[-1].item()))
        manaprint(f"[twist] descend non-last joints to grasp " f"(last joint stays {last_deg:+.1f} deg, gripper close)")
        return self._joint_interp_targets(
            observation,
            start_q=current_q,
            end_q=end_q,
            steps=int(self.cfg.regrasp_joint_steps),
            phase="twist_regrasp",
            gripper=float(self.cfg.gripper_close),
            gripper_start=self._gripper_open_cmd(),
        )

    def _plan_joint_release(self, observation: dict[str, Any]) -> list[SkillTarget]:
        current_q = self._read_arm_joint_q(observation)
        end_q = self._pre_grasp_joint_q
        if current_q is None:
            return []
        if end_q is None:
            end_q = current_q
        return self._joint_interp_targets(
            observation,
            start_q=current_q,
            end_q=end_q,
            steps=int(self.cfg.release_joint_steps),
            phase="twist_release",
            gripper=self._gripper_open_cmd(),
            gripper_start=float(self.cfg.gripper_close),
        )

    @staticmethod
    def _joint_q_preserve_last(target_q: torch.Tensor, current_q: torch.Tensor) -> torch.Tensor:
        """Blend to ``target_q`` but keep the last joint at its current value."""
        end_q = target_q.detach().clone().to(dtype=torch.float32)
        end_q[-1] = current_q[-1]
        return end_q

    def _joint_interp_targets(
        self,
        observation: dict[str, Any],
        *,
        start_q: torch.Tensor,
        end_q: torch.Tensor,
        steps: int,
        phase: str,
        gripper: float,
        gripper_start: float | None = None,
    ) -> list[SkillTarget]:
        arm = self._resolve_arm(observation)
        pos = observation["current_ee_pos"]
        quat = observation["current_ee_quat"]
        n = max(int(steps), 1)
        threshold = self._joint_threshold_for_phase(phase)
        targets: list[SkillTarget] = []
        for idx in range(n):
            ratio = float(idx + 1) / float(n)
            joint_pos = start_q * (1.0 - ratio) + end_q * ratio
            if gripper_start is not None:
                gripper_cmd = float(gripper_start) + (float(gripper) - float(gripper_start)) * ratio
            else:
                gripper_cmd = float(gripper)
            targets.append(
                SkillTarget(
                    pos=pos,
                    quat=quat,
                    gripper=float(gripper_cmd),
                    phase=phase,
                    arm=arm,
                    joint_pos=joint_pos,
                    extras={"joint_position_threshold": threshold},
                )
            )
        return targets

    def _resolve_arm(self, observation: dict[str, Any]) -> str:
        if self._saved_arm:
            return self._normalize_arm(self._saved_arm, observation)
        return self._normalize_arm(None, observation)

    def _read_arm_joint_q(self, observation: dict[str, Any]) -> torch.Tensor | None:
        robot = observation.get("robot")
        if robot is None:
            return None
        env_idx = int(observation.get("env_idx", 0))
        data = getattr(robot, "data", None)
        joint_names = list(getattr(data, "joint_names", None) or [])
        joint_pos = as_torch(getattr(data, "joint_pos", None))
        arm = self._resolve_arm(observation)
        indices = self._arm_joint_indices(joint_names, arm)
        if not indices or not isinstance(joint_pos, torch.Tensor):
            return None
        return joint_pos[env_idx, indices].detach().clone().to(dtype=torch.float32)

    def _arm_joint_indices(self, joint_names: list[str], arm: str) -> list[int]:
        prefix = "right_arm_joint" if arm == "right" else "left_arm_joint"
        matches: list[tuple[int, int]] = []
        for idx, name in enumerate(joint_names):
            if not isinstance(name, str) or not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if suffix.isdigit():
                matches.append((int(suffix), idx))
        if matches:
            matches.sort(key=lambda item: item[0])
            return [idx for _, idx in matches]
        if arm == "right":
            return []
        fallback = str(getattr(self.cfg, "arm_joint_prefix", None) or "").strip()
        if not fallback:
            return []
        matches = []
        for idx, name in enumerate(joint_names):
            if isinstance(name, str) and name.startswith(fallback):
                suffix = name[len(fallback) :]
                if suffix.isdigit():
                    matches.append((int(suffix), idx))
        matches.sort(key=lambda item: item[0])
        return [idx for _, idx in matches]

    def _knob_goal(self, observation: dict[str, Any]):
        env = observation.get("gym_env")
        if env is None:
            return None
        env_idx = int(observation.get("env_idx", 0))
        return read_pinned_twist_goal(env, env_idx)

    def _knob_joint_rad(self, observation: dict[str, Any]) -> float | None:
        goal = self._knob_goal(observation)
        if goal is None:
            return None
        env = observation.get("gym_env")
        if env is None:
            return None
        articulation_name = self.cfg.articulation_name or self.cfg.object_name
        return get_articulation_joint_value(
            env,
            articulation_name,
            int(observation.get("env_idx", 0)),
            joint_name=goal.joint_name,
        )

    def _knob_remaining_rad(self, observation: dict[str, Any]) -> float | None:
        goal = self._knob_goal(observation)
        current = self._knob_joint_rad(observation)
        if goal is None or current is None:
            return None
        from manaenv.assets.objects.object_library import KnobA

        target = KnobA.display_value_to_joint_angle(goal.target_temperature)
        return float(target - float(current))

    def _knob_at_target(self, observation: dict[str, Any]) -> bool:
        remaining = self._knob_remaining_rad(observation)
        if remaining is None:
            return False
        return abs(remaining) <= float(self.cfg.knob_reached_tolerance_rad)

    def _last_joint_twist_direction(self, observation: dict[str, Any]) -> int:
        remaining = self._knob_remaining_rad(observation)
        if remaining is None or abs(remaining) < 1.0e-6:
            return 0
        # Panel temp up (e.g. 15→45) moves knob joint −90°→−270° (remaining < 0).
        # ``last_joint_twist_sign`` is the wrist↔knob coupling: EX001 uses -1 so
        # knob needs more negative angle ⇒ last joint increases (+1).
        knob_sign = 1 if remaining > 0.0 else -1
        coupling = float(self.cfg.last_joint_twist_sign)
        if abs(coupling) < 1.0e-6:
            coupling = -1.0
        return int(knob_sign * coupling)

    def _last_joint_deg(self, observation: dict[str, Any]) -> float | None:
        current_q = self._read_arm_joint_q(observation)
        if current_q is None or current_q.numel() == 0:
            return None
        return joint_rad_to_deg(float(current_q[-1].item()))


register_skill("twist", TwistSkill)

__all__ = ["TwistSkill", "TwistSkillCfg"]
