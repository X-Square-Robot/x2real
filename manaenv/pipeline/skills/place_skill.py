"""Vertical bbox-sampling place skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from manaenv.pipeline.skills.reachability import select_reachable
from manaenv.utils.scene_object_utils import get_env_origin

from .base import MOTION_KIND_LINEAR_CARTESIAN, Skill, SkillCfg, SkillTarget, register_skill
from .place_runtime import (
    PlaceTargetCfg,
    make_natural_pose_candidates,
    make_object_pose_constrained_candidates,
    resolve_place_candidates,
)


@dataclass
class PlaceObjectPoseConstraintCfg:
    """Optional object-root pose constraint for place targets."""

    enabled: bool = False
    root_position_source: str = "target_place_pos"
    root_position_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_mode: str = "axis_parallel_free_yaw"
    object_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)
    target_axis_world: tuple[float, float, float] = (0.0, 0.0, 1.0)
    allow_axis_flip: bool = False
    max_tilt_deg: float = 0.0
    tilt_samples: int = 1
    yaw_samples: int = 5
    yaw_center_mode: str = "current_ee_side"
    yaw_span_deg: float = 20.0
    # Same local-axis direction DSL as PickSkill. Filters are evaluated on the
    # final end-effector rotation in robot-base frame.
    filter_x_dir: list[Any] | None = None
    filter_y_dir: list[Any] | None = None
    filter_z_dir: list[Any] | None = None


@dataclass
class PlaceSkillCfg(SkillCfg):

    skill_type: str = "place"
    object_name: str = ""
    arm: str = "current"
    pre_place_z_offset: float = 0.20
    place_z_offset: float = 0.05
    post_place_z_offset: float = 0.15
    batch_size: int = 64
    top_k_alternatives: int = 8
    fallback_bbox_size: tuple[float, float, float] = (0.2, 0.2, 0.2)
    gripper_close: float | str = "closed"
    gripper_open: float | str = "open"
    hesitate_steps: int = 10

    approach_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    sampler_to_link6_backoff: float = 0.168

    pitch_samples: int = 8
    pitch_min_deg: float = 0.0
    pitch_max_deg: float = 90.0
    wrist_yaw_axis_local: tuple[float, float, float] = (0.0, 0.0, 1.0)

    target: Any = field(default_factory=PlaceTargetCfg)
    object_pose_constraint: Any = field(default_factory=PlaceObjectPoseConstraintCfg)

    def __post_init__(self) -> None:
        if isinstance(self.target, dict):
            self.target = PlaceTargetCfg(**self.target)
        elif self.target is None:
            self.target = PlaceTargetCfg()
        if isinstance(self.object_pose_constraint, dict):
            self.object_pose_constraint = PlaceObjectPoseConstraintCfg(**self.object_pose_constraint)
        elif self.object_pose_constraint is None:
            self.object_pose_constraint = PlaceObjectPoseConstraintCfg()


class PlaceSkill(Skill):
    """Place the held object over a sampled target."""

    Cfg = PlaceSkillCfg

    def __init__(self, cfg: PlaceSkillCfg) -> None:
        super().__init__(cfg)
        self._pending_alternatives: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._blocked_alt_indices: set[int] = set()
        self._alternatives_signature: tuple[Any, ...] | None = None

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("pickable", "object")

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        env_origin = get_env_origin(env, env_idx)
        object_name = self.cfg.object_name or str(obs.get("held_object_name") or "")
        if not object_name:
            raise ValueError("PlaceSkill requires object_name or a held_object_name from the previous skill.")
        # Sanity: the held object must still be resolvable in the scene.
        object_pos, object_quat, _ = self.read_object_pose_and_scale(env, object_name, env_idx, env_origin)
        current_ee_pos = obs.get("current_ee_pos")
        if current_ee_pos is None:
            raise ValueError("PlaceSkill.observe requires current_ee_pos in planner context.")
        current_ee_quat = obs.get("current_ee_quat")
        return {
            "gym_env": env,
            "robot": obs.get("robot"),
            "env_idx": int(env_idx),
            "env_origin": env_origin,
            "object_name": object_name,
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": (
                torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4)
                if current_ee_quat is not None
                else None
            ),
            "object_root_pos": object_pos,
            "object_root_quat": object_quat,
            "active_arm": obs.get("active_arm"),
            "held_object_name": obs.get("held_object_name"),
            "raw_observation": obs.get("raw_observation"),
            "robot_base_pos_world": obs.get("robot_base_pos_world"),
        }

    def reset_episode(self) -> None:
        self._pending_alternatives.clear()
        self._blocked_alt_indices.clear()
        self._alternatives_signature = None

    def notify_unreachable_candidate(self, candidate_index: int) -> None:
        try:
            idx = int(candidate_index)
        except (TypeError, ValueError):
            return
        if idx < 0 or idx >= len(self._pending_alternatives):
            return
        self._blocked_alt_indices.add(idx)
        if len(self._blocked_alt_indices) >= len(self._pending_alternatives):
            self._pending_alternatives.clear()
            self._blocked_alt_indices.clear()
            self._alternatives_signature = None

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        arm = self._resolve_arm(observation)
        target_cfg: PlaceTargetCfg = self.cfg.target
        target_sig: tuple[Any, ...] = (
            str(target_cfg.strategy),
            str(target_cfg.object_name or ""),
            tuple(target_cfg.objects or ()),
            tuple(target_cfg.x_ratio_range),
            tuple(target_cfg.y_ratio_range),
            self._object_pose_constraint_signature(),
        )
        sig = (
            arm,
            str(self.cfg.object_name or ""),
            target_sig,
            int(observation["env_idx"]),
        )
        if sig != self._alternatives_signature or not self._pending_alternatives:
            self._rebuild_alternatives(observation, evaluator, arm, sig)
        if not self._pending_alternatives:
            return []
        chosen = self._next_unblocked_index()
        if chosen is None:
            self._pending_alternatives.clear()
            self._blocked_alt_indices.clear()
            self._rebuild_alternatives(observation, evaluator, arm, sig)
            chosen = self._next_unblocked_index()
            if chosen is None:
                return []
        pre, place, quat = self._pending_alternatives[chosen]
        return self._build_targets(arm=arm, pre_pos=pre, place_pos=place, place_quat=quat, candidate_index=chosen)

    def _next_unblocked_index(self) -> int | None:
        for i in range(len(self._pending_alternatives)):
            if i not in self._blocked_alt_indices:
                return i
        return None

    def _rebuild_alternatives(
        self,
        observation: dict[str, Any],
        evaluator: Any | None,
        arm: str,
        sig: tuple[Any, ...],
    ) -> None:
        self._pending_alternatives.clear()
        self._blocked_alt_indices.clear()
        self._alternatives_signature = sig

        current_pos = observation["current_ee_pos"]
        device = current_pos.device
        object_pose_constrained = bool(getattr(self.cfg.object_pose_constraint, "enabled", False))
        candidates = resolve_place_candidates(
            env=observation["gym_env"],
            target_cfg=self.cfg.target,
            env_idx=int(observation["env_idx"]),
            env_origin=observation.get("env_origin"),
            batch_size=int(self.cfg.batch_size),
            pre_place_z_offset=float(self.cfg.pre_place_z_offset),
            place_z_offset=float(self.cfg.place_z_offset),
            fallback_size=tuple(self.cfg.fallback_bbox_size),
            device=device,
        )

        base_pos_world_obs = observation.get("robot_base_pos_world")
        if base_pos_world_obs is None:
            raise ValueError(
                "PlaceSkill requires robot_base_pos_world in the planner observation "
                "(SkillTrajectoryPlanner must resolve the arm base body pose and forward it)."
            )
        base_pos_env = torch.as_tensor(base_pos_world_obs, dtype=torch.float32, device=device).reshape(3)
        env_origin_t = observation.get("env_origin")
        if env_origin_t is not None:
            env_origin_t = torch.as_tensor(env_origin_t, dtype=torch.float32, device=device).reshape(3)
            base_pos_env = base_pos_env - env_origin_t

        if object_pose_constrained:
            pre_pos, place_pos, place_quat = make_object_pose_constrained_candidates(
                cfg=self.cfg.object_pose_constraint,
                target_pos_world=candidates.place_pos,
                observation=observation,
                pre_place_z_offset=float(self.cfg.pre_place_z_offset),
                dtype=current_pos.dtype,
                device=device,
            )
        else:
            pre_pos, place_pos, place_quat = make_natural_pose_candidates(
                self.cfg,
                target_pos_world=candidates.place_pos,
                base_pos_world=base_pos_env,
                dtype=current_pos.dtype,
                device=device,
            )
        n = int(pre_pos.shape[0])
        if n == 0:
            self._alternatives_signature = None
            return
        k = max(1, min(int(self.cfg.top_k_alternatives), n))
        current = observation["current_ee_pos"].reshape(1, 3)
        dist = torch.norm(pre_pos - current.expand_as(pre_pos), dim=1)

        selection = select_reachable(
            evaluator=evaluator,
            observation=observation,
            arm_order=[arm],
            stages=[(pre_pos, place_quat), (place_pos, place_quat)],
        )
        if selection is not None:
            primary = selection.index
            others = [i for i in range(n) if i != primary]
            others.sort(key=lambda i: float(dist[i].item()))
            order = [primary] + others[: k - 1]
        else:
            order = torch.argsort(dist)[:k].tolist()

        for row in order:
            self._pending_alternatives.append(
                (
                    pre_pos[int(row)].detach().clone(),
                    place_pos[int(row)].detach().clone(),
                    place_quat[int(row)].detach().clone(),
                )
            )

    def _build_targets(
        self,
        *,
        arm: str,
        pre_pos: torch.Tensor,
        place_pos: torch.Tensor,
        place_quat: torch.Tensor,
        candidate_index: int,
    ) -> list[SkillTarget]:
        device = pre_pos.device
        open_pos = place_pos.reshape(3)
        post_offset = torch.tensor([0.0, 0.0, float(self.cfg.post_place_z_offset)], dtype=torch.float32, device=device)
        extras: dict[str, Any] = {"candidate_index": int(candidate_index)}
        targets: list[SkillTarget] = [
            SkillTarget(
                pos=pre_pos.reshape(3),
                quat=place_quat.reshape(4),
                gripper=float(self.cfg.gripper_close),
                phase="pre_place",
                arm=arm,
                extras=dict(extras),
            ),
            SkillTarget(
                pos=open_pos,
                quat=place_quat.reshape(4),
                gripper=float(self.cfg.gripper_close),
                phase="place",
                arm=arm,
                extras={**extras, "motion_kind": MOTION_KIND_LINEAR_CARTESIAN},
            ),
            *[
                SkillTarget(
                    pos=open_pos,
                    quat=place_quat.reshape(4),
                    gripper=float(self.cfg.gripper_close),
                    phase="place_hold",
                    arm=arm,
                    extras=dict(extras),
                )
                for _ in range(max(int(self.cfg.hesitate_steps), 0))
            ],
            SkillTarget(
                pos=open_pos,
                quat=place_quat.reshape(4),
                gripper=float(self.cfg.gripper_open),
                phase="open_gripper",
                arm=arm,
                extras=dict(extras),
            ),
        ]
        if float(self.cfg.post_place_z_offset) > 0.0:
            targets.append(
                SkillTarget(
                    pos=open_pos + post_offset,
                    quat=place_quat.reshape(4),
                    gripper=float(self.cfg.gripper_open),
                    phase="retreat",
                    arm=arm,
                    extras={**extras, "motion_kind": MOTION_KIND_LINEAR_CARTESIAN},
                )
            )
        return targets

    def expand_to_waypoints(
        self,
        targets: list[SkillTarget],
        current_pos: torch.Tensor,
        current_quat: torch.Tensor,
        settings: Any,
    ) -> list[dict[str, Any]]:
        if not targets:
            return []
        normalized = [self._normalize_target(target) for target in targets]
        return [
            {
                "pos": current_pos.detach().clone().to(dtype=torch.float32),
                "quat": current_quat.detach().clone().to(dtype=torch.float32),
                "gripper": float(normalized[0].gripper),
                "phase": "start",
            },
            *[self._target_to_waypoint(target, phase=target.phase) for target in normalized],
        ]

    def _object_pose_constraint_signature(self) -> tuple[Any, ...]:
        cfg = self.cfg.object_pose_constraint
        return (
            bool(getattr(cfg, "enabled", False)),
            str(getattr(cfg, "root_position_source", "")),
            tuple(getattr(cfg, "root_position_offset", ())),
            str(getattr(cfg, "rotation_mode", "")),
            tuple(getattr(cfg, "object_axis_local", ())),
            tuple(getattr(cfg, "target_axis_world", ())),
            bool(getattr(cfg, "allow_axis_flip", False)),
            float(getattr(cfg, "max_tilt_deg", 0.0)),
            int(getattr(cfg, "tilt_samples", 1)),
            int(getattr(cfg, "yaw_samples", 1)),
            str(getattr(cfg, "yaw_center_mode", "")),
            float(getattr(cfg, "yaw_span_deg", 0.0)),
            tuple(getattr(cfg, "filter_x_dir", ()) or ()),
            tuple(getattr(cfg, "filter_y_dir", ()) or ()),
            tuple(getattr(cfg, "filter_z_dir", ()) or ()),
        )


register_skill("place", PlaceSkill)

__all__ = ["PlaceObjectPoseConstraintCfg", "PlaceSkill", "PlaceSkillCfg", "PlaceTargetCfg"]
