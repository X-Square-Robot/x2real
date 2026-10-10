"""Surface touch skill with online OBB affordance (no manifest)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from manaenv.utils.debug_print import manaprint
from manaenv.utils.obb_utils import get_obb_face_centers_env, local_axis_to_env, local_point_to_env, pick_face_by_name
from manaenv.utils.scene_object_utils import get_env_origin

from .base import Skill, SkillCfg, SkillTarget, register_skill
from .pick_runtime.grasp_sampler import isaaclab_math_utils


@dataclass
class TouchSkillCfg(SkillCfg):
    """Configuration for pressing/touching an object surface."""

    skill_type: str = "touch"
    object_name: str = ""
    # ``current`` / ``auto`` select the nearer EE to ``object_name``. Explicit
    # ``left`` / ``right`` overrides also work.
    arm: str = "current"
    surface: str = "top"
    approach_distance: float = 0.05
    hold_steps: int = 5
    retreat_distance: float = 0.05
    gripper: float = 0.0

    approach_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)

    tcp_offset: float = 0.168
    contact_point_local: list[float] | None = None
    axis_local: list[float] | None = None
    # Optional pre-grasp pose in the object's local frame. When set (directly or
    # via ``pregrasp_offset_local``), the arm visits this point before ``touch``.
    pregrasp_point_local: list[float] | None = None
    # Offset added to ``contact_point_local`` (or object root) for pre-grasp.
    pregrasp_offset_local: list[float] | None = None
    # Closed-gripper command (same semantics as PickSkill.gripper_close). When set,
    # a ``close`` phase is inserted after ``hold``.
    gripper_close: float | None = None
    # Retract pose in the object's local frame (relative to the bottle root).
    retract_point_local: list[float] | None = None


class TouchSkill(Skill):
    """Touch an object surface along its outward normal (online OBB or YAML override)."""

    Cfg = TouchSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("drink", "touchable", "object")

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        env_origin = get_env_origin(env, env_idx)
        obj_pos, obj_quat, obj_scale = self.read_object_pose_and_scale(env, self.cfg.object_name, env_idx, env_origin)
        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError("TouchSkill.observe requires current_ee_pos/current_ee_quat in planner context.")

        faces = get_obb_face_centers_env(
            env,
            self.cfg.object_name,
            env_idx,
            env_origin=env_origin,
            obj_scale=obj_scale,
        )

        return {
            "env_idx": env_idx,
            "env_origin": env_origin,
            "object_pose_env": (obj_pos, obj_quat),
            "object_scale": obj_scale,
            "obb_faces": faces,
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4),
            "active_arm": obs.get("active_arm"),
            "held_object_name": obs.get("held_object_name"),
            "raw_observation": obs.get("raw_observation"),
            # Needed by nearest-arm selection.
            "gym_env": env,
            "robot": obs.get("robot"),
            "object_name": self.cfg.object_name,
            "episode_start_ee": obs.get("episode_start_ee"),
        }

    def _resolve_pregrasp_point_local(self) -> list[float] | None:
        if self.cfg.pregrasp_point_local is not None:
            return list(self.cfg.pregrasp_point_local)
        if self.cfg.pregrasp_offset_local is None:
            return None
        offset = [float(v) for v in self.cfg.pregrasp_offset_local]
        if len(offset) != 3:
            raise ValueError(f"pregrasp_offset_local must have 3 elements, got {self.cfg.pregrasp_offset_local!r}")
        if self.cfg.contact_point_local is not None:
            base = [float(v) for v in self.cfg.contact_point_local]
        else:
            base = [0.0, 0.0, 0.0]
        return [base[i] + offset[i] for i in range(3)]

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        obj_pos, obj_quat = observation["object_pose_env"]
        obj_scale = observation["object_scale"]
        device = observation["current_ee_pos"].device

        if self.cfg.contact_point_local is not None:
            contact = local_point_to_env(
                self.cfg.contact_point_local,
                object_pos_env=obj_pos,
                object_quat_xyzw=obj_quat,
                obj_scale=obj_scale,
            ).to(device=device, dtype=torch.float32)
            if self.cfg.axis_local is not None:
                outward = local_axis_to_env(self.cfg.axis_local, object_quat_xyzw=obj_quat).to(
                    device=device, dtype=torch.float32
                )
            else:
                outward = pick_face_by_name(observation["obb_faces"], self.cfg.surface).normal_env.to(
                    device=device, dtype=torch.float32
                )
        else:
            face = pick_face_by_name(observation["obb_faces"], self.cfg.surface)
            contact = face.center_env.to(device=device, dtype=torch.float32)
            outward = face.normal_env.to(device=device, dtype=torch.float32)

        arm = self._select_nearest_arm_to_object(observation, contact)
        current_quat = self._read_arm_ee_quat(observation, arm).to(device=device, dtype=torch.float32)

        approach_dist = float(self.cfg.approach_distance)
        retreat_dist = float(self.cfg.retreat_distance)
        tcp_offset = float(self.cfg.tcp_offset)

        # World-frame gripper approach axis, derived from the current link6
        # orientation. This mirrors grasp_sampler.decode_grasps so the TCP
        # offset and the pre/retreat standoff are taken along the gripper's
        # own +x (or whatever ``approach_axis_local`` specifies), NOT along
        # the surface outward normal.
        approach_axis_local = torch.as_tensor(self.cfg.approach_axis_local, dtype=torch.float32, device=device).reshape(
            3
        )
        approach_axis_local = approach_axis_local / torch.linalg.norm(approach_axis_local).clamp(min=1.0e-8)
        rot_ee = (
            isaaclab_math_utils()
            .matrix_from_quat(current_quat.reshape(1, 4))
            .squeeze(0)
            .to(device=device, dtype=torch.float32)
        )
        approach_world = (rot_ee @ approach_axis_local).reshape(3)

        # Warn if the current gripper orientation does not point into the
        # surface (i.e. approach_world has a non-negative projection on the
        # outward normal). The touch will still execute but the fingertips
        # will not land on the intended face.
        alignment = float(torch.dot(approach_world, outward))
        if alignment > -0.1:

            manaprint(
                f"[touch] gripper approach axis is not aligned with -outward "
                f"(dot={alignment:+.3f}); object={self.cfg.object_name!r} "
                f"surface={self.cfg.surface!r}. Consider orienting the EE before touch."
            )

        # ``contact`` is the fingertip target on the surface. Pull link6 back
        # along ``approach_world`` so the fingertip (TCP) lands on the
        # surface: ``link6 = contact - approach_world * tcp_offset``.
        link6_contact = contact - approach_world * tcp_offset
        pre_touch = link6_contact - approach_world * approach_dist
        touch_pos = link6_contact
        retreat_pos = link6_contact - approach_world * retreat_dist

        pregrasp_local = self._resolve_pregrasp_point_local()
        link6_pregrasp: torch.Tensor | None = None
        if pregrasp_local is not None:
            pregrasp_contact = local_point_to_env(
                pregrasp_local,
                object_pos_env=obj_pos,
                object_quat_xyzw=obj_quat,
                obj_scale=obj_scale,
            ).to(device=device, dtype=torch.float32)
            link6_pregrasp = pregrasp_contact - approach_world * tcp_offset

        gripper = float(self.cfg.gripper)
        targets: list[SkillTarget] = []
        if link6_pregrasp is not None:
            targets.append(
                SkillTarget(pos=link6_pregrasp, quat=current_quat, gripper=gripper, phase="pre_grasp", arm=arm)
            )
            targets.append(SkillTarget(pos=touch_pos, quat=current_quat, gripper=gripper, phase="touch", arm=arm))
        else:
            targets.extend(
                [
                    SkillTarget(pos=pre_touch, quat=current_quat, gripper=gripper, phase="pre_touch", arm=arm),
                    SkillTarget(pos=touch_pos, quat=current_quat, gripper=gripper, phase="touch", arm=arm),
                ]
            )
        hold_count = max(int(self.cfg.hold_steps), 0)
        for _ in range(hold_count):
            targets.append(SkillTarget(pos=touch_pos, quat=current_quat, gripper=gripper, phase="hold", arm=arm))

        close_gripper = self.cfg.gripper_close
        if close_gripper is not None:
            targets.append(
                SkillTarget(
                    pos=touch_pos,
                    quat=current_quat,
                    gripper=float(close_gripper),
                    phase="close",
                    arm=arm,
                )
            )
            post_close_gripper = float(close_gripper)
        else:
            post_close_gripper = gripper

        if self.cfg.retract_point_local is not None:
            retract_contact = local_point_to_env(
                self.cfg.retract_point_local,
                object_pos_env=obj_pos,
                object_quat_xyzw=obj_quat,
                obj_scale=obj_scale,
            ).to(device=device, dtype=torch.float32)
            retract_pos = retract_contact - approach_world * tcp_offset
            targets.append(
                SkillTarget(
                    pos=retract_pos,
                    quat=current_quat,
                    gripper=post_close_gripper,
                    phase="retract_arm",
                    arm=arm,
                )
            )
        elif retreat_dist > 0.0:
            targets.append(
                SkillTarget(pos=retreat_pos, quat=current_quat, gripper=post_close_gripper, phase="retreat", arm=arm)
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


register_skill("touch", TouchSkill)

__all__ = ["TouchSkill", "TouchSkillCfg"]
