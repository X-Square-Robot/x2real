"""Rigid-body push skill: online direction from target + OBB contact face."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from manaenv.utils.debug_print import manaprint
from manaenv.utils.obb_utils import (
    get_obb_face_centers_env,
    local_axis_to_env,
    local_point_to_env,
    pick_face_by_direction,
)
from manaenv.utils.scene_object_utils import (
    get_articulation_body_env_pose,
    get_articulation_joint_value,
    get_env_origin,
)

from .base import Skill, SkillCfg, SkillTarget, register_skill
from .pick_runtime.grasp_sampler import isaaclab_math_utils


@dataclass
class PushSkillCfg(SkillCfg):
    """Configuration for pushing a rigid object toward a target."""

    skill_type: str = "push"
    object_name: str = ""
    target_object_name: str = ""
    target_pos_env: list[float] | None = None
    target_pos_world: list[float] | None = None  # alias; env-local in this stack
    push_distance: float | None = None
    approach_clearance: float = 0.05
    approach_z_offset: float = 0.08
    retreat_distance: float = 0.05
    project_xy: bool = True
    gripper: float = 0.0

    arm: str = "current"

    approach_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)

    tcp_offset: float = 0.168
    contact_point_local: list[float] | None = None
    push_axis_local: list[float] | None = None
    joint_name: str = ""
    # Articulation push: name of the articulation prim (defaults to object_name)
    # and the operated child link whose LIVE pose is the contact anchor (e.g. a
    # pulled-out drawer front). When handle_body_name is set, contact_point_local
    # / push_axis_local resolve in that link's frame and push_distance is read
    # from the joint openness (get_articulation_joint_value) so the drawer ends
    # fully shut -- no grasp, unlike the grip-based pull skill.
    articulation_name: str = ""
    handle_body_name: str = ""
    # Extra travel (m) added on top of the live joint openness when auto-sizing
    # the push so the drawer is driven just past closed.
    close_margin: float = 0.02


class PushSkill(Skill):
    """Push a rigid object using target-relative direction and OBB back-face contact."""

    Cfg = PushSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("pushable", "rotatable", "openable", "containable", "object")

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        env_origin = get_env_origin(env, env_idx)
        obj_pos, obj_quat, obj_scale = self.read_object_pose_and_scale(env, self.cfg.object_name, env_idx, env_origin)

        # Articulation push: re-anchor the contact to the operated child link's
        # LIVE pose (e.g. a pulled-out drawer front) and read the joint openness
        # so push_distance can drive it exactly shut.
        joint_open_distance: float | None = None
        body_name = str(self.cfg.handle_body_name or "").strip()
        if body_name:
            art_name = self.cfg.articulation_name or self.cfg.object_name
            link_pose = get_articulation_body_env_pose(env, art_name, body_name, env_idx, env_origin=env_origin)
            if link_pose is not None:
                obj_pos = link_pose[0].to(device=obj_pos.device, dtype=torch.float32)
                obj_quat = link_pose[1].to(device=obj_quat.device, dtype=torch.float32)
            else:
                manaprint(
                    f"[{self.cfg.skill_type}] handle_body_name={body_name!r} not found on "
                    f"articulation {art_name!r}; falling back to root-anchored pose"
                )
            joint_open_distance = get_articulation_joint_value(
                env, art_name, env_idx, joint_name=(self.cfg.joint_name or None)
            )

        target_pos = self._resolve_target_pos_env(env, env_idx, env_origin, obj_pos.device)
        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError("PushSkill.observe requires current_ee_pos/current_ee_quat in planner context.")

        # OBB faces are only used for rigid-body face picking (no explicit
        # contact_point_local and no operated link). Skip for articulation pushes
        # where the contact anchors to the live link pose.
        faces = None
        if not body_name and self.cfg.contact_point_local is None:
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
            "target_pos_env": target_pos,
            "joint_open_distance": joint_open_distance,
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
        }

    def _resolve_target_pos_env(
        self,
        env: Any,
        env_idx: int,
        env_origin: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Resolve the push target, or ``None`` when none is configured.

        A target is optional for articulation close: ``push_axis_local`` gives the
        direction and the joint openness sizes the push, so no external target is
        needed. ``plan_targets`` raises only if a target is actually required.
        """
        raw_target = self.cfg.target_pos_env
        if raw_target is None and self.cfg.target_pos_world is not None:
            raw_target = self.cfg.target_pos_world
        if raw_target is not None and len(raw_target) >= 3:
            return torch.as_tensor(raw_target[:3], dtype=torch.float32, device=device).reshape(3)

        target_name = str(self.cfg.target_object_name or "").strip()
        if not target_name:
            return None
        tgt_pos, _, _ = self.read_object_pose_and_scale(env, target_name, env_idx, env_origin)
        return tgt_pos.to(device=device, dtype=torch.float32).reshape(3)

    @staticmethod
    def _normalize_push_dir(delta: torch.Tensor, *, project_xy: bool) -> torch.Tensor:
        direction = delta.clone()
        if project_xy:
            direction[2] = 0.0
        norm = torch.linalg.norm(direction)
        if float(norm) <= 1.0e-6:
            raise ValueError("Push direction is degenerate; check object and target positions.")
        return direction / norm

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        obj_pos, obj_quat = observation["object_pose_env"]
        obj_scale = observation["object_scale"]
        target_pos = observation["target_pos_env"]
        device = observation["current_ee_pos"].device
        arm = self._select_nearest_arm_to_object(observation, obj_pos)
        current_pos, current_quat = self._read_selected_arm_ee_pose(observation, arm, device)

        if self.cfg.push_axis_local is not None:
            push_dir = local_axis_to_env(self.cfg.push_axis_local, object_quat_xyzw=obj_quat).to(
                device=device, dtype=torch.float32
            )
            if self.cfg.project_xy:
                push_dir[2] = 0.0
                push_dir = push_dir / torch.linalg.norm(push_dir).clamp(min=1.0e-8)
        else:
            if target_pos is None:
                raise ValueError(
                    "PushSkill needs push_axis_local or a target (target_pos_env/target_object_name) "
                    "to compute the push direction."
                )
            push_dir = self._normalize_push_dir(target_pos - obj_pos, project_xy=self.cfg.project_xy).to(
                device=device, dtype=torch.float32
            )

        uses_link = bool(str(self.cfg.handle_body_name or "").strip())
        if self.cfg.contact_point_local is not None:
            contact = local_point_to_env(
                self.cfg.contact_point_local,
                object_pos_env=obj_pos,
                object_quat_xyzw=obj_quat,
                obj_scale=obj_scale,
            ).to(device=device, dtype=torch.float32)
        elif uses_link:
            # Articulation push with no explicit contact: use the operated link's
            # live origin (a per-link OBB is not available here).
            contact = obj_pos.clone().to(device=device, dtype=torch.float32)
        else:
            contact_face = pick_face_by_direction(observation["obb_faces"], -push_dir)
            contact = contact_face.center_env.to(device=device, dtype=torch.float32)

        clearance = float(self.cfg.approach_clearance)
        z_offset = float(self.cfg.approach_z_offset)
        retreat_dist = float(self.cfg.retreat_distance)
        tcp_offset = float(self.cfg.tcp_offset)

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

        link6_contact = contact - approach_world * tcp_offset

        pre_push = link6_contact - push_dir * clearance
        pre_push[2] = max(float(pre_push[2]), float(current_pos[2]))

        approach = link6_contact.clone()
        approach[2] = max(float(approach[2]), float(link6_contact[2] + z_offset))

        joint_open_distance = observation.get("joint_open_distance")
        if self.cfg.push_distance is not None:
            push_dist = float(self.cfg.push_distance)
        elif joint_open_distance is not None:
            # Adaptive close: drive exactly the live joint openness (+ margin) shut.
            push_dist = abs(float(joint_open_distance)) + float(self.cfg.close_margin)
        else:
            if target_pos is None:
                raise ValueError(
                    "PushSkill needs push_distance, a live joint openness (handle_body_name + joint_name), "
                    "or a target to size the push."
                )
            delta = target_pos - obj_pos
            if self.cfg.project_xy:
                delta = delta.clone()
                delta[2] = 0.0
            push_dist = float(torch.linalg.norm(delta))

        push_end = link6_contact + push_dir * push_dist
        retreat = push_end - push_dir * retreat_dist
        retreat[2] = max(float(retreat[2]), float(current_pos[2]))

        gripper = float(self.cfg.gripper)
        return [
            SkillTarget(pos=pre_push, quat=current_quat, gripper=gripper, phase="pre_push", arm=arm),
            SkillTarget(pos=approach, quat=current_quat, gripper=gripper, phase="push_approach", arm=arm),
            SkillTarget(pos=link6_contact, quat=current_quat, gripper=gripper, phase="push_contact", arm=arm),
            SkillTarget(pos=push_end, quat=current_quat, gripper=gripper, phase="push", arm=arm),
            SkillTarget(pos=retreat, quat=current_quat, gripper=gripper, phase="retreat", arm=arm),
        ]

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


register_skill("push", PushSkill)

__all__ = ["PushSkill", "PushSkillCfg"]
