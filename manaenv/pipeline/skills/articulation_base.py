"""Shared base for grip-based articulation skills.

Concrete skills subclass this and register their own ``skill_type``:

- :class:`~manaenv.pipeline.skills.pull_skill.PullSkill` (``pull``): prismatic
  drawer OPEN.
- :class:`~manaenv.pipeline.skills.open_skill.OpenDoorSkill` (``open``) /
  :class:`~manaenv.pipeline.skills.close_skill.CloseDoorSkill` (``close``):
  revolute door arc.
- :class:`~manaenv.pipeline.skills.twist_skill.TwistSkill` (``twist``): revolute
  knob rotated about its own axis.

This base is NOT registered on its own. Drawer CLOSE is deliberately absent: it
needs no grasp and is handled by :class:`~manaenv.pipeline.skills.push_skill.PushSkill`
(``push``).

Unlike ``PushSkill`` (which contacts a rigid OBB face with a fixed-pose gripper),
these skills **grip the handle** using the same offline grasp labels as
:class:`PickSkill`, then translate/rotate the closed gripper along the joint to
pull a drawer, swing a door, or twist a knob.

Phase sequence emitted per attempt::

    start -> approach -> descend(grasp) -> close(gripper) -> slide(linear/arc)
          -> release(gripper) -> retreat

Articulation model (per operated link): the grasp is anchored to the LIVE pose
of the operated child link (named by ``handle_body_name``), read via
``body_pos_w``/``body_quat_w``, NOT the articulation root. So the grasp labels
live in that link's local frame and all identical handle links share ONE tiny
label set -- the difference comes entirely from each link's live pose.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from manaenv.utils.debug_print import manaprint
from manaenv.utils.scene_object_utils import get_articulation_body_env_pose, get_articulation_joint_value
from manaenv.utils.tensor_utils import as_torch

from .base import MOTION_KIND_ARC, MOTION_KIND_LINEAR_CARTESIAN, SkillTarget
from .pick_runtime.grasp_sampler import isaaclab_math_utils, load_manifest_payload
from .pick_skill import PickSkill, PickSkillCfg


def _draw_grasp_enabled() -> bool:
    """Debug-only grasp-decode overlay gate (``MANAENV_DRAW_GRASP=1``).

    OFF by default so evaluation skips building the viz frames entirely
    (see :meth:`_visualize_grasp_decode_frames`).
    """
    return os.environ.get("MANAENV_DRAW_GRASP", "").strip().lower() in ("1", "true", "yes")


@dataclass
class ArticulationGripSkillCfg(PickSkillCfg):
    """Base config for grip-based articulation skills.

    Inherits all grasp-label fields from :class:`PickSkillCfg`
    (``grasp_label_manifest`` / ``grasp_label_asset`` / ``grasp_region`` /
    ``filter_*_dir`` / ``approach_distance`` / ``top_k`` / gripper values, ...).
    Concrete subclasses override ``skill_type`` / ``joint_type`` / ``direction``.
    """

    skill_type: str = "pull"
    # Handles are grasped from the FRONT (horizontal approach), so the shape/pose
    # auto-profiles inherited from PickSkill (which force a top-down "downward"
    # approach for tabletop objects) would reject the natural handle grasps.
    # Default to the "manual" profile: no shape-based direction filtering, so
    # candidate selection relies solely on grasp_region plus any explicit
    # filter_*_dir a YAML chooses to set.
    filter_profile: str | None = "manual"
    # Revolute only: "open" swings the door out, "close" swings it shut. When
    # left empty it is inferred from skill_type (close -> close, else open).
    # Prismatic pull is always "open" (drawer closing is handled by PushSkill).
    direction: str = ""
    # Joint axis in the asset's object-local frame. Prismatic: the slide axis
    # (e.g. a front-opening drawer slides +X). Revolute: the hinge axis.
    joint_axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    # Travel distance of the prismatic slide (meters, world).
    slide_distance: float = 0.18
    # How far to back the gripper off the handle after releasing.
    retreat_distance: float = 0.08
    # Articulation prim name to read the joint state from. Defaults to
    # ``object_name`` (the asset is usually its own articulation).
    articulation_name: str = ""
    # Operated child link (handle body) this instance grasps, e.g. a specific
    # drawer front or door leaf. When set, the grasp decode is anchored to this
    # link's LIVE pose (``body_pos_w``/``body_quat_w``) instead of the root.
    handle_body_name: str = ""
    # Specific joint name driving the handle this instance targets. ``None`` uses
    # the first joint. Set per handle when an asset exposes one joint per handle.
    joint_name: str | None = None
    # When True, and NOT using ``handle_body_name`` (root-anchored legacy path),
    # the decoded grasp labels (defined in the closed-state root frame) are
    # shifted along the open axis by the live joint position so the grasp tracks
    # the handle. Ignored when ``handle_body_name`` is set.
    track_joint_offset: bool = True
    # Prismatic vs revolute. Empty inherits from the grasp manifest
    # ``joint_type`` when available, else from the skill_type.
    joint_type: str = ""
    # Revolute-only: hinge pivot in the operated link's local frame. For a knob
    # twist keep this at the grasp point (link origin) so the motion is ~in-place.
    hinge_pivot_local: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Revolute-only: signed sweep angle (radians) for one open/close/twist motion.
    arc_sweep_angle: float = 0.52
    # Revolute-only: arc densification sample count (trajectory engine).
    arc_samples: int = 12

    def resolved_direction(self) -> str:
        explicit = str(self.direction or "").strip().lower()
        if explicit in ("open", "close"):
            return explicit
        return "close" if str(self.skill_type).strip().lower() == "close" else "open"


class ArticulationGripSkill(PickSkill):
    """Grip a handle and translate (prismatic) or rotate (revolute) it."""

    Cfg = ArticulationGripSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        # Bind by AFFORDANCE, never by object class: the concrete taxonomy
        # (drawer/door/knob) lives in the scene/task roles, not the skill.
        return ("openable", "object")

    def __init__(self, cfg: ArticulationGripSkillCfg):
        super().__init__(cfg)
        ee_rot = getattr(cfg, "ee_frame_rotation_wxyz", None)
        if ee_rot is not None:
            manaprint(f"[{cfg.skill_type}] ee_frame_rotation_wxyz={list(ee_rot)}")
        else:
            manaprint(f"[{cfg.skill_type}] ee_frame_rotation_wxyz=identity (canonical EE frame)")

    def _resolved_joint_type(self) -> str:
        explicit = str(getattr(self.cfg, "joint_type", "") or "").strip().lower()
        if explicit in ("prismatic", "revolute"):
            return explicit
        manifest_jt = str(getattr(self._grasp_data, "joint_type", "") or "").strip().lower()
        if manifest_jt in ("prismatic", "revolute"):
            return manifest_jt
        # open/close (doors) and twist (knobs) are revolute; pull (drawers) is prismatic.
        if str(self.cfg.skill_type).strip().lower() in ("open", "close", "twist"):
            return "revolute"
        return "prismatic"

    def _joint_axis_local_tensor(self, device: torch.device) -> torch.Tensor:
        axis = getattr(self.cfg, "joint_axis_local", (1.0, 0.0, 0.0))
        manifest_axis = getattr(self._grasp_data, "joint_axis_local", None)
        if manifest_axis is not None and not str(getattr(self.cfg, "joint_type", "") or "").strip():
            axis = manifest_axis
        tensor = torch.as_tensor(axis, dtype=torch.float32, device=device).reshape(3)
        return tensor / torch.linalg.norm(tensor).clamp(min=1.0e-8)

    def _hinge_frame(self, obj_pos_env: Any, obj_quat: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = torch.as_tensor(obj_pos_env, dtype=torch.float32, device=device).reshape(3)
        obj_quat_t = torch.as_tensor(obj_quat, dtype=torch.float32, device=device).reshape(1, 4)
        obj_rot = isaaclab_math_utils().matrix_from_quat(obj_quat_t).squeeze(0).to(dtype=torch.float32)
        pivot_local = torch.as_tensor(self.cfg.hinge_pivot_local, dtype=torch.float32, device=device).reshape(3)
        pivot_env = obj_pos + obj_rot @ pivot_local
        axis_world = obj_rot @ self._joint_axis_local_tensor(device)
        axis_world = axis_world / torch.linalg.norm(axis_world).clamp(min=1.0e-8)
        return pivot_env, axis_world

    def _rotate_pose_about_axis(
        self,
        pivot: torch.Tensor,
        axis: torch.Tensor,
        angle: float,
        pos: torch.Tensor,
        quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu = isaaclab_math_utils()
        rot_q = mu.quat_from_angle_axis(
            torch.tensor(float(angle), dtype=torch.float32, device=pivot.device), axis.reshape(3)
        ).reshape(4)
        delta = pos.reshape(3) - pivot.reshape(3)
        new_pos = pivot.reshape(3) + mu.quat_apply(rot_q.unsqueeze(0), delta.unsqueeze(0)).squeeze(0)
        new_quat = mu.quat_mul(rot_q.unsqueeze(0), quat.reshape(1, 4)).squeeze(0).to(dtype=torch.float32)
        return new_pos, new_quat

    def _compute_slide_pose(
        self,
        *,
        grasp_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
        obj_pos_env: Any,
        obj_quat: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        sign = -1.0 if self.cfg.resolved_direction() == "close" else 1.0
        if self._resolved_joint_type() == "revolute":
            pivot_env, axis_world = self._hinge_frame(obj_pos_env, obj_quat, device)
            sweep = sign * float(self.cfg.arc_sweep_angle)
            slide_pos, slide_quat = self._rotate_pose_about_axis(pivot_env, axis_world, sweep, grasp_pos, grasp_quat)
            extras = {
                "motion_kind": MOTION_KIND_ARC,
                "arc_pivot": pivot_env.detach().clone(),
                "arc_axis": axis_world.detach().clone(),
                "arc_sweep_angle": float(sweep),
                "arc_samples": int(self.cfg.arc_samples),
            }
            return slide_pos, slide_quat, extras
        slide_world = self._slide_world(obj_quat, device)
        slide_pos = grasp_pos + slide_world * float(self.cfg.slide_distance)
        return slide_pos, grasp_quat, {"motion_kind": MOTION_KIND_LINEAR_CARTESIAN}

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        observation = super().observe(env, obs, env_idx)
        if self._grasp_data.rows is None and self.cfg.grasp_label_manifest:
            load_manifest_payload(cfg=self.cfg, data=self._grasp_data)
        articulation_name = self.cfg.articulation_name or self.cfg.object_name
        open_distance = 0.0
        if self.cfg.track_joint_offset:
            value = get_articulation_joint_value(
                env,
                articulation_name,
                int(env_idx),
                joint_name=self.cfg.joint_name,
            )
            if value is not None:
                open_distance = float(value)
        observation["joint_open_distance"] = open_distance
        # Anchor the decode to the operated child link (handle body) so labels
        # authored in that link's local frame are exact and track the joint.
        body_name = str(self.cfg.handle_body_name or "").strip()
        if str(getattr(self._grasp_data, "reference_frame", "") or "") == "body" and not body_name:
            manaprint(
                f"[{self.cfg.skill_type}] manifest reference_frame=body but handle_body_name is unset; "
                "grasps will decode against the articulation root (legacy path)."
            )
        if body_name:
            env_origin = observation.get("env_origin")
            body_pose = get_articulation_body_env_pose(
                env,
                articulation_name,
                body_name,
                int(env_idx),
                env_origin=env_origin if isinstance(env_origin, torch.Tensor) else None,
            )
            if body_pose is not None:
                observation["object_pose_env"] = (body_pose[0], body_pose[1])
                observation["grasp_reference_frame"] = "body"
            else:
                manaprint(
                    f"[{self.cfg.skill_type}] handle_body_name={body_name!r} not found on "
                    f"articulation {articulation_name!r}; falling back to root-anchored pose"
                )
        self._log_articulation_frames(env, int(env_idx))
        return observation

    def _log_articulation_frames(self, env: Any, env_idx: int) -> None:
        """One-shot: dump root + per-body poses so grasp labels can be authored
        in the SAME frame decode reads (``root_pos_w``). ``root_local`` is the
        body position in the articulation-root frame (meters, scaled); divide by
        the scene object scale to get the object-local UNSCALED label coordinate.
        """
        if getattr(self, "_frames_logged", False):
            return
        try:
            scene = getattr(env, "scene", None)
            arts = getattr(scene, "articulations", None) if scene is not None else None
            name = self.cfg.articulation_name or self.cfg.object_name
            art = arts[name] if arts is not None and name in arts else None
            if art is None:
                manaprint(f"[{self.cfg.skill_type}] FRAME-DIAG: articulation {name!r} not in scene.articulations")
                self._frames_logged = True
                return
            data = art.data
            i = int(env_idx)
            root_p = as_torch(data.root_pos_w)[i].to(torch.float32)
            root_q = as_torch(data.root_quat_w)[i].to(torch.float32)
            body_p = as_torch(data.body_pos_w)[i].to(torch.float32)
            names = list(getattr(art, "body_names", None) or getattr(data, "body_names", []) or [])
            mu = isaaclab_math_utils()
            root_r = mu.matrix_from_quat(root_q.unsqueeze(0)).squeeze(0)
            local = torch.einsum("ij,nj->ni", root_r.transpose(0, 1), body_p - root_p.unsqueeze(0))
            manaprint(
                f"[{self.cfg.skill_type}] FRAME-DIAG root_pos_w={[round(v, 4) for v in root_p.tolist()]} "
                f"root_quat_xyzw={[round(v, 4) for v in root_q.tolist()]} n_bodies={len(names)}"
            )
            for bn, w, lo in zip(names, body_p.tolist(), local.tolist()):
                manaprint(
                    f"[{self.cfg.skill_type}] FRAME-DIAG body {bn!r}: world={[round(v, 4) for v in w]} "
                    f"root_local(scaled)={[round(v, 4) for v in lo]}"
                )
            self._frames_logged = True
        except Exception as exc:  # pragma: no cover - diagnostics only
            manaprint(f"[{self.cfg.skill_type}] FRAME-DIAG failed: {type(exc).__name__}: {exc}")
            self._frames_logged = True

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        # Reuse PickSkill's offline-grasp pipeline (decode -> filter -> blocklist
        # -> cooldown -> rank). Unlike pick, grip-articulation does NOT do batched
        # reachability here: it commits the top-ranked handle grasp and lets the
        # engine's replan loop (``notify_unreachable_candidate``) cool down and
        # skip an unreachable one on the next call. The geometric filter must not
        # fall back to the full label set (grasp_region picks the operated handle).
        candidates = self._select_grasp_candidates(
            observation,
            rank_top_k=int(self.cfg.top_k),
            full_fallback_on_empty=False,
            refuse_when_thin=False,
        )
        if candidates is None:
            return []
        ranked_indices = candidates.ranked_indices
        obj_pos_env = candidates.obj_pos_env
        obj_quat = candidates.obj_quat
        grasp_pos_env = candidates.grasp_pos_env

        # Shift env grasp targets onto the live handle: labels are authored in the
        # closed-state root frame, so add the open-axis travel reported by the
        # prismatic joint. SKIP when anchored to the operated link's live pose
        # (``grasp_reference_frame == "body"``): the body pose already tracks it.
        uses_body_frame = str(observation.get("grasp_reference_frame") or "") == "body"
        open_distance = float(observation.get("joint_open_distance", 0.0) or 0.0)
        if not uses_body_frame and abs(open_distance) > 1.0e-6:
            grasp_pos_env = grasp_pos_env + self._open_axis_world(obj_quat, grasp_pos_env.device) * open_distance

        device = grasp_pos_env.device
        selected_arm = self._select_nearest_arm_to_object(observation, obj_pos_env)
        selected_index = int(ranked_indices[0].item())
        selected_grasp_pos = grasp_pos_env[selected_index]
        selected_quat = candidates.grasp_quat_env[selected_index]
        selected_approach_w = candidates.approach_world[selected_index].reshape(3)
        selected_pre_pos = selected_grasp_pos - selected_approach_w * float(self.cfg.approach_distance)
        selected_slide_pos, selected_slide_quat, slide_motion = self._compute_slide_pose(
            grasp_pos=selected_grasp_pos,
            grasp_quat=selected_quat,
            obj_pos_env=obj_pos_env,
            obj_quat=obj_quat,
            device=device,
        )
        selected_retreat_pos = selected_slide_pos - selected_approach_w * float(self.cfg.retreat_distance)
        selected_score = float(candidates.scores[selected_index].item())
        selected_width = float(candidates.widths[selected_index].item())
        label_pos_env = None
        label_quat_env = None
        label_data = self._grasp_data
        if label_data.last_label_pos_env is not None and label_data.last_label_quat_env is not None:
            label_pos_env = label_data.last_label_pos_env[selected_index]
            label_quat_env = label_data.last_label_quat_env[selected_index]

        # self._log_selected_pose(
        #     observation=observation,
        #     arm=selected_arm,
        #     obj_pos_env=obj_pos_env,
        #     obj_quat=obj_quat,
        #     candidate_index=selected_index,
        #     score=selected_score,
        #     width=selected_width,
        #     pre_pos=selected_pre_pos,
        #     grasp_pos=selected_grasp_pos,
        #     slide_pos=selected_slide_pos,
        #     retreat_pos=selected_retreat_pos,
        #     grasp_quat=selected_quat,
        #     approach_world=selected_approach_w,
        #     slide_world=slide_world,
        #     label_pos=label_pos_env,
        #     label_quat=label_quat_env,
        # )
        # Only build the decode-overlay frames when the debug flag is on; the
        # tensor work is pure visualization and must not tax evaluation.
        self._pending_grasp_decode_viz = (
            self._build_grasp_decode_viz_frames(
                obj_pos_env=obj_pos_env,
                obj_quat=obj_quat,
                label_pos=label_pos_env,
                label_quat=label_quat_env,
                grasp_pos=selected_grasp_pos,
                grasp_quat=selected_quat,
            )
            if _draw_grasp_enabled()
            else None
        )
        return self._build_grip_targets(
            arm=selected_arm,
            pre_pos=selected_pre_pos,
            grasp_pos=selected_grasp_pos,
            slide_pos=selected_slide_pos,
            retreat_pos=selected_retreat_pos,
            grasp_quat=selected_quat,
            slide_quat=selected_slide_quat,
            slide_motion=slide_motion,
            candidate_index=selected_index,
            score=selected_score,
            width=selected_width,
        )

    def _log_selected_pose(
        self,
        *,
        observation: dict[str, Any],
        arm: str,
        obj_pos_env: Any,
        obj_quat: Any,
        candidate_index: int,
        score: float,
        width: float,
        pre_pos: torch.Tensor,
        grasp_pos: torch.Tensor,
        slide_pos: torch.Tensor,
        retreat_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
        approach_world: torch.Tensor,
        slide_world: torch.Tensor,
        label_pos: torch.Tensor | None = None,
        label_quat: torch.Tensor | None = None,
    ) -> None:
        """Log the selected handle grasp geometry for debugging.

        ``approach_world`` is the labelled canonical approach axis (+x in the
        grasp sampler frame) rotated into env/world; pre_grasp sits ``approach_distance``
        back along this vector from the handle. ``slide_world`` is the joint axis
        (``joint_axis_local``, signed by open/close).
        """
        env_idx = int(observation.get("env_idx", 0))
        mu = isaaclab_math_utils()
        rot = mu.matrix_from_quat(grasp_quat.reshape(1, 4).to(dtype=torch.float32)).reshape(3, 3)
        ee_approach_w = rot @ torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=rot.device)
        ee_close_w = rot @ torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=rot.device)
        approach_unit = approach_world / torch.linalg.norm(approach_world).clamp(min=1.0e-8)
        slide_unit = slide_world / torch.linalg.norm(slide_world).clamp(min=1.0e-8)

        def _fv(t: torch.Tensor, n: int) -> str:
            return "[" + ", ".join(f"{float(t[i]):+.4f}" for i in range(min(n, t.numel()))) + "]"

        arm_order = self._ranked_arms_to_object(observation, obj_pos_env)
        del env_idx, ee_approach_w, ee_close_w, approach_unit, slide_unit, arm_order
        # label_line = ""
        # if label_pos is not None and label_quat is not None:
        #     label_line = (
        #         f" label_pos_env={_fv(label_pos.reshape(-1), 3)} "
        #         f"label_quat_wxyz={_fv(label_quat.reshape(-1), 4)} "
        #         f"label_to_grasp_env={_fv((grasp_pos.reshape(3) - label_pos.reshape(3)), 3)} "
        #         f"body_to_label_env={_fv((label_pos.reshape(3) - torch.as_tensor(obj_pos_env, dtype=torch.float32).reshape(3)), 3)} "
        #     )
        # manaprint(
        #     f"[{self.cfg.skill_type} goal env={env_idx}] arm={arm} arm_order={arm_order} "
        #     f"dir={self.cfg.resolved_direction()} row={candidate_index} score={score:.3f} width={width:.4f} "
        #     f"approach_dist={float(self.cfg.approach_distance):.3f} slide_dist={float(self.cfg.slide_distance):.3f} "
        #     f"backoff={float(self.cfg.sampler_to_link6_backoff):.4f} "
        #     f"approach_axis_local={list(getattr(self.cfg, 'approach_axis_local', None) or getattr(self._grasp_data, 'effective_approach_axis', None) or ())} "
        #     f"label_approach_world={_fv(approach_unit, 3)} slide_axis_world={_fv(slide_unit, 3)} "
        #     f"{label_line}"
        #     f"pre_grasp_env={_fv(pre_pos.reshape(-1), 3)} grasp_env={_fv(grasp_pos.reshape(-1), 3)} "
        #     f"slide_env={_fv(slide_pos.reshape(-1), 3)} retreat_env={_fv(retreat_pos.reshape(-1), 3)} "
        #     f"grasp_quat_wxyz={_fv(grasp_quat.reshape(-1), 4)} "
        #     f"ee_approach_world={_fv(ee_approach_w, 3)} ee_close_world={_fv(ee_close_w, 3)} "
        #     f"obj_pos_env={_fv(torch.as_tensor(obj_pos_env, dtype=torch.float32).reshape(-1), 3)} "
        #     f"obj_quat_wxyz={_fv(torch.as_tensor(obj_quat, dtype=torch.float32).reshape(-1), 4)}"
        # )

    @staticmethod
    def _build_grasp_decode_viz_frames(
        *,
        obj_pos_env: Any,
        obj_quat: Any,
        label_pos: torch.Tensor | None,
        label_quat: torch.Tensor | None,
        grasp_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
    ) -> list[dict[str, Any]]:
        """Frames for MANAENV_DRAW_GRASP decode overlay: label contact + TCP."""
        frames: list[dict[str, Any]] = []
        if label_pos is not None and label_quat is not None:
            frames.append(
                {
                    "tag": "label",
                    "pos": label_pos.reshape(3).to(dtype=torch.float32),
                    "quat": label_quat.reshape(4).to(dtype=torch.float32),
                    "scale": 0.05,
                }
            )
        frames.append(
            {
                "tag": "grasp_tcp",
                "pos": grasp_pos.reshape(3).to(dtype=torch.float32),
                "quat": grasp_quat.reshape(4).to(dtype=torch.float32),
                "scale": 0.07,
            }
        )
        return frames

    def _open_axis_world(self, obj_quat: Any, device: torch.device) -> torch.Tensor:
        """Unit pull-out (open) axis in world frame."""
        axis_local = torch.as_tensor(self.cfg.joint_axis_local, dtype=torch.float32, device=device).reshape(3)
        axis_local = axis_local / torch.linalg.norm(axis_local).clamp(min=1.0e-8)
        obj_quat_t = torch.as_tensor(obj_quat, dtype=torch.float32, device=device).reshape(1, 4)
        obj_rot = isaaclab_math_utils().matrix_from_quat(obj_quat_t).squeeze(0).to(dtype=torch.float32)
        axis_world = (obj_rot @ axis_local).reshape(3)
        return axis_world / torch.linalg.norm(axis_world).clamp(min=1.0e-8)

    def _slide_world(self, obj_quat: Any, device: torch.device) -> torch.Tensor:
        """Joint axis in world frame, signed by open/close direction."""
        sign = -1.0 if self.cfg.resolved_direction() == "close" else 1.0
        return self._open_axis_world(obj_quat, device) * sign

    def _build_grip_targets(
        self,
        *,
        arm: str,
        pre_pos: torch.Tensor,
        grasp_pos: torch.Tensor,
        slide_pos: torch.Tensor,
        retreat_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
        candidate_index: int,
        score: float,
        width: float,
        slide_quat: torch.Tensor | None = None,
        slide_motion: dict[str, Any] | None = None,
    ) -> list[SkillTarget]:
        extras = {"candidate_index": int(candidate_index), "score": float(score), "width": float(width)}
        gripper_open = float(self.cfg.gripper_open)
        gripper_close = float(self.cfg.gripper_close)
        linear = {"motion_kind": MOTION_KIND_LINEAR_CARTESIAN}
        slide_q = slide_quat if slide_quat is not None else grasp_quat
        slide_extras = dict(slide_motion or {"motion_kind": MOTION_KIND_LINEAR_CARTESIAN})
        return [
            SkillTarget(
                pos=pre_pos, quat=grasp_quat, gripper=gripper_open, phase="pre_grasp", arm=arm, extras=dict(extras)
            ),
            SkillTarget(
                pos=grasp_pos,
                quat=grasp_quat,
                gripper=gripper_open,
                phase="grasp",
                arm=arm,
                extras={**extras, **linear},
            ),
            SkillTarget(
                pos=slide_pos,
                quat=slide_q,
                gripper=gripper_close,
                phase="slide",
                arm=arm,
                extras={**extras, **slide_extras},
            ),
            SkillTarget(
                pos=retreat_pos,
                quat=grasp_quat,
                gripper=gripper_open,
                phase="retreat",
                arm=arm,
                extras={**extras, **linear},
            ),
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
        phases = [t.phase for t in normalized]
        if phases != ["pre_grasp", "grasp", "slide", "retreat"]:
            # Fall back to the generic layout for unexpected target sets.
            return super().expand_to_waypoints(targets, current_pos, current_quat, settings)

        pre_t, grasp_t, slide_t, retreat_t = normalized
        gripper_open = float(self.cfg.gripper_open)
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
            # Close the gripper on the handle without moving (gripper-only step).
            {"pos": grasp_t.pos, "quat": grasp_t.quat, "gripper": gripper_close, "phase": "close"},
            # Drag/rotate the closed gripper along the joint (the link follows).
            self._target_to_waypoint(slide_t, phase="slide"),
            # Open the gripper to release the handle (gripper-only step).
            {"pos": slide_t.pos, "quat": slide_t.quat, "gripper": gripper_open, "phase": "release"},
            self._target_to_waypoint(retreat_t, phase="retreat"),
        ]


__all__ = ["ArticulationGripSkill", "ArticulationGripSkillCfg"]
