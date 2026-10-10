"""Offline grasp-label-backed single-arm pick skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from manaenv.pipeline.skills.reachability import select_reachable
from manaenv.utils.debug_print import manaprint
from manaenv.utils.grasp_runtime import ShapeMetadata, should_force_horizontal_close
from manaenv.utils.scene_object_utils import get_env_origin

from .base import MOTION_KIND_LINEAR_CARTESIAN, Skill, SkillCfg, SkillTarget, register_skill
from .pick_runtime import CandidateSelectorState, PickGraspData, decode_grasps, filter_candidates, poses_env_to_base
from .pick_runtime.candidate_selector import apply_candidate_cooldowns, rank_candidates
from .pick_runtime.grasp_sampler import isaaclab_math_utils, load_manifest_payload


@dataclass
class PickSkillCfg(SkillCfg):
    """Configuration for offline-label-backed single-arm pick."""

    skill_type: str = "pick"
    grasp_label_manifest: str = ""
    grasp_label_asset: str = ""
    object_name: str = ""
    arm: str = "auto"
    approach_axis_local: tuple[float, float, float] | None = None
    ee_frame_rotation_wxyz: tuple[float, float, float, float] | None = None
    sampler_to_link6_backoff: float = 0.168
    approach_distance: float = 0.15
    lift_height: float = 0.20
    top_k: int = 20
    reach_batch_size: int = 16
    reach_max_rounds: int = 3
    winner_plan_failure_cooldown: int = 4
    filter_x_dir: list[Any] | None = None
    filter_y_dir: list[Any] | None = None
    filter_z_dir: list[Any] | None = None
    direction_to_obj: str | None = None
    grasp_region: dict[str, Any] | None = None
    filter_profile: str | None = None
    gripper_open: float | str = "open"
    gripper_close: float | str = "closed"


@dataclass
class GraspCandidates:
    """Output of the shared offline-grasp candidate-selection pipeline.

    Holds the decoded per-label arrays plus the ranked (filtered + blocklisted +
    cooled-down + score-ranked) indices into them. Callers do their own FINAL
    selection: ``pick`` runs batched reachability over the ranked set, while the
    grip-based articulation skills take the top-ranked candidate.
    """

    ranked_indices: torch.Tensor
    scores: torch.Tensor
    widths: torch.Tensor
    grasp_pos_env: torch.Tensor
    grasp_quat_env: torch.Tensor
    approach_world: torch.Tensor
    obj_pos_env: torch.Tensor
    obj_quat: torch.Tensor


class PickSkill(Skill):
    """Pick skill that consumes offline `(N, 17)` grasp labels."""

    Cfg = PickSkillCfg

    @classmethod
    def default_role_keys(cls) -> tuple[str, ...]:
        return ("pickable", "pick_up_object", "object")

    def __init__(self, cfg: PickSkillCfg):
        super().__init__(cfg)
        self._grasp_data = PickGraspData()

        self._selector = CandidateSelectorState()

    def notify_failed_candidate(self, candidate_index: int) -> None:
        try:
            idx = int(candidate_index)
        except (TypeError, ValueError):
            return
        if idx < 0:
            return
        if self._selector.mark_failed(idx):
            manaprint(
                f"[pick] blocked candidate row={idx} for this episode "
                f"(total blocked={len(self._selector.blocked_candidates)})"
            )

    def notify_succeeded_candidate(self, candidate_index: int) -> None:

        try:
            idx = int(candidate_index)
        except (TypeError, ValueError):
            return
        if idx < 0:
            return
        self._selector.mark_succeeded(idx)

    def notify_unreachable_candidate(self, candidate_index: int) -> None:
        """Temporarily cool down a winner whose full cuRobo path became invalid."""
        try:
            idx = int(candidate_index)
        except (TypeError, ValueError):
            return
        if idx < 0:
            return
        failures = self._selector.mark_unreachable(
            idx, cooldown=max(int(getattr(self.cfg, "winner_plan_failure_cooldown", 0)), 0)
        )
        cooldown = max(int(getattr(self.cfg, "winner_plan_failure_cooldown", 0)), 0)
        if idx not in self._selector.succeeded_candidates or cooldown <= 0:
            return
        manaprint(
            f"[pick] cooling down successful candidate row={idx} for {cooldown} replan(s) "
            f"after cuRobo path failure (failures={failures})"
        )

    def reset_episode(self) -> None:
        if self._selector.blocked_candidates:
            manaprint(
                f"[pick] clearing blocklist of {len(self._selector.blocked_candidates)} candidates on episode reset"
            )
        self._selector.reset_episode()

    def observe(self, env: Any, obs: dict[str, Any], env_idx: int) -> dict[str, Any]:
        """Read object pose and current EE pose for the active env."""

        if env is None:
            raise RuntimeError("PickSkill.observe requires a live gym environment.")
        if not self.cfg.object_name:
            raise ValueError("PickSkill requires cfg.object_name to be resolved before runtime.")

        env_origin = get_env_origin(env, env_idx)
        obj_pos_env, obj_quat, obj_scale = self.read_object_pose_and_scale(
            env, self.cfg.object_name, env_idx, env_origin=env_origin
        )

        current_ee_pos = obs.get("current_ee_pos")
        current_ee_quat = obs.get("current_ee_quat")
        if current_ee_pos is None or current_ee_quat is None:
            raise ValueError("PickSkill.observe requires current_ee_pos/current_ee_quat in planner context.")

        return {
            "object_pose_env": (obj_pos_env, obj_quat),
            "object_scale": obj_scale,
            "current_ee_pos": torch.as_tensor(current_ee_pos, dtype=torch.float32).reshape(3),
            "current_ee_quat": torch.as_tensor(current_ee_quat, dtype=torch.float32).reshape(4),
            "env_origin": env_origin,
            "gym_env": env,
            "robot": obs.get("robot"),
            "env_idx": int(env_idx),
            "object_name": self.cfg.object_name,
            "active_arm": obs.get("active_arm"),
            "held_object_name": obs.get("held_object_name"),
            "raw_observation": obs.get("raw_observation"),
        }

    def _select_grasp_candidates(
        self,
        observation: dict[str, Any],
        *,
        rank_top_k: int,
        full_fallback_on_empty: bool,
        refuse_when_thin: bool,
    ) -> GraspCandidates | None:
        """Shared offline-grasp pipeline: decode -> geometric filter -> blocklist
        -> cooldown -> score-rank. Returns ``None`` when nothing survives.

        Reused by :meth:`plan_targets` here and by the grip-based articulation
        skills so the candidate machinery lives in ONE place.

        - ``full_fallback_on_empty``: when the geometric filter rejects every
          candidate, fall back to the full label set (pick) instead of giving up.
        - ``refuse_when_thin``: for thin objects whose closing axis must stay
          horizontal, refuse the full-set fallback and give up this tick.
        """
        (
            scores,
            widths,
            grasp_pos_env,
            grasp_quat_env,
            approach_world,
            grasp_pos_base,
            grasp_quat_base,
            grasp_quat_base_canonical,
        ) = decode_grasps(
            cfg=self.cfg,
            data=self._grasp_data,
            observation=observation,
            scale_points_fn=self.scale_object_local_points,
        )
        total_count = int(scores.numel())
        if total_count == 0:
            return None

        rot_base = isaaclab_math_utils().matrix_from_quat(grasp_quat_base_canonical).to(dtype=torch.float32)
        obj_pos_env, obj_quat = observation["object_pose_env"]
        obj_pos_base, _ = poses_env_to_base(
            pos_env=torch.as_tensor(obj_pos_env, dtype=torch.float32, device=grasp_pos_base.device).reshape(1, 3),
            quat_world=torch.as_tensor(obj_quat, dtype=torch.float32, device=grasp_pos_base.device).reshape(1, 4),
            observation=observation,
        )
        metadata = self._load_shape_metadata()
        object_quat_tensor = torch.as_tensor(obj_quat, dtype=torch.float32)
        candidate_indices = filter_candidates(
            cfg=self.cfg,
            rot_base=rot_base,
            grasp_pos_base=grasp_pos_base,
            obj_pos_base=obj_pos_base,
            metadata=metadata,
            object_quat=object_quat_tensor,
            log_fn=manaprint,
            grasp_pos_obj=self._grasp_data.last_grasp_pos_obj,
        )
        if int(candidate_indices.numel()) == 0:
            if refuse_when_thin and should_force_horizontal_close(metadata, object_quat_tensor):
                manaprint(
                    f"WARNING: [{self.cfg.skill_type}] thin object "
                    f"{str(self.cfg.grasp_label_asset or self.cfg.object_name)!r}: no labelled grasp "
                    "satisfies the horizontal closing-axis constraint; refusing full-set fallback this tick"
                )
                return None
            if not full_fallback_on_empty:
                manaprint(f"[{self.cfg.skill_type}] grasp_region/geometric filter rejected all candidates")
                return None
            manaprint(
                f"WARNING: [{self.cfg.skill_type}] geometric filter rejected all candidates; "
                "falling back to the full label set"
            )
            candidate_indices = torch.arange(total_count, dtype=torch.long, device=scores.device)

        ranked_indices = self._rank_filtered_candidates(candidate_indices, scores, rank_top_k)
        if ranked_indices.numel() == 0:
            return None

        return GraspCandidates(
            ranked_indices=ranked_indices,
            scores=scores,
            widths=widths,
            grasp_pos_env=grasp_pos_env,
            grasp_quat_env=grasp_quat_env,
            approach_world=approach_world,
            obj_pos_env=obj_pos_env,
            obj_quat=obj_quat,
        )

    def _rank_filtered_candidates(
        self,
        candidate_indices: torch.Tensor,
        scores: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        """Apply per-episode blocklist + candidate cooldowns, then score-rank."""
        if self._selector.blocked_candidates:
            blocked_tensor = torch.tensor(
                sorted(self._selector.blocked_candidates),
                dtype=candidate_indices.dtype,
                device=candidate_indices.device,
            )
            keep_mask = ~torch.isin(candidate_indices, blocked_tensor)
            survived = candidate_indices[keep_mask]
            if survived.numel() == 0:
                manaprint(
                    f"[{self.cfg.skill_type}] all {int(candidate_indices.numel())} filtered candidates "
                    f"are blocklisted; clearing blocklist of {len(self._selector.blocked_candidates)} to retry"
                )
                self._selector.blocked_candidates.clear()
            else:
                candidate_indices = survived
        candidate_indices = apply_candidate_cooldowns(
            state=self._selector, candidate_indices=candidate_indices, log_fn=manaprint
        )
        return rank_candidates(
            state=self._selector,
            candidate_indices=candidate_indices,
            scores=scores,
            top_k=top_k,
        )

    def plan_targets(self, observation: dict[str, Any], evaluator: Any | None) -> list[SkillTarget]:
        """Select the cheapest reachable offline grasp and emit sparse pick targets."""

        # Rank enough candidates to feed the full batched search: up to
        # reach_batch_size x reach_max_rounds distinct grasps per arm.
        batch_size = max(int(getattr(self.cfg, "reach_batch_size", 16)), 1)
        max_rounds = max(int(getattr(self.cfg, "reach_max_rounds", 3)), 1)
        rank_top_k = max(int(self.cfg.top_k), batch_size * max_rounds)
        candidates = self._select_grasp_candidates(
            observation,
            rank_top_k=rank_top_k,
            full_fallback_on_empty=True,
            refuse_when_thin=True,
        )
        if candidates is None:
            return []
        ranked_indices = candidates.ranked_indices
        scores = candidates.scores
        widths = candidates.widths
        grasp_pos_env = candidates.grasp_pos_env
        grasp_quat_env = candidates.grasp_quat_env
        approach_world = candidates.approach_world
        obj_pos_env = candidates.obj_pos_env

        ranked_grasp_pos_env = grasp_pos_env[ranked_indices]
        ranked_grasp_quat_env = grasp_quat_env[ranked_indices]
        pregrasp_pos_env = ranked_grasp_pos_env - approach_world[ranked_indices] * float(self.cfg.approach_distance)
        pregrasp_quat_env = ranked_grasp_quat_env

        arm_order = self._ranked_arms_to_object(observation, obj_pos_env)
        selected_rank = 0
        selected_path_cost = float("inf")
        if evaluator is not None:
            selection = select_reachable(
                evaluator=evaluator,
                observation=observation,
                arm_order=arm_order,
                stages=[
                    (pregrasp_pos_env, pregrasp_quat_env),
                    (ranked_grasp_pos_env, ranked_grasp_quat_env),
                ],
                batch_size=batch_size,
                max_rounds=max_rounds,
            )
            if selection is None:
                manaprint(
                    f"[pick] batch motion-gen rejected all {int(ranked_indices.numel())} ranked candidate(s) "
                    f"across arms={list(arm_order)} (batch={batch_size} rounds={max_rounds}); "
                    f"no reachable pre-grasp this tick"
                )
                try:
                    _obj_scale = observation.get("object_scale")
                    _cur = torch.as_tensor(observation["current_ee_pos"], dtype=torch.float32).reshape(3)
                    _obj_e = torch.as_tensor(obj_pos_env, dtype=torch.float32).reshape(3)
                    _pg0_env = pregrasp_pos_env[0].reshape(3)
                    _appr0 = approach_world[ranked_indices[0]].reshape(3)
                    manaprint(
                        f"[pick][diag] object={self.cfg.object_name!r} scale={_obj_scale} "
                        f"obj_pos_env={_obj_e.tolist()} "
                        f"pregrasp0_env={_pg0_env.tolist()} approach0_world={_appr0.tolist()} "
                        f"cur_ee_env={_cur.tolist()} arms={list(arm_order)}"
                    )
                except Exception as _e:  # diagnostic only; never break planning
                    manaprint(f"[pick][diag] failed to dump pose diagnostics: {_e!r}")
                return []
            selected_arm = selection.arm
            selected_rank = int(selection.index)
            selected_path_cost = selection.path_cost
        else:
            selected_arm = arm_order[0]
            manaprint("[pick] evaluator unavailable; falling back to the top score-ranked offline grasp")

        selected_index = int(ranked_indices[selected_rank].item())
        selected_score = float(scores[selected_index].item())
        selected_width = float(widths[selected_index].item())
        selected_pre_pos = pregrasp_pos_env[selected_rank]
        selected_quat = pregrasp_quat_env[selected_rank]
        selected_grasp_pos = grasp_pos_env[selected_index]
        selected_lift_pos = selected_grasp_pos + torch.tensor(
            [0.0, 0.0, self.cfg.lift_height],
            dtype=torch.float32,
            device=selected_grasp_pos.device,
        )

        return self._build_targets(
            arm=selected_arm,
            pre_pos=selected_pre_pos,
            grasp_pos=selected_grasp_pos,
            grasp_quat=selected_quat,
            candidate_index=selected_index,
            score=selected_score,
            width=selected_width,
            path_cost=selected_path_cost,
            lift_pos=selected_lift_pos,
        )

    def _build_targets(
        self,
        *,
        arm: str,
        pre_pos: torch.Tensor,
        grasp_pos: torch.Tensor,
        grasp_quat: torch.Tensor,
        candidate_index: int,
        score: float,
        width: float,
        path_cost: float,
        lift_pos: torch.Tensor | None = None,
    ) -> list[SkillTarget]:
        selected_lift_pos = (
            lift_pos
            if lift_pos is not None
            else grasp_pos
            + torch.tensor(
                [0.0, 0.0, self.cfg.lift_height],
                dtype=torch.float32,
                device=grasp_pos.device,
            )
        )
        extras = {
            "candidate_index": int(candidate_index),
            "score": float(score),
            "width": float(width),
            "path_cost": float(path_cost),
        }
        gripper_open = float(self.cfg.gripper_open)
        gripper_close = float(self.cfg.gripper_close)
        return [
            SkillTarget(
                pos=pre_pos,
                quat=grasp_quat,
                gripper=gripper_open,
                phase="pre_grasp",
                arm=arm,
                extras=dict(extras),
            ),
            SkillTarget(
                pos=grasp_pos,
                quat=grasp_quat,
                gripper=gripper_open,
                phase="grasp",
                arm=arm,
                extras={**extras, "motion_kind": MOTION_KIND_LINEAR_CARTESIAN},
            ),
            SkillTarget(
                pos=selected_lift_pos,
                quat=grasp_quat,
                gripper=gripper_close,
                phase="lift",
                arm=arm,
                extras={**extras, "motion_kind": MOTION_KIND_LINEAR_CARTESIAN},
            ),
        ]

    def _load_shape_metadata(self) -> ShapeMetadata | None:
        if not self._grasp_data.shape_metadata_loaded:
            load_manifest_payload(cfg=self.cfg, data=self._grasp_data)
        return self._grasp_data.shape_metadata


register_skill("pick", PickSkill)

__all__ = ["PickSkill", "PickSkillCfg"]
