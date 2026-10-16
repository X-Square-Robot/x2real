"""Camera and state terms adapted from the dev-6.0 manual recorder."""

from __future__ import annotations

from typing import Any

import torch
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass


class PreStepCameraObservationsRecorder(RecorderTerm):
    """Records the ``camera_obs`` observation group each step.

    IsaacLab's built-in ``PreStepFlatPolicyObservationsRecorder`` only
    captures ``obs_buf["policy"]``. This term captures ``camera_obs`` so
    camera RGB images end up in the HDF5 alongside state data.

    Images are clipped to [0, 255] and stored as uint8 to keep file
    sizes reasonable (~0.9 MB/frame/camera vs ~3.7 MB at float32).

    Depth and semantic observations retain their numeric types in HDF5.
    """

    requires_rendered_camera_frame = True
    camera_action_alignment_phase = "pre_step"

    @staticmethod
    def _to_record_value(name: str, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 3:
            return tensor
        key = str(name)
        if "semantic_segmentation" in key:
            return tensor.to(torch.int32)
        if "distance_to" in key or key.endswith("_depth") or key == "chassis_cam":
            return tensor
        return tensor.clamp(0, 255).to(torch.uint8)

    def record_pre_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if obs_buf is None:
            return (None, None)
        camera_data = obs_buf.get("camera_obs")
        if camera_data is None:
            return (None, None)
        if isinstance(camera_data, dict):
            out = {}
            for name, tensor in camera_data.items():
                out[name] = self._to_record_value(name, tensor)
            if not out:
                return (None, None)
            return ("camera_obs", out)
        if isinstance(camera_data, torch.Tensor):
            return ("camera_obs", camera_data.clamp(0, 255).to(torch.uint8))
        return (None, None)


class PreStepPolicyObservationsRecorder(RecorderTerm):
    """Record policy observations under ``obs/*`` in the HDF5 episode payload.

    Reads directly from ``obs_buf["policy"]`` (populated by the previous
    step's ``observation_manager.compute(update_history=True)``). The scene
    state has not changed between that compute and this recorder call, so
    the cached dict is semantically identical to a fresh ``compute_group``.

    Tensors are cloned to prevent the next ``compute()`` from overwriting
    the recorded snapshot.
    """

    def record_pre_step(self):
        obs_buf = getattr(self._env, "obs_buf", None)
        if not isinstance(obs_buf, dict):
            return (None, None)
        policy_obs = obs_buf.get("policy")
        if isinstance(policy_obs, dict):
            return ("obs", {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in policy_obs.items()})
        if isinstance(policy_obs, torch.Tensor):
            return ("obs_flat", policy_obs.clone())
        return (None, None)


@configclass
class PreStepCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepCameraObservationsRecorder


@configclass
class PreStepPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for structured policy observation recorder."""

    class_type: type[RecorderTerm] = PreStepPolicyObservationsRecorder


@configclass
class ManualRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Record aligned actions, structured observations, state, and camera frames."""

    record_pre_step_flat_policy_observations = None
    record_policy = PreStepPolicyObservationsRecorderCfg()
    record_camera = PreStepCameraObservationsRecorderCfg()


def configure_collection_recorder(env_cfg: Any, *, output_path: str) -> None:
    from pathlib import Path

    path = Path(output_path)
    env_cfg.recorders = ManualRecorderManagerCfg(
        dataset_export_dir_path=str(path.parent),
        dataset_filename=path.stem,
        dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
        export_in_record_pre_reset=False,
        export_in_close=False,
        dataset_compression=False,
    )
    # The collection loop evaluates task success once per tick, before exporting.
    # Keeping automatic success resets would discard the last observation first.
    if getattr(env_cfg, "terminations", None) is not None:
        env_cfg.terminations.success = None
    env_cfg.recorder_obs_refresh = False
