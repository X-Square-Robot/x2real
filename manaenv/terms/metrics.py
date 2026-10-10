"""Evaluation metric builders used by the public task suite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase

from manaenv.utils.asset_utils import AssetNameProxy, resolve_active_scene_entity_name
from manaenv.utils.tensor_utils import as_torch

if TYPE_CHECKING:
    from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg


class ActiveObjectVelocityRecorder(RecorderTerm):
    """Record the active runtime slot's linear velocity."""

    name = "active_object_linear_velocity"

    def __init__(self, cfg: RecorderTermCfg, env: Any):
        super().__init__(cfg, env)
        self.object_name = cfg.object_name

    def record_post_step(self):
        import torch

        velocity = torch.zeros((self._env.num_envs, 3), dtype=torch.float32, device=self._env.device)
        for env_id in range(self._env.num_envs):
            active_name = resolve_active_scene_entity_name(self._env, self.object_name, env_id=env_id)
            asset = self._env.scene[active_name]
            velocity[env_id] = as_torch(asset.data.root_link_vel_w)[env_id, :3]
        return self.name, velocity


@configclass
class ActiveObjectVelocityRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`ActiveObjectVelocityRecorder`."""

    class_type: type[RecorderTerm] = ActiveObjectVelocityRecorder
    object_name: str = ""


class ActiveObjectMovedRateMetric(MetricBase):
    """Measure whether the selected runtime object moved."""

    name = "object_moved_rate"
    recorder_term_name = ActiveObjectVelocityRecorder.name

    def __init__(self, object_name: str, object_velocity_threshold: float = 0.5):
        self.object_name = object_name
        self.object_velocity_threshold = object_velocity_threshold

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return ActiveObjectVelocityRecorderCfg(object_name=self.object_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
        from isaaclab_arena.metrics.object_moved import compute_object_moved_rate

        return MetricTermCfg(
            compute_metric_func=compute_object_moved_rate,
            params={"object_velocity_threshold": self.object_velocity_threshold},
            recorder_term_name=self.recorder_term_name,
        )


def success_rate(**_kwargs) -> MetricBase:
    """Create the task success-rate metric."""
    from isaaclab_arena.metrics.success_rate import SuccessRateMetric

    return SuccessRateMetric()


def object_moved_rate(object_cfg: Any = None, object_velocity_threshold: float = 0.5, **_kwargs) -> MetricBase:
    """Create a moved-rate metric for a logical object or active pool slot."""
    if isinstance(object_cfg, list):
        object_cfg = object_cfg[0] if object_cfg else None
    if isinstance(object_cfg, str):
        object_name = object_cfg
    elif isinstance(object_cfg, AssetNameProxy):
        object_name = object_cfg.name
    else:
        object_name = str(getattr(object_cfg, "name", object_cfg or ""))
    return ActiveObjectMovedRateMetric(
        object_name=object_name,
        object_velocity_threshold=object_velocity_threshold,
    )


__all__ = ["object_moved_rate", "success_rate"]
