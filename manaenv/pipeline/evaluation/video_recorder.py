"""Episode-scoped benchmark video recording through IsaacLab RecorderManager.

This module intentionally records only RGB video. It does not expose dataset,
replay, teleoperation, or demonstration-export surfaces in the public runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from manaenv.utils.debug_print import manaprint

_EPISODE_VIDEO_RE = re.compile(r"^episode_(\d{6})(?:_env_\d+)?\.mp4$")
_VIEWING_ORDER_HINTS = ("left", "head", "right")
_DATASET_RECORDER_TERM_NAMES = frozenset(
    {
        "record_initial_state",
        "record_post_step_states",
        "record_pre_step_actions",
        "record_pre_step_flat_policy_observations",
        "record_post_step_processed_actions",
        "record_pre_step_flat_camera_observations",
        "record_post_step_flat_policy_action_observations",
        "record_camera_obs",
    }
)


@dataclass
class _OpenEpisodeVideo:
    writer: Any
    partial_path: Path
    final_path: Path
    frame_size: tuple[int, int]
    frame_count: int = 0


def _normalize_env_ids(env_ids: Any, num_envs: int) -> tuple[int, ...]:
    if env_ids is None:
        return tuple(range(num_envs))
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu()
    if hasattr(env_ids, "tolist"):
        env_ids = env_ids.tolist()
    if isinstance(env_ids, int):
        return (int(env_ids),)
    return tuple(int(env_id) for env_id in env_ids)


def _viewing_rank(name: str) -> int:
    segments = set(str(name).lower().split("_"))
    for index, hint in enumerate(_VIEWING_ORDER_HINTS):
        if hint in segments:
            return index
    return len(_VIEWING_ORDER_HINTS)


def _in_viewing_order(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(names, key=lambda name: (_viewing_rank(name), names.index(name))))


def _rgb_observation_terms(camera_obs: Any) -> tuple[str, ...]:
    if camera_obs is None or not is_dataclass(camera_obs):
        return ()
    names: list[str] = []
    for field_info in fields(camera_obs):
        term = getattr(camera_obs, field_info.name, None)
        params = getattr(term, "params", None)
        if isinstance(params, dict) and params.get("data_type") == "rgb":
            names.append(field_info.name)
    return tuple(names)


def camera_observation_order(environment: Any) -> tuple[str, ...]:
    """Return the live camera observation keys in left, head, right order."""

    embodiment = getattr(environment, "embodiment", None)
    camera_obs_cfg = getattr(embodiment, "_camera_observation_config", None)
    names = _rgb_observation_terms(getattr(camera_obs_cfg, "camera_obs", None))
    if names:
        return _in_viewing_order(names)

    camera_cfg = getattr(embodiment, "camera_config", None)
    if camera_cfg is not None and is_dataclass(camera_cfg):
        camera_fields = tuple(
            field_info.name
            for field_info in fields(camera_cfg)
            if hasattr(getattr(camera_cfg, field_info.name, None), "prim_path")
        )
        if camera_fields:
            return _in_viewing_order(tuple(f"{name}_rgb" for name in camera_fields))
    return ()


def resolve_video_fps(environment: Any, fallback: float = 20.0) -> float:
    """Resolve the rendered-observation rate used for MP4 playback."""

    scene = getattr(environment, "scene", None)
    sim_fps = getattr(scene, "sim_fps", None)
    render_decremental = getattr(scene, "render_decremental", None)
    if sim_fps and render_decremental:
        fps = float(sim_fps) / float(render_decremental)
        if fps > 0:
            return fps
    return float(fallback)


def prepare_environment_for_video(environment: Any) -> None:
    """Ensure the embodiment exposes its authored camera observations."""

    embodiment = getattr(environment, "embodiment", None)
    if embodiment is None:
        return
    if hasattr(embodiment, "enable_cameras"):
        embodiment.enable_cameras = True
    if hasattr(embodiment, "camera_config") and getattr(embodiment, "camera_config", None) is None:
        camera_cfg_cls = getattr(embodiment, "CameraCfg", None)
        if camera_cfg_cls is not None:
            embodiment.camera_config = camera_cfg_cls()


def list_episode_videos(output_dir: str | Path) -> list[str]:
    """List finalized episode videos, excluding in-progress files."""

    root = Path(output_dir).expanduser()
    if not root.is_dir():
        return []
    return [str(path) for path in sorted(root.glob("episode_*.mp4")) if _EPISODE_VIDEO_RE.match(path.name)]


class BenchmarkEpisodeVideoRecorder(RecorderTerm):
    """Stream one merged RGB MP4 for each completed vectorized environment episode."""

    # ManaPhysxAsyncKitAsyncEnv delays camera-consuming recorder terms until the
    # submitted render has completed and camera observations have been refreshed.
    requires_rendered_camera_frame = True

    def __init__(self, cfg: BenchmarkEpisodeVideoRecorderCfg, env: Any) -> None:
        super().__init__(cfg, env)
        self.output_dir = Path(cfg.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(cfg.fps)
        if self.fps <= 0:
            raise ValueError(f"benchmark video fps must be positive, got {self.fps}")
        self.camera_order = tuple(str(name) for name in cfg.camera_order if str(name))
        self.codec = str(cfg.codec or "mp4v")
        if len(self.codec) != 4:
            raise ValueError(f"benchmark video codec must contain four characters, got {self.codec!r}")
        self._open: dict[int, _OpenEpisodeVideo] = {}
        self._next_episode_index = self._find_next_episode_index()

    def _find_next_episode_index(self) -> int:
        indices = []
        for path in self.output_dir.glob("episode_*.mp4"):
            match = _EPISODE_VIDEO_RE.match(path.name)
            if match is not None:
                indices.append(int(match.group(1)))
        return max(indices, default=-1) + 1

    @staticmethod
    def _as_rgb_batch(value: Any) -> np.ndarray | None:
        value = getattr(value, "torch", value)
        if not isinstance(value, torch.Tensor) or value.ndim not in {3, 4} or int(value.shape[-1]) != 3:
            return None
        if value.ndim == 3:
            value = value.unsqueeze(0)
        value = value.detach().to(device="cpu")
        if value.is_floating_point() and value.numel() and float(value.max()) <= 1.0:
            value = value * 255.0
        return value.clamp(0, 255).to(dtype=torch.uint8).numpy()

    @staticmethod
    def _pad_batch(batch: np.ndarray, height: int, width: int) -> np.ndarray:
        if batch.shape[1:3] == (height, width):
            return batch
        output = np.zeros((batch.shape[0], height, width, 3), dtype=np.uint8)
        y0 = max((height - int(batch.shape[1])) // 2, 0)
        x0 = max((width - int(batch.shape[2])) // 2, 0)
        output[:, y0 : y0 + batch.shape[1], x0 : x0 + batch.shape[2], :] = batch
        return output

    def _merged_camera_batch(self) -> np.ndarray:
        obs_buf = getattr(self._env, "obs_buf", None)
        camera_data = obs_buf.get("camera_obs") if isinstance(obs_buf, dict) else None
        if not isinstance(camera_data, dict):
            raise RuntimeError("benchmark recorder requires obs_buf['camera_obs']")

        camera_order = self.camera_order
        if not camera_order:
            camera_order = _in_viewing_order(
                tuple(name for name, value in camera_data.items() if self._as_rgb_batch(value) is not None)
            )

        batches = [self._as_rgb_batch(camera_data.get(name)) for name in camera_order]
        available = [batch for batch in batches if batch is not None]
        if not available:
            raise RuntimeError(
                "benchmark recorder matched no RGB camera observations; "
                f"configured={list(camera_order)}, available={sorted(camera_data)}"
            )

        batch_size = int(available[0].shape[0])
        if batch_size != int(getattr(self._env, "num_envs", batch_size)):
            raise RuntimeError(
                f"benchmark recorder camera batch has {batch_size} envs; expected {self._env.num_envs}"
            )
        height = max(int(batch.shape[1]) for batch in available)
        width = max(int(batch.shape[2]) for batch in available)
        panels = []
        for batch in batches:
            if batch is None:
                panels.append(np.zeros((batch_size, height, width, 3), dtype=np.uint8))
                continue
            if int(batch.shape[0]) != batch_size:
                raise RuntimeError("benchmark recorder camera observations have inconsistent batch sizes")
            panels.append(self._pad_batch(batch, height, width))
        return np.concatenate(panels, axis=2)

    def _reserve_paths(self, env_id: int) -> tuple[Path, Path]:
        while True:
            index = self._next_episode_index
            self._next_episode_index += 1
            env_suffix = f"_env_{env_id:02d}" if int(getattr(self._env, "num_envs", 1)) > 1 else ""
            final_path = self.output_dir / f"episode_{index:06d}{env_suffix}.mp4"
            partial_path = self.output_dir / f".episode_{index:06d}{env_suffix}.partial.mp4"
            if not final_path.exists() and not partial_path.exists():
                return partial_path, final_path

    def _open_writer(self, env_id: int, frame: np.ndarray) -> _OpenEpisodeVideo:
        partial_path, final_path = self._reserve_paths(env_id)
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(partial_path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            try:
                partial_path.unlink()
            except FileNotFoundError:
                pass
            raise RuntimeError(
                f"failed to open benchmark MP4 encoder codec={self.codec!r} path={partial_path}"
            )
        opened = _OpenEpisodeVideo(
            writer=writer,
            partial_path=partial_path,
            final_path=final_path,
            frame_size=(width, height),
        )
        self._open[env_id] = opened
        return opened

    def record_pre_step(self) -> tuple[None, None]:
        merged = self._merged_camera_batch()
        for env_id, rgb_frame in enumerate(merged):
            opened = self._open.get(env_id) or self._open_writer(env_id, rgb_frame)
            height, width = rgb_frame.shape[:2]
            if opened.frame_size != (width, height):
                raise RuntimeError(
                    "benchmark recorder frame size changed within an episode: "
                    f"{opened.frame_size} -> {(width, height)}"
                )
            opened.writer.write(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            opened.frame_count += 1
        return None, None

    def _finish(self, env_id: int, *, completed: bool) -> None:
        opened = self._open.pop(env_id, None)
        if opened is None:
            return
        opened.writer.release()
        if completed and opened.frame_count > 0 and opened.partial_path.is_file():
            opened.partial_path.replace(opened.final_path)
            manaprint(
                f"INFO: [BenchmarkRecorder] video -> {opened.final_path} "
                f"({opened.frame_count} frames at {self.fps:g} FPS)"
            )
            return
        try:
            opened.partial_path.unlink()
        except FileNotFoundError:
            pass

    def record_pre_reset(self, env_ids: Any) -> tuple[None, None]:
        for env_id in _normalize_env_ids(env_ids, int(self._env.num_envs)):
            self._finish(env_id, completed=True)
        return None, None

    def reset(self, env_ids: Any = None) -> None:
        for env_id in _normalize_env_ids(env_ids, int(self._env.num_envs)):
            self._finish(env_id, completed=False)

    def close(self, file_path: str) -> None:
        del file_path
        for env_id in tuple(self._open):
            self._finish(env_id, completed=False)


@configclass
class BenchmarkEpisodeVideoRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`BenchmarkEpisodeVideoRecorder`."""

    class_type: type[RecorderTerm] = BenchmarkEpisodeVideoRecorder
    output_dir: str = "videos/combined"
    fps: float = 20.0
    camera_order: tuple[str, ...] = tuple()
    codec: str = "mp4v"


def configure_benchmark_video_recorder(
    env_cfg: Any,
    *,
    output_dir: str,
    fps: float,
    camera_order: tuple[str, ...],
) -> None:
    """Attach the public MP4 recorder term without enabling dataset export."""

    if getattr(env_cfg, "recorders", None) is None:
        env_cfg.recorders = RecorderManagerBaseCfg()
    disabled_terms = []
    for name in _DATASET_RECORDER_TERM_NAMES:
        if getattr(env_cfg.recorders, name, None) is not None:
            setattr(env_cfg.recorders, name, None)
            disabled_terms.append(name)
    env_cfg.recorders.record_benchmark_video = BenchmarkEpisodeVideoRecorderCfg(
        output_dir=str(output_dir),
        fps=float(fps),
        camera_order=tuple(camera_order),
    )
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env_cfg.recorders.export_in_record_pre_reset = False
    env_cfg.recorders.export_in_close = False
    if disabled_terms:
        manaprint(
            "INFO: [BenchmarkRecorder] disabled dataset recorder terms: "
            f"{', '.join(sorted(disabled_terms))}"
        )
    manaprint(
        f"INFO: [BenchmarkRecorder] enabled -> {Path(output_dir).expanduser()} "
        f"(fps={float(fps):g}, cameras={list(camera_order) or ['auto']})"
    )


__all__ = [
    "BenchmarkEpisodeVideoRecorder",
    "BenchmarkEpisodeVideoRecorderCfg",
    "camera_observation_order",
    "configure_benchmark_video_recorder",
    "list_episode_videos",
    "prepare_environment_for_video",
    "resolve_video_fps",
]
