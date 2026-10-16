"""Bounded recorder staging for public manual collection.

IsaacLab's default :class:`EpisodeData` keeps cloned tensors on the simulation
device until an episode is exported. Manual collection records camera frames,
actions and state terms, so that behaviour makes recorder memory grow with the
episode length. This module keeps the upstream episode schema while routing
CUDA terms through bounded GPU chunks and a shared D2H stream.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch


_D2H_STREAMS: dict[int, torch.cuda.Stream] = {}
_D2H_STREAMS_LOCK = threading.Lock()
_STAGED_CLASS: type | None = None


def _d2h_stream(device: torch.device) -> torch.cuda.Stream | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _D2H_STREAMS_LOCK:
        stream = _D2H_STREAMS.get(index)
        if stream is None:
            stream = torch.cuda.Stream(device=index)
            _D2H_STREAMS[index] = stream
        return stream


class ChunkedStager:
    """Per-term GPU chunks with batched D2H copies on a dedicated stream."""

    def __init__(self, *, chunk_size: int, pinned_memory: bool) -> None:
        if chunk_size <= 0:
            raise ValueError(f"ChunkedStager.chunk_size must be > 0 (got {chunk_size})")
        self.chunk_size = int(chunk_size)
        self.pinned_memory = bool(pinned_memory)
        self._gpu: dict[str, torch.Tensor] = {}
        self._positions: dict[str, int] = {}
        self._cpu_chunks: dict[str, list[torch.Tensor]] = {}
        self._events: dict[str, list[torch.cuda.Event]] = {}

    def add(self, key: str, value: torch.Tensor) -> None:
        buffer = self._gpu.get(key)
        if buffer is None:
            buffer = torch.empty(
                (self.chunk_size, *value.shape),
                dtype=value.dtype,
                device=value.device,
            )
            self._gpu[key] = buffer
            self._positions[key] = 0
        position = self._positions[key]
        buffer[position].copy_(value)
        position += 1
        self._positions[key] = position
        if position == self.chunk_size:
            self._flush(key, position)

    def count(self, key: str) -> int:
        total = self._positions.get(key, 0)
        total += sum(int(chunk.shape[0]) for chunk in self._cpu_chunks.get(key, ()))
        return int(total)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._gpu) | set(self._cpu_chunks)))

    def drain(self, key: str) -> torch.Tensor | None:
        self._flush(key, self._positions.get(key, 0))
        for event in self._events.pop(key, ()):
            event.synchronize()
        chunks = self._cpu_chunks.pop(key, [])
        self._gpu.pop(key, None)
        self._positions.pop(key, None)
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

    def reset(self) -> None:
        self._gpu.clear()
        self._positions.clear()
        self._cpu_chunks.clear()
        self._events.clear()

    def _flush(self, key: str, count: int) -> None:
        if count <= 0:
            return
        view = self._gpu[key][:count]
        stream = _d2h_stream(view.device)
        if stream is None:
            host = view.detach().to(device="cpu", copy=True).contiguous()
            self._cpu_chunks.setdefault(key, []).append(host)
            self._positions[key] = 0
            return
        try:
            host = torch.empty(
                (count, *view.shape[1:]),
                dtype=view.dtype,
                pin_memory=self.pinned_memory,
            )
        except RuntimeError:
            host = torch.empty((count, *view.shape[1:]), dtype=view.dtype)
        producer = torch.cuda.current_stream(view.device)
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            host.copy_(view, non_blocking=self.pinned_memory)
        event = torch.cuda.Event()
        event.record(stream)
        self._cpu_chunks.setdefault(key, []).append(host)
        self._events.setdefault(key, []).append(event)
        self._positions[key] = 0


def _stack_lists(data: dict[str, Any]) -> None:
    for key, value in list(data.items()):
        if isinstance(value, list):
            data[key] = torch.stack(value)
        elif isinstance(value, dict):
            _stack_lists(value)


def _set_path(data: dict[str, Any], key: str, value: Any) -> None:
    cursor = data
    parts = key.split("/")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def _append_path(data: dict[str, Any], key: str, value: Any) -> None:
    cursor = data
    parts = key.split("/")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor.setdefault(parts[-1], []).append(value)


def _staged_episode_class() -> type:
    global _STAGED_CLASS
    if _STAGED_CLASS is not None:
        return _STAGED_CLASS
    from isaaclab.utils.datasets.episode_data import EpisodeData

    class StagedEpisodeData(EpisodeData):
        """Drop-in EpisodeData with bounded CUDA recorder buffers."""

        def __init__(self, *, chunk_size: int, pinned_memory: bool) -> None:
            super().__init__()
            self._stager = ChunkedStager(chunk_size=chunk_size, pinned_memory=pinned_memory)
            self._stager_keys: set[str] = set()

        def add(self, key: str, value: Any, clone: bool = True):  # type: ignore[override]
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.add(f"{key}/{sub_key}", sub_value, clone=clone)
                return
            if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                self._stager.add(key, value)
                self._stager_keys.add(key)
                return
            if isinstance(value, torch.Tensor):
                value = value.detach().clone().contiguous() if clone else value.detach().contiguous()
            _append_path(self._data, key, value)

        def is_empty(self):  # type: ignore[override]
            return not self._data and not self._stager_keys

        def recorded_count(self, key: str) -> int:
            current: Any = self._data
            for part in key.split("/"):
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(part)
            data_count = len(current) if isinstance(current, list) else 0
            if isinstance(current, torch.Tensor) and current.ndim:
                data_count = int(current.shape[0])
            return max(data_count, self._stager.count(key))

        def pre_export(self):  # type: ignore[override]
            for key in sorted(self._stager_keys):
                value = self._stager.drain(key)
                if value is not None:
                    _set_path(self._data, key, value)
            self._stager_keys.clear()
            self._stager.reset()
            _stack_lists(self._data)

    StagedEpisodeData.__name__ = "StagedEpisodeData"
    _STAGED_CLASS = StagedEpisodeData
    return StagedEpisodeData


def _as_env_ids(env_ids: Any, recorder_manager: Any) -> list[int]:
    if env_ids is None:
        return list(range(int(recorder_manager._env.num_envs)))
    if hasattr(env_ids, "tolist"):
        env_ids = env_ids.tolist()
    if isinstance(env_ids, int):
        return [env_ids]
    return [int(value) for value in env_ids]


def install_collection_staging(recorder_manager: Any) -> bool:
    """Install bounded staging on one live RecorderManager."""
    enabled = os.environ.get("MANAENV_COLLECTION_STAGING", "1").strip().lower()
    if enabled in {"0", "false", "off", "no"}:
        return False
    if recorder_manager is None or not getattr(recorder_manager, "active_terms", None):
        return False
    episodes = getattr(recorder_manager, "_episodes", None)
    if not isinstance(episodes, dict):
        return False

    chunk_size = max(int(os.environ.get("MANAENV_COLLECTION_STAGING_CHUNK", "32")), 1)
    pinned = os.environ.get("MANAENV_COLLECTION_STAGING_PINNED", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
    staged_cls = _staged_episode_class()

    def make_episode(env_id: int, old: Any = None):
        episode = staged_cls(chunk_size=chunk_size, pinned_memory=pinned)
        episode._env_id = int(env_id)
        if old is not None:
            episode._seed = getattr(old, "_seed", None)
            episode._success = getattr(old, "_success", None)
        return episode

    for env_id, old in list(episodes.items()):
        episodes[env_id] = make_episode(int(env_id), old)

    original_reset = recorder_manager.reset

    def reset(env_ids=None, _original=original_reset):
        target = _as_env_ids(env_ids, recorder_manager)
        result = _original(env_ids=env_ids)
        for env_id in target:
            episodes[int(env_id)] = make_episode(int(env_id))
        return result

    def add_to_episodes(key, value, env_ids=None):
        if key is None or not recorder_manager.active_terms:
            return
        targets = _as_env_ids(env_ids, recorder_manager)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                add_to_episodes(f"{key}/{sub_key}", sub_value, targets)
            return
        try:
            import warp as wp

            if isinstance(value, wp.array):
                value = wp.to_torch(value)
        except Exception:
            pass
        for index, env_id in enumerate(targets):
            episode = episodes.get(int(env_id))
            if episode is None or not isinstance(episode, staged_cls):
                episode = make_episode(int(env_id))
                episodes[int(env_id)] = episode
            episode.add(key, value[index], clone=False)

    recorder_manager.reset = reset
    recorder_manager.add_to_episodes = add_to_episodes
    recorder_manager._mana_collection_staging = True
    from manaenv.utils.debug_print import manaprint

    manaprint(
        "INFO: collection recorder staging installed "
        f"(chunk_size={chunk_size}, pinned_memory={pinned}, all CUDA terms)"
    )
    return True


__all__ = ["ChunkedStager", "install_collection_staging"]
