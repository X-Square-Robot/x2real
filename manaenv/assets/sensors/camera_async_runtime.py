"""Host scheduling and CUDA staging primitives for asynchronous cameras.

The module intentionally has no Isaac Sim imports so its scheduling contract can
be covered by the CPU stability suite.  The camera integration lives in
``async_camera.py``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

_STREAM_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _CudaStreams:
    copy: torch.cuda.Stream
    process: torch.cuda.Stream
    process_warp: wp.Stream


_STREAM_CACHE: dict[int, _CudaStreams] = {}


def _cuda_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"Camera async runtime requires a CUDA device, got {resolved}")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _get_cuda_streams(device: torch.device) -> _CudaStreams:
    index = int(device.index)
    with _STREAM_CACHE_LOCK:
        streams = _STREAM_CACHE.get(index)
        if streams is None:
            wp.init()
            process_stream = torch.cuda.Stream(device=index)
            streams = _CudaStreams(
                copy=torch.cuda.Stream(device=index),
                process=process_stream,
                # Keep the Warp wrapper alive beside the cached Torch stream;
                # repeated registration of the same CUDA handle is unsafe.
                process_warp=wp.stream_from_torch(process_stream),
            )
            _STREAM_CACHE[index] = streams
        return streams


class HostDirtyTracker:
    """CPU mirror of the SensorBase timestamp/outdated Warp kernels.

    Keeping this small state on the host removes the hot-path
    ``env_mask.numpy()`` device-to-host synchronization while the Warp arrays
    remain authoritative for the per-environment camera updates themselves.
    """

    _EPSILON = np.float32(1.0e-6)

    def __init__(self, *, num_envs: int, update_period: float) -> None:
        if num_envs <= 0:
            raise ValueError(f"HostDirtyTracker.num_envs must be positive, got {num_envs}")
        self._num_envs = int(num_envs)
        self._update_period = np.float32(update_period)
        self._timestamp = np.zeros(self._num_envs, dtype=np.float32)
        self._timestamp_last_update = np.zeros(self._num_envs, dtype=np.float32)
        self._dirty = np.ones(self._num_envs, dtype=np.bool_)

    @property
    def dirty_mask(self) -> np.ndarray:
        """Return a copy of the host dirty mask for diagnostics and tests."""
        return self._dirty.copy()

    def has_any(self) -> bool:
        """Return whether at least one camera instance needs an update."""
        return bool(self._dirty.any())

    def update(self, dt: float) -> None:
        """Mirror ``update_timestamp_kernel`` using float32 arithmetic."""
        self._timestamp += np.float32(dt)
        elapsed = self._timestamp - self._timestamp_last_update
        self._dirty |= elapsed + self._EPSILON >= self._update_period

    def reset(
        self,
        env_ids: Sequence[int] | np.ndarray | None = None,
        env_mask: np.ndarray | None = None,
    ) -> None:
        """Mirror ``reset_envs_kernel`` for all, indexed, or masked resets."""
        if env_mask is not None:
            mask = np.asarray(env_mask, dtype=np.bool_).reshape(-1)
            if mask.shape != (self._num_envs,):
                raise ValueError(f"HostDirtyTracker env_mask must have shape ({self._num_envs},), got {mask.shape}")
        elif env_ids is None:
            mask = np.ones(self._num_envs, dtype=np.bool_)
        else:
            indices = np.asarray(env_ids, dtype=np.int64).reshape(-1)
            mask = np.zeros(self._num_envs, dtype=np.bool_)
            mask[indices] = True

        self._timestamp[mask] = np.float32(0.0)
        self._timestamp_last_update[mask] = np.float32(0.0)
        self._dirty[mask] = True

    def consume(self) -> None:
        """Mirror ``update_outdated_envs_kernel`` after a successful read."""
        dirty = self._dirty
        self._timestamp_last_update[dirty] = self._timestamp[dirty]
        dirty[dirty] = False


@dataclass(frozen=True)
class CudaSubmission:
    """Events and slot identity for one double-buffer submission."""

    slot_index: int
    source_ready_event: torch.cuda.Event
    copy_done_event: torch.cuda.Event
    ready_event: torch.cuda.Event


@dataclass
class _BufferSlot:
    tensor: torch.Tensor | None = None
    ready_event: torch.cuda.Event | None = None


class CudaDoubleBuffer:
    """Owned two-slot CUDA staging buffer with explicit stream dependencies.

    Submission order is:

    ``producer stream -> source_ready -> copy stream -> copy_done -> process
    stream -> ready -> consumer stream``.

    The caller must invoke :meth:`synchronize_source_copies` before asking the
    external producer to recycle its buffers.  That synchronization is
    intentionally explicit; deleting it would make Replicator's
    ``do_array_copy=False`` output lifetime unsafe.
    """

    def __init__(self, device: torch.device | str) -> None:
        self._device = _cuda_device(device)
        self._streams = _get_cuda_streams(self._device)
        self._slots = (_BufferSlot(), _BufferSlot())
        self._next_slot = 0
        self._pending_source_copies: list[tuple[torch.cuda.Event, torch.Tensor]] = []

    @property
    def pending_source_copies(self) -> int:
        return len(self._pending_source_copies)

    @property
    def process_stream(self) -> torch.cuda.Stream:
        """Stream carrying post-copy work; shared by every buffer on this device.

        Callers that must run after every submitted ``work`` — a whole-frame ISP
        pass, for instance — enqueue onto this stream and inherit that ordering
        without an extra event per buffer.
        """
        return self._streams.process

    @property
    def process_warp_stream(self) -> wp.Stream:
        """Persistent explicit Warp wrapper for :attr:`process_stream`."""
        return self._streams.process_warp

    def submit(
        self,
        source: torch.Tensor,
        work: Callable[[torch.Tensor, torch.cuda.Stream], Any],
        *,
        source_stream: torch.cuda.Stream | None = None,
    ) -> CudaSubmission:
        """Copy ``source`` into the next owned slot and enqueue ``work``."""
        if not isinstance(source, torch.Tensor) or not source.is_cuda:
            raise TypeError("CudaDoubleBuffer.submit source must be a CUDA torch.Tensor")
        if source.device != self._device:
            raise ValueError(f"source device {source.device} does not match buffer device {self._device}")
        if not source.is_contiguous():
            source = source.contiguous()

        slot_index = self._next_slot
        self._next_slot = (slot_index + 1) % len(self._slots)
        slot = self._slots[slot_index]

        needs_allocation = (
            slot.tensor is None
            or slot.tensor.shape != source.shape
            or slot.tensor.dtype != source.dtype
            or slot.tensor.device != source.device
        )
        if needs_allocation:
            if slot.ready_event is not None:
                slot.ready_event.synchronize()
            slot.tensor = torch.empty(source.shape, dtype=source.dtype, device=source.device)

        consumer_stream = torch.cuda.current_stream(self._device)
        producer_stream = source_stream or consumer_stream
        source_ready = torch.cuda.Event()
        source_ready.record(producer_stream)

        copy_stream = self._streams.copy
        copy_stream.wait_event(source_ready)
        if slot.ready_event is not None:
            copy_stream.wait_event(slot.ready_event)
        with torch.cuda.stream(copy_stream):
            slot.tensor.copy_(source, non_blocking=True)
        copy_done = torch.cuda.Event()
        copy_done.record(copy_stream)

        process_stream = self._streams.process
        process_stream.wait_event(copy_done)
        with torch.cuda.stream(process_stream):
            work(slot.tensor, process_stream)
        ready = torch.cuda.Event()
        ready.record(process_stream)

        # ObservationManager's final clone is launched on the producer/main
        # stream.  Waiting on an event orders that clone without blocking the
        # Python thread.
        consumer_stream.wait_event(ready)

        slot.ready_event = ready
        self._pending_source_copies.append((copy_done, source))
        return CudaSubmission(slot_index, source_ready, copy_done, ready)

    def synchronize_source_copies(self) -> None:
        """Wait until external producer buffers are no longer referenced."""
        pending, self._pending_source_copies = self._pending_source_copies, []
        for event, _source in pending:
            event.synchronize()

    def close(self) -> None:
        """Finish in-flight work before releasing owned staging storage."""
        self.synchronize_source_copies()
        for slot in self._slots:
            if slot.ready_event is not None:
                slot.ready_event.synchronize()
            slot.tensor = None
            slot.ready_event = None


def copy_warp_bool_mask_to_host(env_mask: Any) -> np.ndarray:
    """Copy one Warp bool mask to host through an explicit CUDA event.

    This is used only for the uncommon ``Camera.reset(env_mask=...)`` API.  The
    per-frame dirty check uses :class:`HostDirtyTracker` and performs no D2H.
    """
    mask_torch = wp.to_torch(env_mask).reshape(-1)
    if not mask_torch.is_cuda:
        return mask_torch.to(dtype=torch.bool).numpy().copy()

    device = _cuda_device(mask_torch.device)
    streams = _get_cuda_streams(device)
    host_mask = torch.empty(mask_torch.shape, dtype=torch.bool, device="cpu", pin_memory=True)

    # Record readiness on Warp's active stream, then let the shared copy stream
    # perform the single batched D2H transfer.  The host wait is explicit and
    # confined to reset instead of occurring on every camera observation.
    warp_stream = wp.get_stream(env_mask.device)
    producer_stream = wp.stream_to_torch(warp_stream)
    source_ready = torch.cuda.Event()
    source_ready.record(producer_stream)
    streams.copy.wait_event(source_ready)
    with torch.cuda.stream(streams.copy):
        host_mask.copy_(mask_torch, non_blocking=True)
    copy_done = torch.cuda.Event()
    copy_done.record(streams.copy)
    copy_done.synchronize()
    return host_mask.numpy().copy()


__all__ = [
    "CudaDoubleBuffer",
    "CudaSubmission",
    "HostDirtyTracker",
    "copy_warp_bool_mask_to_host",
]
