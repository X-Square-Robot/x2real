"""ManaEnv camera with host dirty tracking and asynchronous RTX RGB staging."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import warp as wp
import warp.types
from isaaclab.renderers import RenderBufferKind
from isaaclab.sensors.camera.camera import Camera

from manaenv.assets.sensors.camera_async_runtime import CudaDoubleBuffer, HostDirtyTracker, copy_warp_bool_mask_to_host
from manaenv.assets.sensors.camera_async_warp import (
    apply_ppisp_on_stream,
    clamp_depth_to_inf_on_stream,
    replace_inf_depth_on_stream,
)
from manaenv.utils.camera_utils import normalize_opencv_camera_intrinsics_to_render_resolution

logger = logging.getLogger(__name__)

_RGBA = str(RenderBufferKind.RGBA)
_RGB_HDR = str(RenderBufferKind.RGB_HDR)
_CAMERA_RENDER_DEVICE = "cuda:0"

# Per-data-type post-staging fixups, mirroring ``IsaacRtxRenderer.render()``.
# Colorized segmentation arrives as uint32 and is reinterpreted as 4x uint8.
_COLORIZE_CFG_FLAG = {
    "semantic_segmentation": "colorize_semantic_segmentation",
    "instance_segmentation": "colorize_instance_segmentation",
    "instance_id_segmentation_fast": "colorize_instance_id_segmentation",
}
# Annotators whose tiled buffer carries more channels than the destination.
_LEADING_CHANNELS = {"motion_vectors": 2, "normals": 3, _RGB_HDR: 3}
_SIMPLE_SHADING_PREFIX = "simple_shading_"
_SIMPLE_SHADING_CHANNELS = 3
_DEPTH_TYPES = ("distance_to_camera", "distance_to_image_plane", "depth")


class _RenderProductFrameTicket:
    """One render-product completion expected from the next Kit app update."""

    def __init__(self, render_product_path: str, synthetic_data_interface: Any) -> None:
        self.render_product_path = render_product_path
        self._synthetic_data_interface = synthetic_data_interface
        self._ready = threading.Event()
        self._completed_swh_frame: int | None = None
        self._error: BaseException | None = None
        self._render_subscription: Any = None

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    @property
    def completed_swh_frame(self) -> int | None:
        return self._completed_swh_frame

    @classmethod
    def arm(cls, render_product_path: str) -> _RenderProductFrameTicket:
        """Subscribe before ``app.update()`` to observe its completed frame batch."""
        import carb.eventdispatcher  # noqa: PLC0415
        import omni.syntheticdata._syntheticdata as synthetic_data  # noqa: PLC0415
        import omni.usd  # noqa: PLC0415

        ticket = cls(render_product_path, synthetic_data.acquire_syntheticdata_interface())
        dispatcher = carb.eventdispatcher.get_eventdispatcher()
        context = omni.usd.get_context()
        ticket._render_subscription = dispatcher.observe_event(
            event_name=context.stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME, True),
            on_event=ticket._on_render_event,
            observer_name="manaenv.ManaAsyncCamera.external_render_new_frame",
        )
        return ticket

    @staticmethod
    def _event_value(event: Any, *names: str) -> Any:
        for name in names:
            try:
                value = event[name]
            except (KeyError, TypeError):
                continue
            if value is not None:
                return value
        return None

    def _fail(self, exc: BaseException) -> None:
        if not self._ready.is_set():
            self._error = exc
            self._ready.set()

    def _on_render_event(self, event: Any) -> None:
        if self._ready.is_set():
            return
        try:
            product_path_handle = event["product_path_handle"]
            results = event["results"]
            rendered = self._synthetic_data_interface.parse_rendered_simulation_event(
                product_path_handle,
                results,
            )
        except (KeyError, TypeError):
            return
        except Exception as exc:  # noqa: BLE001 - callback errors cross to the main thread
            error = RuntimeError("Could not parse Kit NEW_FRAME event")
            error.__cause__ = exc
            self._fail(error)
            return
        if not rendered or rendered[0] != self.render_product_path:
            return

        value = self._event_value(event, "swh_frame_number", "SWHFrameNumber", "frameNumber")
        if value is None:
            self._fail(
                RuntimeError(f"Kit NEW_FRAME event for {self.render_product_path} did not publish an SWH frame number")
            )
            return
        try:
            rendered_frame = int(value)
        except (TypeError, ValueError) as exc:
            error = RuntimeError(f"Invalid rendered SWH frame number: {value!r}")
            error.__cause__ = exc
            self._fail(error)
            return
        self._completed_swh_frame = rendered_frame
        self._ready.set()

    def require_completed(self) -> int:
        if not self._ready.is_set():
            raise RuntimeError(f"Kit render product {self.render_product_path} has not completed")
        if self._error is not None:
            raise self._error
        assert self._completed_swh_frame is not None
        return self._completed_swh_frame

    def cancel(self) -> None:
        """Release dispatcher observers; assigning ``None`` unsubscribes them."""
        self._render_subscription = None


def _leading_channels(data_type: str) -> int | None:
    """Channels the destination keeps when the annotator over-reports them."""
    if data_type.startswith(_SIMPLE_SHADING_PREFIX):
        return _SIMPLE_SHADING_CHANNELS
    return _LEADING_CHANNELS.get(data_type)


def _as_byte_view(array: Any) -> torch.Tensor:
    """View a contiguous Warp array as a flat uint8 CUDA tensor.

    The staging slots are dtype-agnostic so one buffer implementation serves
    every annotator: uint8 colour, float32 depth and uint32 segmentation all
    stage as raw bytes and are reinterpreted at launch time.
    """
    nbytes = int(array.size) * wp.types.type_size_in_bytes(array.dtype)
    byte_view = wp.array(ptr=array.ptr, shape=(nbytes,), dtype=wp.uint8, device=array.device)
    return wp.to_torch(byte_view)


class _OutputPlan:
    """Static per-annotator metadata for one asynchronous output."""

    __slots__ = ("colorize", "data_type", "depth_clip_max", "depth_replacement", "tiled_channels")

    def __init__(
        self,
        data_type: str,
        *,
        colorize: bool,
        tiled_channels: int | None,
        depth_clip_max: float | None,
        depth_replacement: float | None,
    ) -> None:
        self.data_type = data_type
        self.colorize = colorize
        self.tiled_channels = tiled_channels
        self.depth_clip_max = depth_clip_max
        self.depth_replacement = depth_replacement


class _AsyncRtxPath:
    """Asynchronous staging path for the RTX tiled annotators.

    Only two constraints are enforced, because only two are inherent: the
    outputs must live on a CUDA device (the whole design is stream/event
    ordering) and every destination must already sit on that device (the owned
    staging slots copy device-to-device).  Output count, data types and a
    composed ISP are all handled rather than rejected — one double buffer per
    annotator, with the ISP enqueued on the shared process stream so it still
    observes every reshape.
    """

    def __init__(
        self,
        render_data: Any,
        device: str,
        plans: tuple[_OutputPlan, ...],
    ) -> None:
        self._render_data = render_data
        self._device = device
        self._plans = plans
        torch_device = torch.device(device)
        self._buffers = {plan.data_type: CudaDoubleBuffer(torch_device) for plan in plans}
        # Warp arrays backing in-flight staging copies. Released only after
        # ``synchronize_source_copies``, since ``do_array_copy=False`` output and
        # our own ``.to()``/``.contiguous()`` temporaries own that memory.
        self._pending_sources: list[Any] = []

    @classmethod
    def try_create(cls, camera: ManaAsyncCamera) -> _AsyncRtxPath | None:
        renderer = camera._renderer
        render_data = camera._render_data
        device = camera.render_device
        if renderer is None or render_data is None:
            return None
        # Duck-typed capability probe, not a renderer allow-list: the path needs
        # tiled annotators writing into Warp destinations it can address.
        spec = getattr(render_data, "spec", None)
        annotators = getattr(render_data, "annotators", None)
        output_data = getattr(render_data, "output_data", None)
        if spec is None or not annotators or output_data is None:
            return None
        if getattr(render_data, "renderer_info", None) is None:
            return None

        # Hard constraint 1: staging is CUDA stream/event ordering.
        if not device.startswith("cuda"):
            return None

        renderer_cfg = getattr(renderer, "cfg", None)
        depth_behavior = getattr(renderer_cfg, "depth_clipping_behavior", "none")
        clipping_range = getattr(getattr(spec.cfg, "spawn", None), "clipping_range", None)
        plans: list[_OutputPlan] = []
        for data_type in annotators:
            destination = cls._resolve_destination(render_data, data_type)
            # Hard constraint 2: reshape writes straight into the destination.
            if destination is None or str(destination.device) != device:
                return None
            colorize_flag = _COLORIZE_CFG_FLAG.get(data_type)
            depth_clip_max = None
            if data_type == "distance_to_camera" and clipping_range is not None:
                depth_clip_max = float(clipping_range[1])
            depth_replacement = None
            if data_type in _DEPTH_TYPES and depth_behavior != "none" and clipping_range is not None:
                depth_replacement = 0.0 if depth_behavior == "zero" else float(clipping_range[1])
            plans.append(
                _OutputPlan(
                    data_type,
                    colorize=bool(colorize_flag is not None and getattr(renderer_cfg, colorize_flag, False)),
                    tiled_channels=_leading_channels(data_type),
                    depth_clip_max=depth_clip_max,
                    depth_replacement=depth_replacement,
                )
            )
        return cls(render_data, device, tuple(plans))

    @staticmethod
    def _resolve_destination(render_data: Any, data_type: str) -> Any:
        """Destination Warp array for *data_type*, mirroring the stock renderer.

        A composed ISP consumes HDR through the renderer's private scratch
        buffer whenever the user did not ask for the raw HDR AOV, so that case
        has no entry in ``output_data``.
        """
        output_data = render_data.output_data
        proxy = output_data.get(data_type)
        if proxy is not None:
            return proxy.warp
        if data_type == _RGB_HDR:
            return getattr(render_data, "_hdr_scratch_wp", None)
        return None

    @property
    def plans(self) -> tuple[_OutputPlan, ...]:
        """Static per-annotator plans, in annotator declaration order."""
        return self._plans

    def prepare_for_render(self) -> None:
        """Release the previous producer buffers before Kit may recycle them."""
        for buffer in self._buffers.values():
            buffer.synchronize_source_copies()
        self._pending_sources.clear()

    def capture_rendered_frame(self) -> None:
        """Copy every completed annotator frame and enqueue its processing."""
        render_data = self._render_data
        spec = render_data.spec
        output_data = render_data.output_data
        if spec is None or output_data is None:
            return

        view_count = int(spec.view_count)
        height = int(spec.cfg.height)
        width = int(spec.cfg.width)
        num_tiles_x = math.ceil(math.sqrt(view_count))

        for plan in self._plans:
            data_type = plan.data_type
            destination_wp = self._resolve_destination(render_data, data_type)
            if destination_wp is None:
                continue
            tiled_wp = self._stage_annotator_output(plan)
            if tiled_wp is None:
                continue
            buffer = self._buffers[data_type]
            warp_stream = buffer.process_warp_stream

            def reshape(
                staged: torch.Tensor,
                _process_stream: torch.cuda.Stream,
                *,
                plan: _OutputPlan = plan,
                destination_wp: Any = destination_wp,
                staged_shape: tuple[int, ...] = tuple(tiled_wp.shape),
                staged_dtype: Any = tiled_wp.dtype,
                warp_stream: wp.Stream = warp_stream,
            ) -> None:
                # ``staged`` is the byte view of the slot; reinterpret it back to
                # the annotator's own dtype before launching the reshape.
                from isaaclab.utils.warp.kernels import reshape_tiled_image

                staged_wp = wp.array(
                    ptr=wp.from_torch(staged).ptr,
                    shape=staged_shape,
                    dtype=staged_dtype,
                    device=self._device,
                )
                wp.launch(
                    kernel=reshape_tiled_image,
                    dim=(view_count, height, width),
                    inputs=[
                        staged_wp.flatten(),
                        destination_wp,
                        *list(destination_wp.shape[1:]),
                        num_tiles_x,
                    ],
                    device=self._device,
                    stream=warp_stream,
                )
                if plan.depth_clip_max is not None:
                    clamp_depth_to_inf_on_stream(
                        destination_wp,
                        plan.depth_clip_max,
                        device=self._device,
                        stream=warp_stream,
                    )
                if plan.depth_replacement is not None:
                    replace_inf_depth_on_stream(
                        destination_wp,
                        plan.depth_replacement,
                        device=self._device,
                        stream=warp_stream,
                    )

            source_stream = wp.stream_to_torch(wp.get_stream(tiled_wp.device))
            buffer.submit(
                _as_byte_view(tiled_wp),
                reshape,
                source_stream=source_stream,
            )
            self._pending_sources.append(tiled_wp)

        self._apply_isp()

    def _stage_annotator_output(self, plan: _OutputPlan) -> Any:
        """Fetch one annotator output as a contiguous Warp array on this device.

        Mirrors the per-type reinterpretation and channel trimming that
        ``IsaacRtxRenderer.render()`` performs before its reshape launch.
        """
        render_data = self._render_data
        data_type = plan.data_type
        output = render_data.annotators[data_type].get_data()
        if isinstance(output, dict):
            tiled = output["data"]
            render_data.renderer_info[data_type] = output.get("info")
        else:
            tiled = output

        if isinstance(tiled, np.ndarray):
            tiled_wp = wp.array(tiled, device=self._device)
        else:
            tiled_wp = tiled.to(device=self._device)

        if plan.colorize:
            # Raw uint32 segmentation reinterpreted as 4 uint8 channels.
            tiled_wp = wp.array(
                ptr=tiled_wp.ptr,
                shape=(*tiled_wp.shape, 4),
                dtype=wp.uint8,
                device=self._device,
            )
        if plan.tiled_channels is not None:
            tiled_wp = tiled_wp[:, :, : plan.tiled_channels].contiguous()
        elif not tiled_wp.is_contiguous:
            tiled_wp = tiled_wp.contiguous()
        return tiled_wp

    def _apply_isp(self) -> None:
        """Run a composed ISP after every reshape, on the shared process stream."""
        pipeline = getattr(self._render_data, "ppisp_pipeline", None)
        if pipeline is None or not self._buffers:
            return
        hdr_source = self._resolve_destination(self._render_data, _RGB_HDR)
        ldr_proxy = self._render_data.output_data.get(_RGBA)
        if hdr_source is None or ldr_proxy is None:
            return
        # Every reshape was enqueued on this stream, so the ISP already trails
        # them; the consumer ordering established by ``submit`` still holds.
        buffer = next(iter(self._buffers.values()))
        process_stream = buffer.process_stream
        process_warp_stream = buffer.process_warp_stream
        apply_ppisp_on_stream(
            pipeline,
            hdr_source,
            ldr_proxy.warp,
            stream=process_warp_stream,
        )
        done = torch.cuda.Event()
        done.record(process_stream)
        torch.cuda.current_stream(torch.device(self._device)).wait_event(done)

    def render(self, pump: Any) -> None:
        """Pump the renderer and capture when no external render owns the frame.

        *pump* advances the renderer's own frame source; it is passed in rather
        than imported so this path stays free of a specific renderer's module.
        """
        self.prepare_for_render()
        pump()
        self.capture_rendered_frame()

    def close(self) -> None:
        for buffer in self._buffers.values():
            buffer.close()
        self._pending_sources.clear()


class ManaAsyncCamera(Camera):
    """Camera extension used by EX001-6R real-time observations.

    The host tracker mirrors SensorBase's Warp timestamp kernels exactly, so
    the per-frame ``env_mask.numpy()`` synchronization is unnecessary.  Renderer
    output is copied into owned double buffers — one per annotator — and
    reshaped on a dedicated CUDA stream, with any composed ISP trailing on that
    same stream.  Only two things disqualify the asynchronous path: image output
    that is not on a CUDA device, and destinations that do not already live on
    the resolved render device.  Those cases fall back to the stock Camera
    renderer path.
    """

    def __init__(self, cfg: Any) -> None:
        import isaaclab.sim as sim_utils

        self._host_dirty_tracker: HostDirtyTracker | None = None
        self._async_rtx_path: _AsyncRtxPath | None = None
        self._external_render_capture_pending = False
        self._external_render_ticket: _RenderProductFrameTicket | None = None
        self._external_render_completion_required = False
        self._external_render_capture_valid = False
        self._render_device: str | None = None
        super().__init__(cfg)
        for camera_prim in sim_utils.find_matching_prims(self.cfg.prim_path, self.stage):
            normalize_opencv_camera_intrinsics_to_render_resolution(
                camera_prim,
                width=self.cfg.width,
                height=self.cfg.height,
            )

    @property
    def render_device(self) -> str:
        """Device owning image outputs; camera pose/state stays on ``device``."""
        if self._render_device is not None:
            return self._render_device
        return str(self.device)

    def update(self, dt: float, force_recompute: bool = False) -> None:
        if self._is_initialized and self._host_dirty_tracker is not None:
            self._host_dirty_tracker.update(dt)
        super().update(dt, force_recompute=force_recompute)

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        # Preserve an already-submitted ticket so the scheduler can prove that
        # native Hydra work completed before it publishes the reset state. The
        # resulting pixels no longer belong to this camera state and are dropped.
        self._external_render_capture_valid = False
        async_path = self._async_rtx_path
        if async_path is not None:
            async_path.prepare_for_render()
        self._external_render_capture_pending = False
        super().reset(env_ids=env_ids, env_mask=env_mask)
        tracker = self._host_dirty_tracker
        if tracker is None:
            return
        if env_mask is not None:
            tracker.reset(env_mask=copy_warp_bool_mask_to_host(env_mask))
        elif env_ids is None:
            tracker.reset()
        else:
            tracker.reset(env_ids=self._env_ids_to_host(env_ids))

    def _initialize_impl(self) -> None:
        self.cancel_external_render()
        if self._async_rtx_path is not None:
            self._async_rtx_path.close()
            self._async_rtx_path = None
        self._external_render_capture_pending = False
        super()._initialize_impl()
        state_device = str(self.device)
        self._render_device = state_device
        if _CAMERA_RENDER_DEVICE != state_device:
            if self._try_rebind_render_device(_CAMERA_RENDER_DEVICE):
                self._render_device = _CAMERA_RENDER_DEVICE
            else:
                logger.warning(
                    "ManaAsyncCamera cannot place image output on %s for %s; the renderer needs annotators "
                    "that can be reattached on another device. Keeping render_device=%s.",
                    _CAMERA_RENDER_DEVICE,
                    self.cfg.prim_path,
                    state_device,
                )
        self._host_dirty_tracker = HostDirtyTracker(
            num_envs=self._num_envs,
            update_period=self.cfg.update_period,
        )
        self._async_rtx_path = _AsyncRtxPath.try_create(self)
        if self._async_rtx_path is None:
            logger.info(
                "ManaAsyncCamera uses the stock renderer path for %s (renderer=%s, outputs=%s)",
                self.cfg.prim_path,
                type(self._renderer).__name__ if self._renderer is not None else None,
                tuple(getattr(self.cfg, "data_types", ())),
            )
        else:
            logger.info(
                "ManaAsyncCamera enabled double buffering for %s (state_device=%s, render_device=%s, outputs=%s)",
                self.cfg.prim_path,
                state_device,
                self.render_device,
                tuple(plan.data_type for plan in self._async_rtx_path.plans),
            )

    def prepare_for_external_render(self, *, wait_for_completion: bool = False) -> bool:
        """Protect the current Replicator output before an external Kit pump."""
        async_path = self._async_rtx_path
        if async_path is None:
            return False
        self.cancel_external_render()
        async_path.prepare_for_render()
        if wait_for_completion:
            render_product_path = self._render_product_path()
            self._external_render_ticket = _RenderProductFrameTicket.arm(render_product_path)
            self._external_render_completion_required = True
        self._external_render_capture_valid = True
        return True

    def external_render_is_ready(self) -> bool:
        """Return whether the latest Kit update dispatched this product's frame."""
        ticket = self._external_render_ticket
        if ticket is None:
            raise RuntimeError("ManaAsyncCamera external render ticket was not armed")
        if ticket.is_ready:
            ticket.require_completed()
        return ticket.is_ready

    def external_render_is_armed(self) -> bool:
        """Return whether a pending environment handoff still owns a live ticket."""
        return self._external_render_ticket is not None and self._external_render_completion_required

    def cancel_external_render(self) -> None:
        """Cancel an incomplete handoff during teardown or error unwind."""
        ticket = getattr(self, "_external_render_ticket", None)
        if ticket is not None:
            ticket.cancel()
        self._external_render_ticket = None
        self._external_render_completion_required = False
        self._external_render_capture_valid = False

    def capture_after_external_render(self) -> bool:
        """Enqueue processing for the frame produced by an external Kit pump."""
        async_path = self._async_rtx_path
        if async_path is None:
            return False
        if getattr(self, "_external_render_completion_required", False):
            ticket = getattr(self, "_external_render_ticket", None)
            if ticket is None:
                raise RuntimeError("ManaAsyncCamera external render ticket was not armed")
            ticket.require_completed()
            ticket.cancel()
            self._external_render_ticket = None
        capture_valid = getattr(self, "_external_render_capture_valid", True)
        if capture_valid:
            async_path.capture_rendered_frame()
            self._external_render_capture_pending = True
        self._external_render_completion_required = False
        self._external_render_capture_valid = False
        return capture_valid

    def _render_product_path(self) -> str:
        render_data = self._render_data
        render_product = getattr(render_data, "render_product", None)
        path = getattr(render_product, "path", None)
        if path:
            return str(path)
        paths = getattr(render_data, "render_product_paths", None)
        if paths and len(paths) == 1:
            return str(paths[0])
        raise RuntimeError("ManaAsyncCamera requires exactly one render product for async handoff")

    def _update_outdated_buffers(self, force_recompute: bool = False) -> None:
        if not force_recompute and self._data_generation == self._data_generation_last_update:
            return
        super()._update_outdated_buffers(force_recompute=force_recompute)
        if self._host_dirty_tracker is not None:
            self._host_dirty_tracker.consume()

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        import isaaclab.sim as sim_utils

        tracker = self._host_dirty_tracker
        if tracker is not None and not tracker.has_any():
            return
        async_path = self._async_rtx_path
        if async_path is None:
            super()._update_buffers_impl(env_mask)
            return

        if self.cfg.update_latest_camera_pose:
            self._update_poses(env_mask=env_mask, frame_op=1)
        else:
            self._update_camera_state(env_mask=env_mask, frame_op=1)

        sim_ctx = sim_utils.SimulationContext.instance()
        if sim_ctx is not None:
            sim_ctx.render_context.update_scene_state(sim_ctx.get_physics_step_count())
        if not self._external_render_capture_pending:
            async_path.render(self._pump_renderer)
        assert self._renderer is not None
        self._renderer.read_output(self._render_data, self._data)
        self._external_render_capture_pending = False

    def _pump_renderer(self) -> None:
        """Advance the renderer's frame source for the synchronous fallback.

        Kit-backed renderers need their app update pumped before annotators hold
        a fresh frame; renderers that produce frames on demand need nothing here.
        """
        try:
            from isaaclab_physx.renderers.isaac_rtx_renderer_utils import (  # noqa: PLC0415
                ensure_isaac_rtx_render_update,
            )
        except ImportError:
            return
        ensure_isaac_rtx_render_update()

    def _env_mask_has_any(self, env_mask: wp.array) -> bool:
        """Read the CPU mirror instead of synchronously copying the Warp mask."""
        del env_mask
        tracker = self._host_dirty_tracker
        return True if tracker is None else tracker.has_any()

    def _invalidate_initialize_callback(self, event: Any) -> None:
        self.cancel_external_render()
        if self._async_rtx_path is not None:
            self._async_rtx_path.close()
            self._async_rtx_path = None
        self._external_render_capture_pending = False
        self._host_dirty_tracker = None
        self._render_device = None
        super()._invalidate_initialize_callback(event)

    def _try_rebind_render_device(self, render_device: str) -> bool:
        """Move annotator/output storage to *render_device* without moving state.

        Every annotator is recreated from its own registered name and init
        params, so this is not limited to a particular output set.  Either all
        annotators move or none do — a partial move would leave destinations
        split across devices, which the asynchronous path rejects anyway.
        """
        from isaaclab.sensors.camera.camera_data import CameraData

        renderer = self._renderer
        render_data = self._render_data
        if renderer is None or render_data is None:
            return False
        spec = getattr(render_data, "spec", None)
        old_annotators = getattr(render_data, "annotators", None)
        render_product_paths = getattr(render_data, "render_product_paths", None)
        if not render_product_paths:
            render_product = getattr(render_data, "render_product", None)
            render_product_path = getattr(render_product, "path", None)
            if render_product_path:
                render_product_paths = [render_product_path]
        if spec is None or not old_annotators or not render_product_paths:
            return False
        if not hasattr(renderer, "supported_output_types") or not hasattr(renderer, "set_outputs"):
            return False

        specs = [(name, *self._annotator_identity(annotator)) for name, annotator in old_annotators.items()]
        if any(registered is None for _name, registered, _params in specs):
            return False

        replacement = CameraData.allocate(
            data_types=list(self.cfg.data_types),
            height=self.cfg.height,
            width=self.cfg.width,
            num_views=self._view.count,
            device=render_device,
            supported_specs=renderer.supported_output_types(),
        )
        replacement_output = replacement.output
        if replacement_output is None:
            return False

        import omni.replicator.core as rep  # noqa: PLC0415

        new_annotators: dict[str, Any] = {}
        attached: list[Any] = []
        for name, registered, params in specs:
            new_annotators[name] = rep.AnnotatorRegistry.get_annotator(
                registered,
                init_params=params or None,
                device=render_device,
                do_array_copy=False,
            )
        for annotator in old_annotators.values():
            annotator.detach(render_product_paths)
        try:
            for annotator in new_annotators.values():
                annotator.attach(render_product_paths)
                attached.append(annotator)
        except Exception:
            for annotator in attached:
                annotator.detach(render_product_paths)
            for annotator in old_annotators.values():
                annotator.attach(render_product_paths)
            raise

        old_spec = spec
        old_output = self._data.output
        old_info = self._data.info
        old_image_shape = self._data.image_shape
        old_hdr_scratch = getattr(render_data, "_hdr_scratch_wp", None)
        try:
            render_data.annotators = new_annotators
            render_data.spec = replace(spec, device=render_device)
            self._data._output = replacement_output
            self._data.info = replacement.info
            self._data.image_shape = replacement.image_shape
            # Re-runs the renderer's own output wiring, which also reallocates a
            # composed ISP's HDR scratch buffer on the new device.
            renderer.set_outputs(render_data, replacement_output)
        except Exception:
            for annotator in new_annotators.values():
                annotator.detach(render_product_paths)
            for annotator in old_annotators.values():
                annotator.attach(render_product_paths)
            render_data.annotators = old_annotators
            render_data.spec = old_spec
            self._data._output = old_output
            self._data.info = old_info
            self._data.image_shape = old_image_shape
            render_data._hdr_scratch_wp = old_hdr_scratch
            if old_output is not None:
                renderer.set_outputs(render_data, old_output)
            raise
        return True

    @staticmethod
    def _annotator_identity(annotator: Any) -> tuple[str | None, dict[str, Any]]:
        """Registered annotator name and init params needed to recreate it.

        Replicator keeps both on the instance; without the registered name the
        annotator cannot be rebuilt on another device, so the caller aborts.
        """
        registered = getattr(annotator, "_name", None)
        params = getattr(annotator, "_init_params", None)
        return (registered if isinstance(registered, str) else None, dict(params) if params else {})

    @staticmethod
    def _env_ids_to_host(env_ids: Sequence[int]) -> np.ndarray:
        if isinstance(env_ids, torch.Tensor):
            if env_ids.is_cuda:
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(env_ids.device))
                ready.synchronize()
            return env_ids.detach().to(device="cpu", dtype=torch.int64).numpy().reshape(-1)
        if isinstance(env_ids, wp.array):
            return wp.to_torch(env_ids).to(device="cpu", dtype=torch.int64).numpy().reshape(-1)
        return np.asarray(env_ids, dtype=np.int64).reshape(-1)


__all__ = ["ManaAsyncCamera"]
