"""Explicit-stream Warp launches used by the asynchronous camera path.

Warp's device current stream is shared across Python threads. Helpers in this
module therefore require a stream and pass it to every launch.
"""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.kernel
def _clamp_depth_to_inf_kernel(
    buffer: wp.array(dtype=wp.float32, ndim=4),
    max_range: float,
):
    camera, height, width, channel = wp.tid()
    value = buffer[camera, height, width, channel]
    if value > max_range:
        buffer[camera, height, width, channel] = wp.inf


@wp.kernel
def _replace_inf_depth_kernel(
    buffer: wp.array(dtype=wp.float32, ndim=4),
    replacement: float,
):
    camera, height, width, channel = wp.tid()
    if wp.isinf(buffer[camera, height, width, channel]):
        buffer[camera, height, width, channel] = replacement


def clamp_depth_to_inf_on_stream(
    buffer: wp.array,
    max_range: float,
    *,
    device: str,
    stream: wp.Stream,
) -> None:
    wp.launch(
        _clamp_depth_to_inf_kernel,
        dim=buffer.shape,
        inputs=[buffer, float(max_range)],
        device=device,
        stream=stream,
    )


def replace_inf_depth_on_stream(
    buffer: wp.array,
    replacement: float,
    *,
    device: str,
    stream: wp.Stream,
) -> None:
    wp.launch(
        _replace_inf_depth_kernel,
        dim=buffer.shape,
        inputs=[buffer, float(replacement)],
        device=device,
        stream=stream,
    )


def _static_ppisp_inputs(hdr: wp.array, rgba: wp.array, cfg: Any) -> list[Any]:
    if hdr.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP HDR input must be wp.float32, got {hdr.dtype}.")
    if rgba.dtype is not wp.uint8:
        raise ValueError(f"Camera PPISP RGBA output must be wp.uint8, got {rgba.dtype}.")

    inputs = cfg.inputs
    return [
        hdr,
        rgba,
        int(rgba.shape[2]),
        int(rgba.shape[1]),
        float(inputs.get("responsivity", 1.0)),
        float(inputs["exposureOffset"]),
        wp.vec2f(*inputs["vignettingCenterR"]),
        float(inputs["vignettingAlpha1R"]),
        float(inputs["vignettingAlpha2R"]),
        float(inputs["vignettingAlpha3R"]),
        wp.vec2f(*inputs["vignettingCenterG"]),
        float(inputs["vignettingAlpha1G"]),
        float(inputs["vignettingAlpha2G"]),
        float(inputs["vignettingAlpha3G"]),
        wp.vec2f(*inputs["vignettingCenterB"]),
        float(inputs["vignettingAlpha1B"]),
        float(inputs["vignettingAlpha2B"]),
        float(inputs["vignettingAlpha3B"]),
        wp.vec2f(*inputs["colorLatentBlue"]),
        wp.vec2f(*inputs["colorLatentRed"]),
        wp.vec2f(*inputs["colorLatentGreen"]),
        wp.vec2f(*inputs["colorLatentNeutral"]),
        float(inputs["crfToeR"]),
        float(inputs["crfShoulderR"]),
        float(inputs["crfGammaR"]),
        float(inputs["crfCenterR"]),
        float(inputs["crfToeG"]),
        float(inputs["crfShoulderG"]),
        float(inputs["crfGammaG"]),
        float(inputs["crfCenterG"]),
        float(inputs["crfToeB"]),
        float(inputs["crfShoulderB"]),
        float(inputs["crfGammaB"]),
        float(inputs["crfCenterB"]),
    ]


def _controller_ppisp_inputs(
    hdr: wp.array,
    rgba: wp.array,
    controller_params: wp.array,
    cfg: Any,
) -> list[Any]:
    import isaaclab_ppisp.kernels as ppisp_kernels

    if hdr.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP HDR input must be wp.float32, got {hdr.dtype}.")
    if rgba.dtype is not wp.uint8:
        raise ValueError(f"Camera PPISP RGBA output must be wp.uint8, got {rgba.dtype}.")
    if controller_params.dtype is not wp.float32:
        raise ValueError(f"Camera PPISP controller params must be wp.float32, got {controller_params.dtype}.")
    if controller_params.shape != (rgba.shape[0], ppisp_kernels.PPISP_CONTROLLER_PARAM_COUNT):
        raise ValueError(
            "Camera PPISP controller params must have shape "
            f"({rgba.shape[0]}, {ppisp_kernels.PPISP_CONTROLLER_PARAM_COUNT}), got {controller_params.shape}."
        )

    inputs = cfg.inputs
    return [
        hdr,
        rgba,
        controller_params,
        int(rgba.shape[2]),
        int(rgba.shape[1]),
        float(inputs.get("responsivity", 1.0)),
        wp.vec2f(*inputs["vignettingCenterR"]),
        float(inputs["vignettingAlpha1R"]),
        float(inputs["vignettingAlpha2R"]),
        float(inputs["vignettingAlpha3R"]),
        wp.vec2f(*inputs["vignettingCenterG"]),
        float(inputs["vignettingAlpha1G"]),
        float(inputs["vignettingAlpha2G"]),
        float(inputs["vignettingAlpha3G"]),
        wp.vec2f(*inputs["vignettingCenterB"]),
        float(inputs["vignettingAlpha1B"]),
        float(inputs["vignettingAlpha2B"]),
        float(inputs["vignettingAlpha3B"]),
        float(inputs["crfToeR"]),
        float(inputs["crfShoulderR"]),
        float(inputs["crfGammaR"]),
        float(inputs["crfCenterR"]),
        float(inputs["crfToeG"]),
        float(inputs["crfShoulderG"]),
        float(inputs["crfGammaG"]),
        float(inputs["crfCenterG"]),
        float(inputs["crfToeB"]),
        float(inputs["crfShoulderB"]),
        float(inputs["crfGammaB"]),
        float(inputs["crfCenterB"]),
    ]


def _controller_params_on_stream(pipeline: Any, hdr: wp.array, stream: wp.Stream) -> wp.array:
    import isaaclab_ppisp.kernels as ppisp_kernels

    cfg = pipeline.cfg
    controller_weights = cfg.controller_weights
    assert controller_weights is not None

    device = str(hdr.device)
    weights = pipeline._controller_weights_by_device.get(device)
    if weights is None:
        host_weights = wp.array(controller_weights, dtype=wp.float32, device="cpu", pinned=True)
        weights = wp.empty(len(controller_weights), dtype=wp.float32, device=device)
        wp.copy(weights, host_weights, stream=stream)
        pipeline._controller_weights_by_device[device] = weights
        host_weights_by_device = getattr(pipeline, "_controller_host_weights_by_device", None)
        if host_weights_by_device is None:
            host_weights_by_device = {}
            pipeline._controller_host_weights_by_device = host_weights_by_device
        host_weights_by_device[device] = host_weights

    features, controller_params = pipeline._controller_buffers(hdr)
    ppisp_kernels._validate_ppisp_controller_inputs(hdr, weights, features, controller_params)
    image_height = int(hdr.shape[1])
    image_width = int(hdr.shape[2])
    wp.launch_tiled(
        ppisp_kernels._ppisp_controller_pool_features_native_kernel,
        dim=(int(hdr.shape[0]), ppisp_kernels.PPISP_CONTROLLER_POOL_CELL_COUNT),
        inputs=[hdr, weights, features, image_width, image_height, float(cfg.controller_responsivity)],
        device=device,
        stream=stream,
        block_dim=ppisp_kernels.PPISP_CONTROLLER_POOL_THREAD_GROUP_SIZE,
    )
    wp.launch_tiled(
        ppisp_kernels._ppisp_controller_mlp_native_kernel,
        dim=(int(hdr.shape[0]),),
        inputs=[features, weights, controller_params, float(cfg.controller_prior_exposure)],
        device=device,
        stream=stream,
        block_dim=ppisp_kernels.PPISP_CONTROLLER_MLP_THREAD_GROUP_SIZE,
    )
    return controller_params


def apply_ppisp_on_stream(
    pipeline: Any,
    hdr: wp.array,
    rgba: wp.array,
    *,
    stream: wp.Stream,
) -> None:
    """Apply an IsaacLab PPISP pipeline without changing Warp's current stream."""
    import isaaclab_ppisp.kernels as ppisp_kernels

    native_apply = getattr(pipeline, "apply_on_stream", None)
    if native_apply is not None:
        native_apply(hdr, rgba, stream=stream)
        return

    cfg = getattr(pipeline, "cfg", None)
    if cfg is None:
        raise TypeError(f"Unsupported camera ISP pipeline without a normalized cfg: {type(pipeline)!r}")
    if cfg.controller_weights is None:
        wp.launch(
            ppisp_kernels._apply_ppisp_kernel,
            dim=rgba.shape[:3],
            inputs=_static_ppisp_inputs(hdr, rgba, cfg),
            device=str(rgba.device),
            stream=stream,
        )
        return

    controller_params = _controller_params_on_stream(pipeline, hdr, stream)
    wp.launch(
        ppisp_kernels._apply_ppisp_controller_kernel,
        dim=rgba.shape[:3],
        inputs=_controller_ppisp_inputs(hdr, rgba, controller_params, cfg),
        device=str(rgba.device),
        stream=stream,
    )


__all__ = [
    "apply_ppisp_on_stream",
    "clamp_depth_to_inf_on_stream",
    "replace_inf_depth_on_stream",
]
