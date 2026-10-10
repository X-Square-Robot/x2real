"""Compatibility shims for supported Warp releases and consumers."""

import logging

logger = logging.getLogger(__name__)


def configure_warp_cuda_compilation() -> bool:
    """Disable Warp 1.15.0 NVRTC PCH compilation on ``sm_120`` devices.

    Warp's precompiled-header path crashes with ``SIGBUS`` while compiling the
    first CUDA kernel on RTX 50-series GPUs in the fallback 1.15.0 runtime.
    Keep the workaround narrow so Warp 1.16 and other unaffected runtimes
    preserve their configured behavior.

    Returns:
        Whether this call changed Warp's global compilation configuration.
    """
    import warp as wp

    version = str(getattr(wp, "__version__", "")).split("+", 1)[0]
    if version != "1.15.0" or bool(wp.config.llvm_cuda) or not bool(wp.config.use_precompiled_headers):
        return False
    if not any(getattr(device, "arch", None) == 120 for device in wp.get_cuda_devices()):
        return False

    wp.config.use_precompiled_headers = False
    logger.info("Disabled Warp 1.15.0 NVRTC precompiled headers on sm_120 to avoid first-kernel SIGBUS")
    return True


def ensure_wp_torch_namespace() -> None:
    """Restore ``wp.torch`` for cuRobo 0.8.0 on Warp releases that omit it."""
    import warp as wp

    if not hasattr(wp, "torch"):
        import warp._src.torch as wp_torch

        wp.torch = wp_torch
