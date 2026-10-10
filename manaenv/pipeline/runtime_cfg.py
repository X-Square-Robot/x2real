from __future__ import annotations

from dataclasses import dataclass

# Runtime CFG schema epoch. Bump when a BACKWARD-INCOMPATIBLE definition-schema change
# lands (a field rename/removal older images can't parse) so remote dispatch can refuse a
# too-old image up front instead of crashing inside the container. History since the 5.1
# image (epoch 0): flat-solver PipelineRuntimeCFG (solver/solver_params) + actuator rename
# (actuator_model -> actuator) => epoch 1; SceneRuntimeCFG removed the explicit override gate
# and now applies every authored non-null field => epoch 2; native-actuator
# execution settings moved from PipelineRuntimeCFG to RobotGeneratorCFG => epoch 3;
# asynchronous PhysX bool replaced by an explicit physics step strategy => epoch 4;
# Kit render strategy added and async PhysX now requires its native handshake => epoch 5.
RUNTIME_SCHEMA_EPOCH = 5

# How one env step drives PhysX relative to rendering.
#   "sync"  — upstream ``ManagerBasedRLEnv.step()``; physics and render are serial.
#   "async" — the complete physics chain runs on a persistent worker thread
#             while the CPU-only native Kit handshake renders concurrently.
PHYSICS_STEP_MODE_SYNC = "sync"
PHYSICS_STEP_MODE_ASYNC = "async"
PHYSICS_STEP_MODES = frozenset(
    {
        PHYSICS_STEP_MODE_SYNC,
        PHYSICS_STEP_MODE_ASYNC,
    }
)

# How Kit completes a render requested by ``SimulationContext.render()``.
# ``async_handshake`` keeps the Python caller on the main thread while Kit
# executes Hydra/RTX work on its native render thread.
KIT_RENDER_MODE_SYNC = "sync"
KIT_RENDER_MODE_ASYNC_HANDSHAKE = "async_handshake"
KIT_RENDER_MODES = frozenset(
    {
        KIT_RENDER_MODE_SYNC,
        KIT_RENDER_MODE_ASYNC_HANDSHAKE,
    }
)


def runtime_device_type(device: object) -> str:
    """Return the normalized device type without importing a tensor runtime."""
    return str(device).strip().lower().partition(":")[0]

@dataclass
class PipelineRuntimeCFG:
    """Global pipeline runtime defaults for environment construction."""

    device: str = "cuda:0"
    disable_fabric: bool = False
    actuator: str = "pd"
    start_paused: bool = False
    render_cfg: str = "isaac_rtx"
    # Public physics backend selector. Native actuator graph selection is
    # handled separately by the embodiment/actuator configuration.
    solver: str | None = None
    cuda_graph: bool | None = None
    # Optional DiffIK device. When unset, it stays on the simulation device.
    # Native-actuator placement is authored on RobotGeneratorCFG; AssemblyFactory
    # resolves StablePD to CPU for every CPU PhysX schedule.
    diff_ik_device: str | None = None
    diff_ik_cuda_graph: bool | None = None
    # Select ManaEnv's Warp-native pseudo-inverse/DLS DiffIK action. It runs on
    # diff_ik_device (or the simulation device when unset).
    use_warp_diff_ik: bool = False
    solver_params: dict | None = None
    native_actuators: bool | None = None
    world_spacing: tuple[float, float, float] | None = None
    # Which robot cameras get a stream viewport during evaluation.
    stream_viewports: dict | None = None
    # Opt in to PhysX collider debug drawing. Async stepping detects this mode
    # and serializes Kit rendering with physics to protect the live PxScene.
    enable_colliders: bool = False
    # Select the PhysX step implementation. ``sync`` uses IsaacLab's
    # ManagerBasedRLEnv.step; ``async`` moves the complete physics substep chain
    # to a worker thread and therefore requires the overlapping Kit handshake.
    physics_step_mode: str = PHYSICS_STEP_MODE_SYNC
    # Select Kit rendering independently from PhysX stepping. Synchronous PhysX
    # keeps the upstream environment; asynchronous PhysX uses Mana's worker.
    kit_render_mode: str = KIT_RENDER_MODE_SYNC

    def __post_init__(self) -> None:
        if self.solver is not None and self.solver != "physx":
            raise ValueError(
                "PipelineRuntimeCFG.solver only supports PhysX in the public build; "
                f"got {self.solver!r}."
            )
        if self.physics_step_mode not in PHYSICS_STEP_MODES:
            expected = ", ".join(sorted(PHYSICS_STEP_MODES))
            raise ValueError(f"physics_step_mode must be one of: {expected}; got {self.physics_step_mode!r}")
        if self.kit_render_mode not in KIT_RENDER_MODES:
            expected = ", ".join(sorted(KIT_RENDER_MODES))
            raise ValueError(f"kit_render_mode must be one of: {expected}; got {self.kit_render_mode!r}")
        if self.physics_step_mode == PHYSICS_STEP_MODE_ASYNC and self.kit_render_mode == KIT_RENDER_MODE_SYNC:
            raise ValueError(
                "physics_step_mode='async' has no overlap with kit_render_mode='sync'; "
                "use sync/sync or async/async_handshake"
            )
        if self.kit_render_mode == KIT_RENDER_MODE_ASYNC_HANDSHAKE:
            if runtime_device_type(self.device) != "cpu":
                raise ValueError("kit_render_mode='async_handshake' currently supports only CPU PhysX")
            if self.physics_step_mode == PHYSICS_STEP_MODE_ASYNC:
                diff_ik_device = self.diff_ik_device or self.device
                if runtime_device_type(diff_ik_device) != "cpu":
                    raise ValueError("asynchronous PhysX with kit_render_mode='async_handshake' requires CPU DiffIK")
                if self.diff_ik_cuda_graph is True or self.cuda_graph is True:
                    raise ValueError("asynchronous PhysX cannot run DiffIK or physics CUDA Graphs")
                if self.disable_fabric:
                    raise ValueError("asynchronous PhysX requires Fabric snapshot publication")
                if self.enable_colliders:
                    raise ValueError("asynchronous PhysX cannot be combined with enable_colliders=true")


PipelineRuntime = PipelineRuntimeCFG
