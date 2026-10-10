"""Robot Factory Module

Creates and manages robot embodiments in the ManaEnv pipeline.
All robots are looked up from the global AssetRegistry (populated via
@register_asset decorators).  Add a new robot: implement its embodiment
class, decorate with @register_asset, and place it under
manaenv.assets.robots using the asset-name naming convention.
"""

import copy
import inspect
import traceback
from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.data_types import DataBucket
from manaenv.pipeline.factory import BaseFactory, BaseFactoryCFG
from manaenv.pipeline.registry.lazy import ModuleImportSpec
from manaenv.pipeline.runtime_cfg import PipelineRuntimeCFG
from manaenv.utils.debug_print import manaprint
from manaenv.utils.fingerprint import constraints_fingerprint

_ROBOT_ASSET_PACKAGE = "manaenv.assets.robots"


@dataclass
class RobotGeneratorCFG:
    """One robot generator order.

    ``name`` is both the concrete robot asset name and the identifier used by
    assembly orders.
    """

    name: str = ""
    #: Optional AssetRegistry / import key when ``name`` is an order alias
    #: (e.g. ``EX001-6R-H`` → asset ``EX001-6R`` for multi-gripper suites).
    asset_name: str | None = None
    initial_position: list[float] = field(default_factory=lambda: [0, 0, 0])
    initial_orientation: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    initial_joint_positions: list[float] | None = None
    # Optional articulation solver overrides. ``None`` preserves the asset's
    # authored defaults; definition fragments set these when the profile owns
    # a stability-tuned iteration contract.
    solver_position_iteration_count: int | None = None
    solver_velocity_iteration_count: int | None = None
    #: Optional spawn USD override. Honored only by robot constructors that
    #: accept a ``usd_path`` argument; ignored otherwise.
    usd_path: str | None = None
    enable_cameras: bool = False
    #: Optional depth + semantic camera outputs for evaluation policies.
    camera_depth_semantic: bool = False
    gripper_mode: str = "raw_joint"
    # EX001-6R only: "H" selects the legacy H-jaw asset; "G" selects the
    # current G-jaw asset. ``None`` preserves the embodiment default.
    gripper_variant: str | None = None
    #: DiffIK evaluation cadence: command caches one target per env step;
    #: physics preserves IsaacLab's per-physics-step feedback update.
    diff_ik_compute_mode: str = "command"
    actuator: str = "pd"
    # Optional native-actuator execution defaults. These belong to the robot;
    # AssemblyFactory resolves StablePD to CPU for every CPU PhysX schedule.
    actuator_device: str | None = None
    actuator_cuda_graph: bool | None = None
    actuator_params: dict = field(default_factory=dict)
    disable_gripper_shadow: bool = False
    disable_scene_shadow: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class RobotOrdersCFG:
    """Typed robot order section."""

    orders: list[RobotGeneratorCFG] = field(default_factory=list)


def _resolve_camera_cfg(robot_tag: str) -> type | None:
    """Resolve a robot tag to its camera configclass via the global registry.

    Returns ``None`` when the tag has no manaenv-side camera profile. The cfg
    classes are heavy (IsaacSim ``@configclass``), so resolution is lazy.
    """
    from manaenv.pipeline.factories.camera_discovery import ROBOT_TAG_TO_SYMBOL
    from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
    from manaenv.pipeline.registry.global_provider import global_registry

    symbol = ROBOT_TAG_TO_SYMBOL.get(robot_tag)
    if symbol is None:
        return None
    bootstrap_discovery()
    return global_registry.get_impl(symbol)


# (robot_name, RobotGeneratorCFG)
RobotOrder = tuple[str, RobotGeneratorCFG]


@dataclass
class RobotFactoryCFG(BaseFactoryCFG):
    """Configuration for RobotFactory."""

    class_type: Any = None
    type: str = "default_robot_factory"
    enable_cameras: bool = False

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = RobotFactory


class RobotFactory(BaseFactory):
    """Factory for creating robot embodiments.

    Robots are resolved from the global AssetRegistry by generator name.  Task
    robot_constraints may fill a missing name; identical robots are
    deduplicated before returning.
    """

    def __init__(self):
        super().__init__()
        self.cfg: RobotFactoryCFG = None
        self.data_center = None  # set by Pipeline
        self.runtime_cfg = PipelineRuntimeCFG()
        self._orders: list[RobotOrder] = []
        self._enable_cameras_override: bool | None = None

    def setup(self, cfg: RobotFactoryCFG | None = None):
        self.cfg = cfg if cfg is not None else RobotFactoryCFG()
        self._orders = []
        manaprint(f"INFO: RobotFactory setup with type: {self.cfg.type}")

    @property
    def asset_registry(self):
        if self.data_center is None:
            return None
        return self.data_center.asset_registry

    def set_runtime_cfg(self, runtime_cfg: PipelineRuntimeCFG | None) -> None:
        self.runtime_cfg = runtime_cfg if runtime_cfg is not None else PipelineRuntimeCFG()

    def set_enable_cameras(self, enabled: bool | None) -> None:
        """Override robot cameras for this launch, or preserve YAML with None."""
        self._enable_cameras_override = None if enabled is None else bool(enabled)

    def receive_orders(self, orders: RobotOrdersCFG) -> None:
        if not isinstance(orders, RobotOrdersCFG):
            raise TypeError(f"RobotFactory.receive_orders expected RobotOrdersCFG, got {type(orders).__name__}")
        self._orders = []
        for robot_cfg in orders.orders:
            self._apply_runtime_defaults(robot_cfg)
            self._orders.append((robot_cfg.name, robot_cfg))
            manaprint(f"INFO: RobotFactory received order: {robot_cfg.name}")
        manaprint(f"INFO: RobotFactory received {len(self._orders)} order(s)")

    def get_order_count(self) -> int:
        return len(self._orders)

    def has_type(self, name: str) -> bool:
        registry = self.asset_registry
        return registry.is_registered(name) if registry is not None else False

    def fulfill_orders(self, robot_type: str | None = None) -> list[Any]:
        """Fulfill robot orders driven by per-task constraints.

        Each YAML robot template is expanded per unique robot_constraints
        group found across tasks, then deduplicated by USD identity.
        """
        if not self._orders:
            manaprint("WARNING: No robot orders to fulfill")
            return []

        tasks = self.data_center.get(DataBucket.TASKS)
        orders = self._orders
        if robot_type:
            orders = [(n, c) for n, c in self._orders if n == robot_type]

        manaprint(f"INFO: Fulfilling {len(orders)} order(s) with {len(tasks)} task(s)...")

        all_robots: list[Any] = []
        for order_name, order_cfg in orders:
            if not tasks:
                robot = self._create_robot(order_cfg)
                if robot is not None:
                    all_robots.append(robot)
                continue

            merged = self._merge_task_robot_orders(tasks)
            candidates: list[Any] = []
            for constraints in merged:
                cfg_copy = copy.deepcopy(order_cfg)
                self._apply_task_constraints(cfg_copy, constraints)
                robot = self._create_robot(cfg_copy)
                if robot is not None:
                    candidates.append(robot)
                else:
                    manaprint(f"ERROR: Failed to produce robot '{order_name}'")

            unique = self._deduplicate_robots(candidates)
            manaprint(f"INFO: '{order_name}': {len(candidates)} candidates -> {len(unique)} unique")
            all_robots.extend(unique)

        manaprint(f"INFO: Robot production complete. Total: {len(all_robots)}")
        return all_robots

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_runtime_defaults(self, robot_cfg: RobotGeneratorCFG) -> None:
        if getattr(robot_cfg, "actuator", "pd") == "pd":
            robot_cfg.actuator = getattr(self.runtime_cfg, "actuator", "pd") or "pd"

    @staticmethod
    def _merge_task_robot_orders(tasks: list) -> list[dict[str, Any]]:
        """Return list of unique robot_constraints dicts across tasks."""
        seen: dict[str, dict[str, Any]] = {}
        for task in tasks:
            rc = getattr(task, "robot_constraints", {}) or {}
            fp = constraints_fingerprint(rc)
            if fp not in seen:
                seen[fp] = rc
        return list(seen.values())

    @staticmethod
    def _apply_task_constraints(cfg, demands: dict[str, Any]) -> None:
        """Overlay task embodiment constraints onto a robot order cfg.

        Only applies when cfg.name is empty (i.e. YAML left it unset).
        """
        if not isinstance(demands, dict):
            return
        emb = demands.get("embodiment")
        if not emb or cfg.name:
            return
        if isinstance(emb, dict):
            cfg.name = emb.get("name") or emb.get("tags", "")
        else:
            cfg.name = emb

    @staticmethod
    def _deduplicate_robots(robots: list) -> list:
        seen: set = set()
        unique: list = []
        for robot in robots:
            usd = getattr(robot, "_robot_usd_path", None)
            fp = usd or tuple(getattr(robot, "tags", []))
            if fp not in seen:
                seen.add(fp)
                unique.append(robot)
        return unique

    def _create_robot(self, robot_cfg: RobotGeneratorCFG) -> Any | None:
        from isaaclab_arena.utils.pose import Pose

        robot_class = self._get_robot_class(robot_cfg)
        if robot_class is None:
            manaprint(f"ERROR: Robot '{robot_cfg.name}' not found in AssetRegistry")
            return None

        initial_pose = Pose(
            position_xyz=tuple(robot_cfg.initial_position),
            rotation_xyzw=tuple(robot_cfg.initial_orientation),
        )
        try:
            ctor_kwargs = {
                "enable_cameras": (
                    robot_cfg.enable_cameras if self._enable_cameras_override is None else self._enable_cameras_override
                ),
                "initial_pose": initial_pose,
            }
            sig = inspect.signature(robot_class.__init__)
            if "usd_path" in sig.parameters and getattr(robot_cfg, "usd_path", None):
                ctor_kwargs["usd_path"] = robot_cfg.usd_path
            if "gripper_mode" in sig.parameters and hasattr(robot_cfg, "gripper_mode"):
                ctor_kwargs["gripper_mode"] = getattr(robot_cfg, "gripper_mode", "raw_joint")
            if "gripper_variant" in sig.parameters and getattr(robot_cfg, "gripper_variant", None) is not None:
                ctor_kwargs["gripper_variant"] = robot_cfg.gripper_variant
            if "diff_ik_compute_mode" in sig.parameters and hasattr(robot_cfg, "diff_ik_compute_mode"):
                ctor_kwargs["diff_ik_compute_mode"] = getattr(robot_cfg, "diff_ik_compute_mode", "command")
            if "actuator" in sig.parameters and hasattr(robot_cfg, "actuator"):
                ctor_kwargs["actuator"] = getattr(robot_cfg, "actuator", "pd")
            if "actuator_params" in sig.parameters and hasattr(robot_cfg, "actuator_params"):
                ctor_kwargs["actuator_params"] = getattr(robot_cfg, "actuator_params", None)
            if "disable_gripper_shadow" in sig.parameters and hasattr(robot_cfg, "disable_gripper_shadow"):
                ctor_kwargs["disable_gripper_shadow"] = getattr(robot_cfg, "disable_gripper_shadow", False)
            if "disable_scene_shadow" in sig.parameters and hasattr(robot_cfg, "disable_scene_shadow"):
                ctor_kwargs["disable_scene_shadow"] = getattr(robot_cfg, "disable_scene_shadow", False)
            if "camera_depth_semantic" in sig.parameters:
                ctor_kwargs["camera_depth_semantic"] = bool(getattr(robot_cfg, "camera_depth_semantic", False))
            robot = robot_class(**ctor_kwargs)
            self._apply_articulation_solver_iterations(robot, robot_cfg)

            from manaenv.actions.command_rate_diff_ik import (
                configure_diff_ik_action_cfg,
                normalize_diff_ik_compute_mode,
            )

            diff_ik_compute_mode = normalize_diff_ik_compute_mode(robot_cfg.diff_ik_compute_mode)
            use_warp_diff_ik = bool(getattr(self.runtime_cfg, "use_warp_diff_ik", False))
            diff_ik_device = getattr(self.runtime_cfg, "diff_ik_device", None)
            use_cuda_graph = getattr(self.runtime_cfg, "diff_ik_cuda_graph", None)
            if use_cuda_graph is None:
                use_cuda_graph = getattr(self.runtime_cfg, "cuda_graph", None)
            robot.diff_ik_compute_mode = diff_ik_compute_mode
            robot.use_warp_diff_ik = use_warp_diff_ik
            configured_diff_ik_terms = configure_diff_ik_action_cfg(
                getattr(robot, "action_config", None),
                diff_ik_compute_mode,
                use_warp=use_warp_diff_ik,
                use_cuda_graph=use_cuda_graph,
                device=diff_ik_device,
            )
            if configured_diff_ik_terms:
                manaprint(
                    f"INFO:   Robot DiffIK backend={'warp' if use_warp_diff_ik else 'torch'} "
                    f"compute_mode={diff_ik_compute_mode!r} terms={configured_diff_ik_terms}"
                )

            if robot_cfg.initial_joint_positions is not None:
                self._apply_initial_joint_positions(robot, robot_cfg.initial_joint_positions)

            asset_key = (robot_cfg.asset_name or robot_cfg.name or "").strip()
            if robot.camera_config is None:
                cam_cfg = _resolve_camera_cfg(asset_key)
                if cam_cfg is not None:
                    robot.camera_config = cam_cfg()

            if hasattr(robot, "gripper_mode"):
                manaprint(f"INFO:   Robot gripper_mode={getattr(robot, 'gripper_mode', '?')!r}")

            robot._mana_cfg = robot_cfg
            robot._generator_name = robot_cfg.name
            robot.robot_type = asset_key or robot_cfg.name
            robot.tags = [robot_cfg.name] if robot_cfg.name else []
            robot._robot_usd_path = self._extract_robot_usd_path(robot, robot_class)
            return robot

        except Exception as e:
            manaprint(f"ERROR: Failed to create robot '{robot_cfg.name}': {e}")
            traceback.print_exc()
            return None

    def _get_robot_class(self, robot_cfg: RobotGeneratorCFG) -> type | None:
        registry = self.asset_registry
        if registry is None:
            manaprint("ERROR: RobotFactory has no asset registry")
            return None

        asset_key = (robot_cfg.asset_name or robot_cfg.name or "").strip()
        if not asset_key:
            return None
        ModuleImportSpec.from_symbol(_ROBOT_ASSET_PACKAGE, asset_key).import_module()
        if registry.is_registered(asset_key, ensure_loaded=False):
            return registry.get_asset_by_name(asset_key)
        return None

    @staticmethod
    def _extract_robot_usd_path(robot, robot_class) -> str | None:
        if hasattr(robot, "usd_path"):
            return robot.usd_path
        if hasattr(robot, "scene_config"):
            sc = robot.scene_config
            if hasattr(sc, "robot") and hasattr(sc.robot, "spawn"):
                usd = getattr(sc.robot.spawn, "usd_path", None)
                if usd:
                    return usd
        if hasattr(robot, "robot_cfg") and hasattr(robot.robot_cfg, "usd_path"):
            return robot.robot_cfg.usd_path
        robot_cfg = getattr(robot_class, "ROBOT_CFG", None) or getattr(robot, "ROBOT_CFG", None)
        if robot_cfg:
            return getattr(robot_cfg, "usd_path", None)
        return None

    @staticmethod
    def _apply_articulation_solver_iterations(robot: Any, robot_cfg: RobotGeneratorCFG) -> None:
        """Apply explicit definition-owned iteration counts to an articulated robot."""
        position_iterations = robot_cfg.solver_position_iteration_count
        velocity_iterations = robot_cfg.solver_velocity_iteration_count
        if position_iterations is None and velocity_iterations is None:
            return

        scene_cfg = getattr(robot, "scene_config", None)
        robot_scene_cfg = getattr(scene_cfg, "robot", None)
        spawn_cfg = getattr(robot_scene_cfg, "spawn", None)
        articulation_props = getattr(spawn_cfg, "articulation_props", None)
        if articulation_props is None:
            raise ValueError(
                f"Robot {robot_cfg.name!r} does not expose articulation solver properties required by its definition."
            )

        if position_iterations is not None:
            articulation_props.solver_position_iteration_count = position_iterations
        if velocity_iterations is not None:
            articulation_props.solver_velocity_iteration_count = velocity_iterations

    @staticmethod
    def _apply_initial_joint_positions(robot: Any, joint_positions: list[float]) -> None:
        if hasattr(robot, "scene_config") and robot.scene_config is not None:
            sc = robot.scene_config
            if hasattr(sc, "robot") and hasattr(sc.robot, "init_state"):
                init_state = sc.robot.init_state
                if hasattr(init_state, "joint_pos") and isinstance(init_state.joint_pos, dict):
                    for i, name in enumerate(init_state.joint_pos):
                        if i < len(joint_positions):
                            init_state.joint_pos[name] = joint_positions[i]

        if hasattr(robot, "event_config") and robot.event_config is not None:
            for attr_name in dir(robot.event_config):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(robot.event_config, attr_name, None)
                if attr is not None and hasattr(attr, "params"):
                    params = attr.params
                    if isinstance(params, dict) and "default_pose" in params:
                        params["default_pose"] = joint_positions
