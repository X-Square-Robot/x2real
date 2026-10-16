"""Public ManaEnv pipeline for evaluation and manual demonstration collection."""

from __future__ import annotations

import copy
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlsplit, urlunsplit

from manaenv.pipeline.data_types import DataBucket
from manaenv.pipeline.datacenter.output_orders import OutputOrdersCFG
from manaenv.pipeline.pipeline_cfg import PipelineDagNodeCFG, PipelineDagSolution
from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
from manaenv.pipeline.runtime import runtime_flags
from manaenv.pipeline.translator.reflective_translator import ReflectiveTranslator, reflect_section
from manaenv.policy.mana_policy import ManaPolicyCFG
from manaenv.utils.config_io import create_config_from_dict_strict
from manaenv.utils.debug_print import manaprint


class ManaPipelineCFG:
    setup_mode: Literal[0, 1] = 0

    def __init__(
        self,
        output_path: str | None = None,
        description_yaml_path: str | None = None,
        policy_server_address: str | None = None,
        policy_server_port: int | None = None,
        policy_config_path: str | None = None,
        enable_cameras: bool | None = None,
    ) -> None:
        self.output_path = output_path
        self.description_yaml_path = description_yaml_path
        self.policy_server_address = policy_server_address
        self.policy_server_port = policy_server_port
        self.policy_config_path = policy_config_path
        self.enable_cameras = None if enable_cameras is None else bool(enable_cameras)


def _normalized_accessor_name(name: str) -> str:
    return name.replace("_", "").lower()


def _looks_like_pipeline_component_accessor(name: str) -> bool:
    return name.endswith("_factory") or name.endswith("_center")


class ManaPipeline:
    """Build and evaluate the environments declared by one public suite."""

    active_robot: Any = None

    def __init__(self, asset_registry: Any = None) -> None:
        bootstrap_discovery()
        self.translator = ReflectiveTranslator()
        self.components: dict[str, Any] = {}
        self.dag_nodes: dict[str, PipelineDagNodeCFG] = {}
        self.dag_solution: PipelineDagSolution | None = None
        self._component_accessor_names: set[str] = set()
        self._asset_registry = asset_registry
        self._policy_runtime_overrides: dict[str, Any] = {}
        self._policy_cfg_override: ManaPolicyCFG | None = None
        self._assets_registered = False

    def __getattr__(self, name: str) -> Any:
        component = self._component_by_accessor_name(name)
        if component is not None:
            return component
        if _looks_like_pipeline_component_accessor(name):
            return None
        raise AttributeError(f"{type(self).__name__} object has no attribute {name!r}")

    def _component_by_accessor_name(self, name: str) -> Any | None:
        direct = self.components.get(name)
        if direct is not None:
            return direct
        normalized = _normalized_accessor_name(name)
        for node_id, node in self.dag_nodes.items():
            if normalized in {
                _normalized_accessor_name(node_id),
                _normalized_accessor_name(node.cfg_key),
            }:
                return self.components.get(node_id)
        return None

    def get_asset_registry(self) -> Any:
        data_center = getattr(self, "data_center", None)
        if data_center is None:
            raise RuntimeError("DataCenter has not been configured")
        return data_center.asset_registry

    def setup(self, cfg: ManaPipelineCFG) -> None:
        if not cfg.description_yaml_path:
            raise ValueError("description_yaml_path is required")
        manaprint(f"Parsing evaluation suite: {cfg.description_yaml_path}")
        self._policy_runtime_overrides = {
            key: value
            for key, value in (
                ("server_address", cfg.policy_server_address),
                ("server_port", cfg.policy_server_port),
            )
            if value is not None
        }
        self._policy_cfg_override = self._load_policy_cfg_override(
            suite_path=cfg.description_yaml_path,
            policy_config_path=cfg.policy_config_path,
        )

        from manaenv.pipeline.runtime import master_seed as master_seed_runtime

        tree = self.translator.precompile(cfg.description_yaml_path)
        master_seed_runtime.publish_from_pipeline(
            yaml_master_seed=master_seed_runtime.peek_yaml_master_seed(tree)
        )
        self.translator.compile(tree)
        pipeline_cfg = self.translator.section("pipeline", required=True)
        runtime_cfg = getattr(pipeline_cfg, "pipeline_runtime", None)
        if runtime_cfg is not None and runtime_cfg.stream_viewports is not None:
            runtime_flags.set_value("stream_viewports", runtime_cfg.stream_viewports)

        reproducibility = getattr(pipeline_cfg, "reproducibility", None)
        yaml_master = getattr(reproducibility, "master_seed", None)
        effective = master_seed_runtime.publish_from_pipeline(yaml_master_seed=yaml_master)
        if effective is None:
            manaprint("INFO: master seed unset; samplers use operating-system entropy")
        else:
            manaprint(f"INFO: effective master seed: {effective}")

        self._setup_pipeline_components(pipeline_cfg)
        self._apply_camera_runtime_override(cfg.enable_cameras)
        self._route_dag_orders()
        self._register_policy_configs()

    # DAG setup -------------------------------------------------------
    def _setup_pipeline_components(self, pipeline_cfg: Any) -> None:
        dag = getattr(pipeline_cfg, "pipeline_dag", None)
        if dag is None:
            raise TypeError("pipeline.pipeline_dag is required")
        self.dag_solution = dag.solve()
        self.dag_nodes = {node.id: node for node in dag.nodes}
        self._clear_pipeline_component_accessors()
        self.components = {}

        for node_id in self.dag_solution.order:
            node = self.dag_nodes[node_id]
            node_cfg = self._resolve_node_cfg(node)
            component_cls = getattr(node_cfg, "class_type", None)
            if not isinstance(component_cls, type):
                raise TypeError(f"{node.cfg_key} must declare a component class_type")
            component = component_cls()
            self.components[node_id] = component
            self._publish_pipeline_component_accessors(node, component)
            self._wire_pipeline_component(node, component, pipeline_cfg)
            component.setup(node_cfg)
            self._post_setup_pipeline_component(component)

    def _resolve_node_cfg(self, node: PipelineDagNodeCFG) -> Any:
        cfg = self.translator.section(node.cfg_key)
        if cfg is None:
            raise KeyError(f"DAG node {node.id!r} references missing section {node.cfg_key!r}")
        return cfg

    def _wire_pipeline_component(self, node: PipelineDagNodeCFG, component: Any, pipeline_cfg: Any) -> None:
        for attr_name, node_ref in node.deps.items():
            if node_ref not in self.components:
                raise KeyError(f"DAG dependency {node_ref!r} for {node.id!r} is unavailable")
            setattr(component, attr_name, self.components[node_ref])
        if node.runtime == "pipeline_runtime":
            component.set_runtime_cfg(pipeline_cfg.pipeline_runtime)
        elif node.runtime is not None:
            raise ValueError(f"unsupported DAG runtime binding: {node.runtime!r}")

    def _post_setup_pipeline_component(self, component: Any) -> None:
        accessor = getattr(type(component), "asset_registry", None)
        if isinstance(accessor, property) and accessor.fset is not None:
            self._bind_asset_registry(component)

    def _bind_asset_registry(self, data_center: Any) -> None:
        if self._asset_registry is None:
            module = importlib.import_module("isaaclab_arena.assets.registries")
            self._asset_registry = module.AssetRegistry()
        data_center.asset_registry = self._asset_registry

    def _publish_pipeline_component_accessors(self, node: PipelineDagNodeCFG, component: Any) -> None:
        for name in {node.id, node.cfg_key}:
            if name.isidentifier() and not hasattr(type(self), name):
                setattr(self, name, component)
                self._component_accessor_names.add(name)

    def _clear_pipeline_component_accessors(self) -> None:
        for name in self._component_accessor_names:
            self.__dict__.pop(name, None)
        self._component_accessor_names.clear()

    def _route_dag_orders(self) -> None:
        if self.dag_solution is None:
            return
        for node_id in self.dag_solution.order:
            node = self.dag_nodes[node_id]
            if not node.orders:
                continue
            if len(node.orders) != 1:
                raise ValueError(f"DAG node {node_id!r} must consume exactly one order section")
            section_name = node.orders[0]
            orders = self.translator.section(section_name)
            if orders is None and section_name == "output":
                orders = OutputOrdersCFG()
            receive_orders = getattr(self.components[node_id], "receive_orders", None)
            if not callable(receive_orders):
                raise TypeError(f"DAG component {node_id!r} cannot receive orders")
            receive_orders(orders)

    def _apply_camera_runtime_override(self, enabled: bool | None) -> None:
        for component in self.components.values():
            setter = getattr(component, "set_enable_cameras", None)
            if callable(setter):
                setter(enabled)

    # Policy endpoint override ---------------------------------------
    @staticmethod
    def _load_policy_cfg_override(
        *, suite_path: str, policy_config_path: str | None
    ) -> ManaPolicyCFG | None:
        if not policy_config_path:
            return None
        suite_dir = Path(suite_path).expanduser().resolve().parent
        override_orders = reflect_section(
            "rollout",
            [
                {
                    "env_ref": "__policy_override__",
                    "planner": {
                        "PolicyPlanner": {
                            "policy_cfg": {"include": str(Path(policy_config_path).expanduser())}
                        }
                    },
                }
            ],
            base_dir=suite_dir,
            context="policy override",
        )
        orders = getattr(override_orders, "orders", None) or []
        if len(orders) != 1:
            raise ValueError("policy override must resolve to exactly one policy configuration")
        planner_cfg = getattr(orders[0], "planner", None)
        policy_cfg = getattr(planner_cfg, "policy_cfg", None)
        if isinstance(policy_cfg, dict):
            policy_cfg = create_config_from_dict_strict(
                policy_cfg,
                ManaPolicyCFG,
                context="policy override",
            )
        if not isinstance(policy_cfg, ManaPolicyCFG):
            raise TypeError("policy override did not resolve to ManaPolicyCFG")
        return policy_cfg

    def _register_policy_configs(self) -> None:
        if not self._policy_runtime_overrides and self._policy_cfg_override is None:
            return
        for component in self.components.values():
            for order in getattr(component, "_orders", None) or []:
                planner_cfg = getattr(order, "planner", None)
                if planner_cfg is None or not hasattr(planner_cfg, "policy_cfg"):
                    continue
                if self._policy_cfg_override is not None:
                    policy_cfg = copy.deepcopy(self._policy_cfg_override)
                    planner_cfg.policy_cfg = policy_cfg
                else:
                    policy_cfg = planner_cfg.policy_cfg
                if isinstance(policy_cfg, dict):
                    policy_cfg = create_config_from_dict_strict(
                        policy_cfg,
                        ManaPolicyCFG,
                        context="rollout.planner.policy_cfg",
                    )
                    planner_cfg.policy_cfg = policy_cfg
                if policy_cfg is not None:
                    self._apply_policy_runtime_overrides(policy_cfg)

    def _apply_policy_runtime_overrides(self, policy_cfg: ManaPolicyCFG) -> None:
        overrides = self._policy_runtime_overrides
        remote_cfg = getattr(policy_cfg, "remote", None)
        if remote_cfg is not None:
            if "server_address" in overrides and hasattr(remote_cfg, "server_address"):
                remote_cfg.server_address = overrides["server_address"]
            if "server_port" in overrides and hasattr(remote_cfg, "server_port"):
                remote_cfg.server_port = overrides["server_port"]
            return

        implementation_cfg = getattr(policy_cfg, "implementation", None)
        endpoint = str(getattr(implementation_cfg, "endpoint", "") or "")
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            return
        host = str(overrides.get("server_address", parsed.hostname))
        port = overrides.get("server_port", parsed.port)
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        authority = f"{host}:{int(port)}" if port is not None else host
        implementation_cfg.endpoint = urlunsplit(
            (parsed.scheme, authority, parsed.path, parsed.query, parsed.fragment)
        )

    # Product generation ---------------------------------------------
    def generate_task(self) -> int:
        tasks = self.task_factory.fulfill_orders()
        if tasks:
            self.data_center.store_tasks(tasks)
        return len(tasks)

    def generate_scene(self, scene_type: str | None = None) -> int:
        scenes = self.scene_factory.fulfill_orders(scene_type)
        if scenes:
            self.data_center.store_scenes(scenes)
        return len(scenes)

    def generate_robot(self, robot_type: str | None = None) -> int:
        robots = self.robot_factory.fulfill_orders(robot_type)
        if robots:
            self.data_center.store_robots(robots)
        return len(robots)

    def assemble(self, env_name: str | None = None) -> int:
        environments = self.assembly_factory.assemble(self.data_center, env_name=env_name)
        if environments:
            self.data_center.store_environments(environments)
        return len(environments)

    def get_scenes(self) -> list[Any]:
        return self.data_center.get(DataBucket.SCENES)

    def get_robots(self) -> list[Any]:
        return self.data_center.get(DataBucket.ROBOTS)

    def get_tasks(self) -> list[Any]:
        return self.data_center.get(DataBucket.TASKS)

    def get_environments(self) -> list[Any]:
        return self.data_center.get(DataBucket.ENVIRONMENTS)

    def get_environment(self, name: str) -> Any | None:
        return next(
            (env for env in self.get_environments() if getattr(env, "name", None) == name),
            None,
        )

    def _assembly_order(self, env_name: str) -> Any | None:
        section = self.translator.section("assembly_orders")
        return next(
            (order for order in getattr(section, "orders", None) or [] if order.name == env_name),
            None,
        )

    def _has_product_for_generator(self, bucket: DataBucket, generator_name: str) -> bool:
        matcher = self.assembly_factory._product_matches_name
        return any(matcher(item, generator_name) for item in self.data_center.get(bucket) or [])

    def ensure_environment(self, env_name: str) -> Any | None:
        cached = self.get_environment(env_name)
        if cached is not None:
            return cached
        order = self._assembly_order(env_name)
        if order is None:
            return None
        self._register_assets()
        if not self._has_product_for_generator(DataBucket.TASKS, order.task_generator_name):
            self._generate_tasks_filtered({order.task_generator_name})
        if not self._has_product_for_generator(DataBucket.SCENES, order.scene_generator_name):
            self._generate_scenes_filtered({order.scene_generator_name})
        if not self.data_center.get(DataBucket.ROBOTS):
            self.generate_robot()
        environments = self.assembly_factory.assemble(self.data_center, env_name=env_name)
        self.data_center.store_environments(environments)
        return self.get_environment(env_name) or (environments[0] if environments else None)

    # Evaluation ------------------------------------------------------
    def run_rollout(
        self,
        env_name: str | set[str] | list[str] | tuple[str, ...] | None = None,
        external_gym_env: Any = None,
    ) -> List[Dict[str, Any]]:
        if self.rollout_factory is None:
            return []
        return self.rollout_factory.fulfill_orders(
            data_center=self.data_center,
            env_name=env_name,
            external_gym_env=external_gym_env,
        )

    def save(self) -> Dict[str, List[str]]:
        return self.data_center.save()

    # Warmup ----------------------------------------------------------
    def _register_assets(self) -> None:
        if self._assets_registered:
            return
        import manaenv.assets.objects.object_library  # noqa: F401
        from manaenv.assets.local_discovery import discover_and_register_local_assets

        discover_and_register_local_assets(asset_registry=self.get_asset_registry())
        self._assets_registered = True

    def _collect_needed_env_refs(self) -> set[str] | None:
        refs: set[str] = set()
        for component in self.components.values():
            for order in getattr(component, "_orders", None) or []:
                ref = getattr(order, "env_ref", None)
                if ref:
                    refs.add(str(ref))
        return refs or None

    def _collect_needed_generator_names(self, env_refs: set[str]) -> tuple[set[str], set[str], set[str]]:
        scenes: set[str] = set()
        robots: set[str] = set()
        tasks: set[str] = set()
        section = self.translator.section("assembly_orders")
        for order in getattr(section, "orders", None) or []:
            if order.name not in env_refs:
                continue
            scenes.add(order.scene_generator_name)
            robots.add(order.robot_generator_name)
            tasks.add(order.task_generator_name)
        return scenes, robots, tasks

    def warmup(self, env_refs: set[str] | None = None) -> None:
        self._register_assets()
        if self.data_center.get(DataBucket.ENVIRONMENTS):
            return
        needed = set(env_refs) if env_refs is not None else self._collect_needed_env_refs()
        if needed:
            self._warmup_filtered(needed)
        else:
            self.generate_task()
            self.generate_scene()
            self.generate_robot()
            self.assemble()

    def _warmup_filtered(self, env_refs: set[str]) -> None:
        scene_names, _robot_names, task_names = self._collect_needed_generator_names(env_refs)
        self._generate_tasks_filtered(task_names) if task_names else self.generate_task()
        self._generate_scenes_filtered(scene_names) if scene_names else self.generate_scene()
        self.generate_robot()
        for env_name in sorted(env_refs):
            self.assemble(env_name=env_name)

    def _generate_tasks_filtered(self, needed_names: set[str]) -> int:
        original = list(self.task_factory._orders)
        filtered = [(name, cfg) for name, cfg in original if name in needed_names]
        if not filtered:
            raise ValueError(f"no task order matches {sorted(needed_names)}")
        self.task_factory._orders = filtered
        try:
            return self.generate_task()
        finally:
            self.task_factory._orders = original

    def _generate_scenes_filtered(self, needed_names: set[str]) -> int:
        original = list(self.scene_factory._orders)
        filtered = [entry for entry in original if entry[0] in needed_names]
        if not filtered:
            raise ValueError(f"no scene order matches {sorted(needed_names)}")
        self.scene_factory._orders = filtered
        try:
            return self.generate_scene()
        finally:
            self.scene_factory._orders = original

    # Optional DAG execution surface used by isolated evaluation workers.
    def dag_target_nodes(self, target: str) -> list[str]:
        if self.dag_solution is None:
            return []
        return [node_id for node_id in self.dag_solution.order if target in self.dag_nodes[node_id].orders]

    def execute_node(self, node_id: str, shard: str | None = None, **kwargs: Any) -> Any:
        component = self.components.get(node_id)
        fulfill = getattr(component, "fulfill_orders", None)
        if not callable(fulfill):
            raise ValueError(f"DAG node {node_id!r} is not executable")
        context = {"shard": shard, **kwargs}
        signature = inspect.signature(fulfill)
        accepted = {key: value for key, value in context.items() if key in signature.parameters and value is not None}
        return fulfill(**accepted)

    def cleanup(self) -> None:
        cleaned: set[int] = set()
        for component in self.components.values():
            cleanup = getattr(component, "cleanup", None)
            if callable(cleanup) and id(component) not in cleaned:
                cleanup()
                cleaned.add(id(component))
        self.active_robot = None


ManaBenchPipeLine = ManaPipeline
PipelineSetupCFG = ManaPipelineCFG

__all__ = ["ManaPipeline", "ManaPipelineCFG"]
