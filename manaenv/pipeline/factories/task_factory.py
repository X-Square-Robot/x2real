import types as _types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from manaenv.pipeline.factory import BaseFactory, BaseFactoryCFG
from manaenv.pipeline.generator import BaseGenerator
from manaenv.pipeline.generators.task.dag_success import failure_reason_label
from manaenv.pipeline.generators.task_generator import TaskGeneratorCFG
from manaenv.utils.debug_print import manaprint

GENERATOR_REF_ATTR = "_originating_generator"


@dataclass
class ExtModelAPICFG:
    api_base: str = ""
    api_key: str = ""
    default_model: str = "qwen-plus"


@dataclass
class TaskFactoryCFG(BaseFactoryCFG):
    class_type: Any = None
    type: str = "default_task_factory"
    metadata: dict[str, Any] = field(default_factory=dict)
    ext_model_api: ExtModelAPICFG = field(default_factory=ExtModelAPICFG)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = TaskFactory


TaskOrder = tuple[str, Any]


@dataclass
class TaskOrdersCFG:
    """Typed task order section."""

    orders: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized: list[TaskGeneratorCFG] = []
        for index, item in enumerate(self.orders or []):
            if isinstance(item, TaskGeneratorCFG):
                normalized.append(item)
                continue
            normalized.append(_normalize_task_order_item(item, index=index))
        self.orders = normalized


def _normalize_task_order_item(item: Any, *, index: int) -> TaskGeneratorCFG:
    """Return the typed generator CFG represented by one task order item."""
    if isinstance(item, TaskGeneratorCFG):
        return item

    if isinstance(item, dict):
        from manaenv.pipeline.translator.yaml_reflection import reflect_yaml_dict

        generator_items = [(key, value) for key, value in item.items() if str(key).endswith("_task_generator")]
        if len(generator_items) == 1:
            key, value = generator_items[0]
            if isinstance(value, TaskGeneratorCFG):
                return value
            reflected = reflect_yaml_dict({key: value}, context=f"task_orders.orders[{index}]")
            reflected_value = reflected[key]
            if isinstance(reflected_value, TaskGeneratorCFG):
                return reflected_value

    raise TypeError(
        "TaskOrdersCFG.orders entries must be TaskGeneratorCFG objects or single *_task_generator mappings; "
        f"entry {index} is {type(item).__name__}"
    )


@dataclass
class TaskRuntimeSupport:
    env_obj: Any
    evaluation_runtime: Any | None = None
    extra_key: str = "evaluation_episode_records"
    _report_path: Path | None = None
    _report_env_name: str = ""
    _report_prompt: str = ""

    @property
    def enabled(self) -> bool:
        return self.evaluation_runtime is not None

    def describe(self) -> str:
        if self.evaluation_runtime is None:
            return ""
        return f"task evaluation enabled, max_possible_score={self.evaluation_runtime.max_possible_score:g}"

    def register_hooks(self, runner: Any) -> None:
        if self.evaluation_runtime is not None:
            self.evaluation_runtime.register_hooks(runner, extra_key=self.extra_key)

    def configure_resume(
        self,
        records: list[dict[str, Any]],
        *,
        total_episodes: int,
    ) -> None:
        """Restore durable JSONL episodes before opening a resumed report."""

        if self.evaluation_runtime is None:
            raise RuntimeError("episode resume requires a task evaluation runtime")
        target = int(total_episodes)
        if target < 1:
            raise ValueError(f"resume total_episodes must be positive, got {target}")
        if len(records) > target:
            raise ValueError(f"evaluation report contains {len(records)} episodes, exceeding target {target}")
        self.evaluation_runtime.restore_episode_records(records)

    def open_evaluation_report(
        self,
        output_dir: str,
        *,
        env_name: str,
        prompt: str = "",
        total_episodes: int | None = None,
    ) -> str | None:
        """Start streaming the public evaluation report as episodes finalize."""
        if self.evaluation_runtime is None:
            return None
        from manaenv.pipeline.evaluation.report import write_evaluation_episode_jsonl

        report_path = Path(output_dir) / "evaluation_report.jsonl"
        write_evaluation_episode_jsonl(
            report_path,
            records=self.evaluation_runtime.episode_records(),
            env_name=env_name,
            prompt=prompt,
        )
        self._report_path = report_path
        self._report_env_name = env_name
        self._report_prompt = prompt
        del total_episodes

        def _persist(records: list[dict[str, Any]]) -> None:
            path = self._report_path
            if path is None or not records:
                return
            write_evaluation_episode_jsonl(
                path,
                records=self.evaluation_runtime.episode_records(),
                env_name=self._report_env_name,
                prompt=self._report_prompt,
            )

        self.evaluation_runtime.set_on_finalized_records(_persist)
        return str(report_path)

    # Kept as a narrow source-compatibility alias for external task adapters.
    # It does not enable collection or recorder behavior.
    open_collection_report = open_evaluation_report

    def handle_manual_reset(self, gym_env: Any, env_ids: list[int], extra: dict[str, Any]) -> None:
        if self.evaluation_runtime is None or not env_ids:
            return
        extra[self.extra_key] = self.evaluation_runtime.finalize_env_ids(gym_env, env_ids)

    def collect_result_payload(self) -> dict[str, Any]:
        if self.evaluation_runtime is None:
            return {}
        return {
            "episode_evaluations": self.evaluation_runtime.episode_records(),
            "evaluation": self.evaluation_runtime.build_summary(),
        }

    def collect_evaluation_artifacts(
        self, output_dir: str, *, env_name: str, prompt: str = ""
    ) -> dict[str, Any] | None:
        """Materialize the final JSONL report and summary for one evaluation."""
        if self.evaluation_runtime is None:
            return None

        from manaenv.pipeline.evaluation.report import write_evaluation_report_jsonl

        evaluation_records = self.evaluation_runtime.episode_records()
        evaluation_summary = self.evaluation_runtime.build_summary()
        report_path = self._report_path or (Path(output_dir) / "evaluation_report.jsonl")
        written = write_evaluation_report_jsonl(
            report_path,
            records=evaluation_records,
            summary=evaluation_summary,
            env_name=env_name or self._report_env_name,
            prompt=prompt if prompt else self._report_prompt,
        )
        return {
            "evaluation_enabled": True,
            "evaluation_summary": evaluation_summary,
            "evaluation_report_path": written,
        }


class TaskFactory(BaseFactory):
    def __init__(self):
        super().__init__()
        self.cfg: TaskFactoryCFG = None
        self._orders: list[TaskOrder] = []
        self._generators: dict[str, BaseGenerator] = {}
        self.data_center = None

    def setup(self, cfg=None):
        if isinstance(cfg, dict):
            ext_api_dict = cfg.get("ext_model_api", {})
            ext_model_api = ExtModelAPICFG(
                api_base=ext_api_dict.get("api_base", ""),
                api_key=ext_api_dict.get("api_key", ""),
                default_model=ext_api_dict.get("default_model", "qwen-plus"),
            )
            self.cfg = TaskFactoryCFG(
                type=cfg.get("type", "default_task_factory"),
                ext_model_api=ext_model_api,
            )
        else:
            self.cfg = cfg if cfg is not None else TaskFactoryCFG()

        self._orders = []
        self._generators = {}
        manaprint(f"INFO: TaskFactory setup with type: {self.cfg.type}")

    def receive_orders(self, generator_cfgs: TaskOrdersCFG) -> None:
        if not isinstance(generator_cfgs, TaskOrdersCFG):
            raise TypeError(f"TaskFactory.receive_orders expected TaskOrdersCFG, got {type(generator_cfgs).__name__}")
        self._orders = []
        for gen_cfg in generator_cfgs.orders:
            if gen_cfg is None:
                manaprint("WARNING: Skip empty task generator config")
                continue

            gen_class = getattr(gen_cfg, "class_type", None)
            gen_name = gen_class.__name__ if gen_class else "unknown"
            definition = getattr(gen_cfg, "definition", {}) or {}
            task_name = definition.get("task_type") or gen_name
            self._orders.append((task_name, gen_cfg))
            manaprint(f"INFO: TaskFactory received spec order: {task_name} (generator={gen_name})")

        for order_idx, (task_name, gen_cfg) in enumerate(self._orders):
            order_key = self._get_order_generator_key(task_name, order_idx)
            self._assemble_generator(gen_cfg, cache_key=order_key)

        manaprint(f"INFO: TaskFactory received {len(self._orders)} spec order(s), {len(self._generators)} generator(s)")

    def _assemble_generator(self, gen_cfg: Any, cache_key: str | None = None) -> BaseGenerator | None:
        if gen_cfg is None:
            return None

        implementation_class = getattr(gen_cfg, "class_type", None)
        if implementation_class is None:
            manaprint("WARNING: No class_type in generator config")
            return None

        gen_key = implementation_class.__name__
        cache_key = cache_key or gen_key

        if cache_key in self._generators:
            return self._generators[cache_key]

        try:
            generator = implementation_class(gen_cfg, ext_model_api=self.cfg.ext_model_api)
            self._generators[cache_key] = generator
            manaprint(f"INFO: Assembled generator '{gen_key}'")
            return generator
        except Exception as e:
            manaprint(f"ERROR: Failed to assemble generator '{gen_key}': {e}")
            return None

    def _get_order_generator_key(self, task_name: str, order_idx: int) -> str:
        return f"{task_name}::{order_idx}"

    def get_order_count(self) -> int:
        return len(self._orders)

    def get_generator_count(self) -> int:
        return len(self._generators)

    def get_generator(self, generator_class_name: str) -> BaseGenerator | None:
        direct = self._generators.get(generator_class_name)
        if direct is not None:
            return direct
        for generator in self._generators.values():
            if type(generator).__name__ == generator_class_name:
                return generator
        return None

    def fulfill_orders(self) -> list[Any]:
        """Run generators; returns list of TaskCFG results."""
        if not self._orders:
            manaprint("WARNING: No task orders to fulfill")
            return []

        manaprint(f"INFO: Fulfilling {len(self._orders)} task order(s)...")

        results = self._run_generators()
        manaprint(f"INFO: Task production complete. Total: {len(results)}")
        return results

    def _run_generators(self) -> list[Any]:
        all_results: list[Any] = []
        runtime = self.data_center.collect_available_assets() if self.data_center else {}

        for order_idx, (task_name, gen_cfg) in enumerate(self._orders):
            implementation_class = getattr(gen_cfg, "class_type", None)
            if implementation_class is None:
                continue
            order_key = self._get_order_generator_key(task_name, order_idx)
            generator = self._generators.get(order_key)
            gen_key = implementation_class.__name__

            if generator is None:
                generator = self._assemble_generator(gen_cfg, cache_key=order_key)
                if generator is None:
                    manaprint(f"WARNING: No generator for '{gen_key}' (task='{task_name}')")
                    continue

            if runtime:
                self._inject_runtime_assets(generator, runtime)

            manaprint(f"INFO: Running generator '{gen_key}' for task '{task_name}'...")

            try:
                raw = generator.generate()
                results = self._normalize_generator_output(raw)
                if results:
                    for task_cfg in results:
                        setattr(task_cfg, "_generator_name", task_name)
                        setattr(task_cfg, GENERATOR_REF_ATTR, generator)
                    all_results.extend(results)
                    manaprint(f"INFO:   Generated {len(results)} task config(s)")
            except Exception as e:
                manaprint(f"ERROR: Generator '{gen_key}' failed: {e}")
                import traceback

                traceback.print_exc()

        manaprint(f"INFO: Generator run complete. Total results: {len(all_results)}")
        return all_results

    @staticmethod
    def _normalize_generator_output(raw: Any) -> list[Any]:
        """Ensure generator output is a flat list.

        Old-style generators return ``List[TaskCFG]``.
        ``TaskGenerator`` returns a single ``ManaTask``.
        """
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        return [raw]

    @staticmethod
    def _inject_runtime_assets(generator: BaseGenerator, runtime: dict[str, dict[str, list]]) -> None:
        cfg = getattr(generator, "cfg", None)
        if cfg is None:
            return

        _mapping = [
            ("object_tags", "objects"),
            ("scene_tags", "scenes"),
            ("robot_tags", "robots"),
        ]

        for attr, dim in _mapping:
            if not hasattr(cfg, attr):
                continue
            current = getattr(cfg, attr) or []
            dim_data = runtime.get(dim) or {}
            names = dim_data.get("names", [])
            tags = dim_data.get("tags", [])
            combined = list(dict.fromkeys(names + tags))
            if not current and combined:
                setattr(cfg, attr, combined)
                manaprint(f"INFO:   Injected {attr} from DataCenter: {combined}")

    def build_task(self, task_cfg: Any) -> Any:
        """Return *task_cfg* which is already a ``ManaTask`` (``TaskBase``)."""
        return task_cfg

    @staticmethod
    def create_task_evaluation_runtime(env_obj: Any) -> Any:
        from manaenv.pipeline.evaluation.runtime import create_evaluation_runtime

        return create_evaluation_runtime(env_obj)

    @staticmethod
    def create_task_runtime_support(
        env_obj: Any,
        *,
        extra_key: str | None = None,
    ) -> TaskRuntimeSupport | None:
        runtime = TaskFactory.create_task_evaluation_runtime(env_obj)
        if runtime is None:
            return None

        resolved_extra_key = extra_key
        if resolved_extra_key is None:
            resolved_extra_key = "evaluation_episode_records"

        return TaskRuntimeSupport(
            env_obj=env_obj,
            evaluation_runtime=runtime,
            extra_key=resolved_extra_key,
        )

    def finalize_task(self, builder: Any, scene: Any = None, robot: Any = None) -> Any:
        """Finalize a task for assembly by delegating to the originating generator.

        ``TaskFactory`` no longer owns task lifecycle details. Instead it
        looks up the originating ``TaskGenerator`` (attached during
        ``_run_generators``) and invokes its ``finalize`` hook, passing
        the scene and robot context. Generator subclasses are responsible
        for their own scene-bound finalisation (DIRECT resolves roles /
        visualizer / contact paths; PickAndPlace resolves Asset references
        and constructs the final task object).
        """
        generator = getattr(builder, GENERATOR_REF_ATTR, None)
        if generator is None:
            return self.build_task(builder)
        finalized = generator.finalize(builder, scene=scene, robot=robot)
        if finalized is not builder:
            for attr in ("_generator_name",):
                value = getattr(builder, attr, None)
                if value is not None:
                    try:
                        setattr(finalized, attr, value)
                    except (AttributeError, TypeError):
                        pass
        if hasattr(finalized, GENERATOR_REF_ATTR):
            try:
                delattr(finalized, GENERATOR_REF_ATTR)
            except (AttributeError, TypeError):
                pass
        return self.build_task(finalized)

    @staticmethod
    def _materialize_task_default_visualizer_cfg(builder: Any, scene: Any) -> None:
        """Resolve deferred viewer terms into a native Kit visualizer camera hint."""
        if not hasattr(builder, "runtime_cfgs"):
            return
        raw = builder.runtime_cfgs.get("default_visualizer_cfg")
        if not isinstance(raw, dict):
            return
        term_map = raw.get("terms")
        if not isinstance(term_map, dict) or not term_map:
            return
        visualizer_cfg = TaskFactory._build_default_visualizer_cfg_from_terms(term_map, scene)
        builder.runtime_cfgs["default_visualizer_cfg"] = visualizer_cfg

    @staticmethod
    def _build_default_visualizer_cfg_from_terms(term_map: dict[str, Any], scene: Any):
        from isaaclab_visualizers.kit import KitVisualizerCfg

        import manaenv.terms.viewers as viewer_terms

        for func_name, params in term_map.items():
            p = dict(params or {})
            TaskFactory._bind_viewer_lookat_to_scene_asset(p, scene)
            func = getattr(viewer_terms, func_name, None)
            if func is None:
                continue
            cfg = func(**p)
            if not isinstance(cfg, KitVisualizerCfg):
                continue
            empty = KitVisualizerCfg()
            if getattr(cfg, "eye", ()) != getattr(empty, "eye", ()) or getattr(cfg, "lookat", ()) != getattr(
                empty, "lookat", ()
            ):
                return cfg
        return None

    @staticmethod
    def _bind_viewer_lookat_to_scene_asset(params: dict[str, Any], scene: Any) -> None:
        if scene is None or not hasattr(scene, "assets"):
            return
        assets = scene.assets
        lookat_object = params.get("lookat_object")
        if isinstance(lookat_object, str):
            asset = assets.get(lookat_object)
            if asset is not None:
                params["lookat_object"] = asset
                return
            for candidate in assets.values():
                semantic_tags = getattr(candidate, "semantic_tags", None) or []
                if lookat_object in semantic_tags:
                    params["lookat_object"] = candidate
                    return
            for candidate in assets.values():
                asset_tags = getattr(candidate, "tags", None) or []
                if lookat_object in asset_tags:
                    params["lookat_object"] = candidate
                    return
        elif isinstance(lookat_object, list) and lookat_object:
            for name in lookat_object:
                if not isinstance(name, str):
                    continue
                asset = assets.get(name)
                if asset is not None:
                    params["lookat_object"] = asset
                    break

    @staticmethod
    def _patch_runtime_cfgs(runtime_cfgs: dict[str, Any], tag_to_name: dict[str, Any]) -> None:  # noqa: C901
        """Walk *runtime_cfgs* and replace semantic-tag strings with resolved asset names.

        This handles values embedded inside ``EventTermCfg.params``,
        ``TerminationTermCfg.params``, ``KitVisualizerCfg`` fields, etc.
        Also rewrites ``.../{semantic_tag}`` path segments (e.g. contact-sensor
        ``prim_path`` ``{ENV_REGEX_NS}/pickable``) after roles resolve to
        concrete asset names. List-valued role resolutions skip path substitution.
        The replacement is done in-place on mutable containers so that
        configclass instances (whose fields store plain strings) are also
        updated via ``setattr``.
        """

        _SKIP_REPLACE_KEYS = frozenset({"sequence_order", "button_order"})

        def _replace(obj: Any, *, _key: str | None = None) -> Any:
            if _key in _SKIP_REPLACE_KEYS:
                return obj
            if isinstance(obj, str):
                if obj in tag_to_name:
                    return tag_to_name[obj]
                out = obj
                for tag, name in tag_to_name.items():
                    if isinstance(name, list):
                        continue
                    # ``name`` may be a live Asset object (its ``.name`` is a str)
                    # or already a plain string. Project to the string name so a
                    # path segment substitution never embeds an object ``repr``
                    # (e.g. ``{ENV_REGEX_NS}/<Asset object at 0x...>``).
                    name_str = getattr(name, "name", name)
                    if not isinstance(name_str, str):
                        continue
                    out = out.replace(f"/{tag}", f"/{name_str}")
                return out
            if isinstance(obj, list):
                return [_replace(item) for item in obj]
            if isinstance(obj, dict):
                return {k: _replace(v, _key=k) for k, v in obj.items()}
            return obj

        def _replace_scalar(obj: Any, *, _key: str | None = None) -> Any:
            """Like _replace but collapses list results to first element.

            Metric/termination fields store scene entity *names*. When a role
            value equals that name (``pickable: usb`` → object_cfg ``\"usb\"``),
            ``_replace`` injects the live Asset; project back to ``asset.name``
            so recorders never index the scene with ``str(Asset)``.
            """
            result = _replace(obj, _key=_key)
            if isinstance(obj, str) and isinstance(result, list) and result:
                result = result[0]
            if isinstance(obj, str) and result is not None and not isinstance(result, (str, list)):
                name = getattr(result, "name", None)
                if isinstance(name, str) and name:
                    return name
            return result

        visited: set[int] = set()
        # Resolved role values in ``tag_to_name`` are LIVE, scene-owned Asset
        # objects shared across ALL composite graph nodes. ``_walk`` must never
        # descend into them: rewriting an asset's ``.tags`` here (when a literal
        # role-tag string elsewhere resolves to that asset and _walk recurses in)
        # strips the very tag — e.g. ``shelf_stock`` — that sibling nodes,
        # finalized later against the same scene, still need to resolve. That
        # silently corrupts them to empty lists and raises in the composite event
        # dedup. Pre-seeding their ids makes the guard below skip them; string
        # substitution into runtime_cfgs still happens, only the shared assets
        # are left untouched.
        for _resolved in tag_to_name.values():
            for _asset in _resolved if isinstance(_resolved, list) else [_resolved]:
                if hasattr(_asset, "__dict__"):
                    visited.add(id(_asset))

        def _walk(obj: Any, scalar_only: bool = False) -> None:
            from isaaclab_arena.metrics.metric_base import MetricBase

            # Skip leaves that cannot contain semantic-tag strings and that
            # would otherwise drag _walk into class/module/function graphs
            # (e.g. ``RecorderTermCfg.class_type`` points to a Python class
            # whose ``vars()`` chains into module globals — see recorder
            # ownership unification plan §6, Step 2 runtime regression).
            if isinstance(
                obj, (type, _types.ModuleType, _types.FunctionType, _types.MethodType, _types.BuiltinFunctionType)
            ):
                return
            oid = id(obj)
            if oid in visited:
                return
            visited.add(oid)

            _so = scalar_only or isinstance(obj, MetricBase)
            rep = _replace_scalar if _so else _replace
            if isinstance(obj, dict):
                for k in list(obj.keys()):
                    if k in _SKIP_REPLACE_KEYS:
                        continue
                    obj[k] = rep(obj[k])
                    _walk(obj[k], _so)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _walk(item, _so)
            elif hasattr(obj, "__dict__"):
                for attr_name in list(vars(obj).keys()):
                    if attr_name in _SKIP_REPLACE_KEYS:
                        continue
                    val = getattr(obj, attr_name)
                    new_val = rep(val)
                    if new_val is not val:
                        try:
                            setattr(obj, attr_name, new_val)
                        except (AttributeError, TypeError):
                            pass
                    _walk(getattr(obj, attr_name), _so)

        _walk(runtime_cfgs)

    @staticmethod
    def _expand_runtime_termination_terms(runtime_cfgs: dict[str, Any]) -> None:
        """Expand atomic termination terms after scene-role resolution.

        ``TaskGenerator.generate()`` runs before semantic-tag roles are bound to
        concrete scene assets, so atomic termination params such as
        ``object_cfg="@sortable"`` are still scalar strings during initial
        generation and cannot be expanded there. After ``_patch_runtime_cfgs``
        replaces those strings with concrete asset-name lists, rebuild the
        termination config so each list-backed ``*_cfg`` / ``*_cfgs`` param
        becomes one atomic term per list element (e.g. ``contact_sensor_cfgs``).
        """
        from isaaclab.managers import TerminationTermCfg
        from isaaclab_arena.utils.configclass import make_configclass

        from manaenv.pipeline.generators.task.components.termination_component import TerminationComponent

        termination_terms = runtime_cfgs.get("termination_terms")
        if termination_terms is None:
            return

        term_combine_modes = runtime_cfgs.get("_term_combine_modes", {})
        rebuilt_fields: list[tuple[str, type, Any]] = []
        changed = False

        for field_name, term_cfg in vars(termination_terms).items():
            if not isinstance(term_cfg, TerminationTermCfg):
                rebuilt_fields.append((field_name, type(term_cfg), term_cfg))
                continue

            expanded_term_cfgs, term_changed = TerminationComponent.expand_runtime_term_cfg(term_cfg)
            if len(expanded_term_cfgs) == 1:
                rebuilt_fields.append((field_name, type(expanded_term_cfgs[0]), expanded_term_cfgs[0]))
            else:
                base_key = field_name.replace("__", ".")
                mode = term_combine_modes.get(base_key, term_combine_modes.get(field_name, "and"))
                if field_name == "success":
                    combined_success = TerminationComponent._make_combined_term(expanded_term_cfgs, mode=mode)
                    rebuilt_fields.append((field_name, type(combined_success), combined_success))
                else:
                    for index, expanded_term_cfg in enumerate(expanded_term_cfgs):
                        rebuilt_fields.append((f"{field_name}__{index}", type(expanded_term_cfg), expanded_term_cfg))
            changed = changed or term_changed

        def _collect_failure_term_cfgs(container: Any) -> list:
            # Every episode is Success or Failure — so EVERY termination term that is not
            # `success` IS a failure, and is folded into the stage machine's per-stage
            # failure specs with its own name as the reason. Nothing is opt-in: a task
            # author writes `object_dropped` / `mdp.time_out` as they always have, and the
            # machine attributes them (which stage, which reason) for free. An explicit
            # `failure_<label>` still works — the prefix is stripped to give the label —
            # but it is a way to RENAME a reason, never a way to enable classification.
            specs: list = []
            for fname, fcfg in vars(container).items():
                if not isinstance(fcfg, TerminationTermCfg) or fname == "success":
                    continue
                specs.append((failure_reason_label(fname), fcfg))
            return specs

        if not changed:
            runtime_cfgs["success_term_cfg"] = getattr(termination_terms, "success", None)
            runtime_cfgs["failure_term_cfgs"] = _collect_failure_term_cfgs(termination_terms)
            return

        term_cfg_cls = make_configclass("TerminationCfg", rebuilt_fields)
        runtime_cfgs["termination_terms"] = term_cfg_cls()
        runtime_cfgs["success_term_cfg"] = getattr(runtime_cfgs["termination_terms"], "success", None)
        runtime_cfgs["failure_term_cfgs"] = _collect_failure_term_cfgs(runtime_cfgs["termination_terms"])

    @staticmethod
    def _resolve_roles_from_scene(roles: dict[str, Any], scene: Any) -> dict[str, Any]:
        """Replace role tag values with actual asset names from *scene*.

        For each role value:
        - If ``find_assets_by_tag`` finds matches in asset ``tags``, replace with the asset
          name(s).  A single match yields a string; multiple matches yield
          a list.
        - If no match is found the value is kept as-is.
        """
        from manaenv.tasks.utils import find_assets_by_tag

        resolved: dict[str, Any] = {}
        for role_name, tag_or_names in roles.items():
            if isinstance(tag_or_names, str):
                assets = find_assets_by_tag(scene, tag_or_names)
                if len(assets) == 1:
                    resolved[role_name] = assets[0].name
                    manaprint(f"INFO:   Role '{role_name}': tag '{tag_or_names}' -> '{assets[0].name}'")
                elif len(assets) > 1:
                    names = [a.name for a in assets]
                    resolved[role_name] = names
                    manaprint(f"INFO:   Role '{role_name}': tag '{tag_or_names}' -> {names}")
                else:
                    resolved[role_name] = tag_or_names
            elif isinstance(tag_or_names, list):
                resolved[role_name] = tag_or_names
            else:
                resolved[role_name] = tag_or_names
        return resolved

    def cleanup(self):
        gen_count = len(self._generators)

        for name, generator in self._generators.items():
            cleanup = getattr(generator, "cleanup", None)
            if callable(cleanup):
                cleanup()

        self._orders.clear()
        self._generators.clear()

        manaprint(f"INFO: TaskFactory cleanup - released {gen_count} generator(s)")
