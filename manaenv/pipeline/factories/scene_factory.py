"""Scene factory for assembling scene generators and fulfilling scene orders.

Generator and Component classes are discovered via the global lazy registry --
no direct import of concrete generator/component types needed.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any

from manaenv.pipeline.data_types import DataBucket
from manaenv.pipeline.factory import BaseFactory, BaseFactoryCFG
from manaenv.pipeline.generators.scene.component import SceneComponentCFG
from manaenv.pipeline.generators.scene_generator import SceneGeneratorCFG
from manaenv.pipeline.registry.global_provider import global_registry
from manaenv.utils.config_io import create_config_from_dict_strict
from manaenv.utils.debug_print import manaprint
from manaenv.utils.fingerprint import constraints_fingerprint
from manaenv.utils.semantic_tags import normalize_semantic_tags

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

    from manaenv.pipeline.generators.scene_generator import SceneGenerator


@dataclass
class SceneRuntimeCFG:
    """SceneFactory-level runtime overrides for scene generator CFGs."""

    sim_fps: int | None = None
    render_decremental: int | None = None
    simulation_decimation: int | None = None
    seed: int | None = None


@dataclass
class SceneRenderSettingsCFG:
    """SceneFactory-level renderer overrides applied to every scene order."""

    antialiasing_mode: str | None = None
    enable_dlssg: bool | None = None
    dlssg_multiplier: int | None = None
    dlss_mode: int | None = None
    carb_settings: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        authored = {
            render_field.name: value
            for render_field in fields(self)
            if (value := getattr(self, render_field.name)) is not None
        }
        if not authored:
            return

        # Reuse the concrete component's validation/defaulting contract without
        # importing concrete scene components at module import time.
        from manaenv.pipeline.generators.scene.components.render_settings_component import RenderSettingsComponentCFG

        validated = RenderSettingsComponentCFG(**authored)
        for field_name in authored:
            setattr(self, field_name, getattr(validated, field_name))
        if self.antialiasing_mode == "":
            self.antialiasing_mode = None
        if "enable_dlssg" in authored or "dlssg_multiplier" in authored:
            self.enable_dlssg = validated.enable_dlssg
            self.dlssg_multiplier = validated.dlssg_multiplier


@dataclass
class SceneFactoryCFG(BaseFactoryCFG):
    class_type: Any = None
    type: str = "default_scene_factory"
    scene_runtime: SceneRuntimeCFG = field(default_factory=SceneRuntimeCFG)
    render_settings: SceneRenderSettingsCFG = field(default_factory=SceneRenderSettingsCFG)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = SceneFactory


SceneOrder = tuple[str, "SceneGeneratorCFG", list[SceneComponentCFG]]


@dataclass
class SceneOrderCFG:
    """One scene order: generator settings plus authored component CFGs."""

    name: str = ""
    scene_generator: SceneGeneratorCFG = field(default_factory=SceneGeneratorCFG)
    objects_component: list[SceneComponentCFG] = field(default_factory=list)
    background_component: SceneComponentCFG | None = None
    render_settings_component: SceneComponentCFG | None = None
    lighting_component: SceneComponentCFG | None = None

    @property
    def components(self) -> list[SceneComponentCFG]:
        singles = (self.background_component, self.render_settings_component, self.lighting_component)
        return [*self.objects_component, *(cfg for cfg in singles if cfg is not None)]


def _normalize_scene_order_item(item: Any, *, index: int) -> SceneOrderCFG:
    """Return the typed ``SceneOrderCFG`` for one raw scene-order item.

    Standard scene-order dicts reflect straight into ``SceneOrderCFG``; schema-vNext
    scene configs (``{background, layout, variants}``) are first materialized into the
    standard shape by the scene-domain bridge, then reflected. The reflective
    translator stays a pure YAML->CFG reflector — this compatibility normalization is
    owned by the scene layer: CFG-Instance reflection is preserved (via
    ``create_config_from_dict_strict``), tree reflection is not forced. Mirrors
    ``TaskOrdersCFG`` / ``_normalize_task_order_item``.
    """
    if isinstance(item, SceneOrderCFG):
        return item
    if not isinstance(item, dict):
        raise TypeError(
            "SceneOrdersCFG.orders entries must be SceneOrderCFG objects or mappings; "
            f"entry {index} is {type(item).__name__}"
        )
    # Lazy imports: legacy scene orders never pull in schema-vNext / config_io eagerly.
    from manaenv.pipeline.generators.scene_layout_solver.scene_order_bridge import (
        is_vnext_scene_order,
        vnext_scene_order_from_config,
    )

    if is_vnext_scene_order(item):
        item = vnext_scene_order_from_config(item)

    # Materialized / raw component dicts need nested key-dispatch reflection so
    # fields like ``render_settings_component.antialiasing_mode`` land on the
    # concrete CFG (RenderSettingsComponentCFG), not the SceneComponentCFG base.
    # Idempotent for orders already reflected by the translator upstream.
    from manaenv.pipeline.translator.reflective_translator import _reflect_nested_tree

    item = _reflect_nested_tree(item, context=f"scene_orders.orders[{index}]")

    order = create_config_from_dict_strict(item, SceneOrderCFG, context=f"scene_orders.orders[{index}]")
    return _promote_render_antialiasing_mode(order)


def _promote_render_antialiasing_mode(order: SceneOrderCFG) -> SceneOrderCFG:
    """Lift ``render_settings_component.antialiasing_mode`` onto the generator CFG."""
    rs = order.render_settings_component
    mode = str(getattr(rs, "antialiasing_mode", "") or "").strip() if rs is not None else ""
    if not mode:
        return order
    from manaenv.pipeline.generators.scene_generator import normalize_antialiasing_mode

    order.scene_generator.antialiasing_mode = normalize_antialiasing_mode(mode)
    return order


@dataclass
class SceneOrdersCFG:
    """Typed scene order section.

    ``orders`` is loosely typed (``list[Any]``) so the reflective translator leaves the
    raw items untouched; ``__post_init__`` reflects each on demand (materializing any
    schema-vNext orders first). Same pattern as ``TaskOrdersCFG``.
    """

    orders: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.orders = [_normalize_scene_order_item(item, index=index) for index, item in enumerate(self.orders or [])]


class SceneFactory(BaseFactory):
    """Factory that holds long-lived SceneGenerators and produces scenes.

    Generators are discovered via the global lazy registry, created once via
    receive_orders and reused across fulfill_orders calls.
    Components are reloaded per batch.
    """

    nums_merge_policy: str = "max"

    def __init__(self):
        super().__init__()
        self.cfg: SceneFactoryCFG = SceneFactoryCFG()
        self._orders: list[SceneOrder] = []
        self._generators: dict[str, SceneGenerator] = {}
        self.data_center = None

    def setup(self, cfg: SceneFactoryCFG | None = None) -> None:
        """Setup SceneFactory orchestration policy."""
        self.cfg = cfg if cfg is not None else SceneFactoryCFG()

    # ------------------------------------------------------------------
    # Order intake
    # ------------------------------------------------------------------

    def receive_orders(self, orders: SceneOrdersCFG):
        if not isinstance(orders, SceneOrdersCFG):
            raise TypeError(f"SceneFactory.receive_orders expected SceneOrdersCFG, got {type(orders).__name__}")
        self._orders = [
            self._apply_runtime_overrides((order.scene_generator.scene_type, order.scene_generator, order.components))
            for order in orders.orders
        ]
        manaprint(f"INFO: SceneFactory received {len(self._orders)} order(s)")
        for scene_type, gen_cfg, comp_cfgs in self._orders:
            manaprint(f"INFO:   - {scene_type}: {len(comp_cfgs)} components, nums={gen_cfg.nums}")
            if scene_type not in self._generators:
                self._generators[scene_type] = self._create_generator(gen_cfg, comp_cfgs)

    def _apply_runtime_overrides(self, order: SceneOrder) -> SceneOrder:
        """Apply independent factory-level timestep and renderer policies."""
        scene_type, gen_cfg, comp_cfgs = order
        runtime_cfg = self.cfg.scene_runtime
        runtime_overrides = {
            runtime_field.name: value
            for runtime_field in fields(runtime_cfg)
            if (value := getattr(runtime_cfg, runtime_field.name)) is not None
        }
        render_cfg = self.cfg.render_settings
        render_overrides = {
            render_field.name: value
            for render_field in fields(render_cfg)
            if (value := getattr(render_cfg, render_field.name)) is not None
        }
        if not runtime_overrides and not render_overrides:
            return order
        patched_cfg = copy.copy(gen_cfg)
        generator_field_names = (
            {field.name for field in fields(gen_cfg)} if is_dataclass(gen_cfg) else set(vars(gen_cfg))
        )
        for field_name, value in runtime_overrides.items():
            if field_name in generator_field_names:
                setattr(patched_cfg, field_name, value)

        antialiasing_mode = render_overrides.get("antialiasing_mode")
        if antialiasing_mode is not None and "antialiasing_mode" in generator_field_names:
            patched_cfg.antialiasing_mode = antialiasing_mode

        patched_components = self._apply_render_settings_overrides(comp_cfgs, render_overrides)
        return scene_type, patched_cfg, patched_components

    @staticmethod
    def _apply_render_settings_overrides(
        comp_cfgs: list[SceneComponentCFG],
        overrides: dict[str, Any],
    ) -> list[SceneComponentCFG]:
        """Patch or add the renderer component for factory-wide settings."""
        if not overrides:
            return comp_cfgs

        from manaenv.pipeline.generators.scene.components.render_settings_component import RenderSettingsComponentCFG

        patched_components = list(comp_cfgs)
        replace_overrides = dict(overrides)
        if replace_overrides.get("enable_dlssg") is False and "dlssg_multiplier" not in replace_overrides:
            # A disable override must also clear an authored multiplier, or the
            # component CFG validation would correctly reject the contradictory pair.
            replace_overrides["dlssg_multiplier"] = None
        for index, component_cfg in enumerate(patched_components):
            if isinstance(component_cfg, RenderSettingsComponentCFG):
                # ``replace`` reruns the CFG validation/defaulting hooks (e.g.
                # enable_dlssg=True -> the documented x2 multiplier).
                patched_components[index] = replace(component_cfg, **replace_overrides)
                return patched_components

        patched_components.append(RenderSettingsComponentCFG(**overrides))
        return patched_components

    # ------------------------------------------------------------------
    # Generator lifecycle
    # ------------------------------------------------------------------

    def _create_generator(self, gen_cfg: SceneGeneratorCFG, comp_cfgs: list[SceneComponentCFG]) -> SceneGenerator:
        """Create the generator declared by ``gen_cfg`` and inject asset registry."""
        from manaenv.pipeline.generators.scene_generator import SceneGenerator

        gen_cls = self._resolve_generator_class(gen_cfg) or SceneGenerator
        if not isinstance(gen_cls, type) or not issubclass(gen_cls, SceneGenerator):
            raise TypeError(f"Invalid scene generator class_type on {type(gen_cfg).__name__}: {gen_cls!r}")
        generator = gen_cls(gen_cfg)
        generator.set_asset_registry(self.data_center.asset_registry)
        self._reload_components(generator, comp_cfgs)
        manaprint(f"INFO:   Assembled generator '{gen_cfg.scene_type}' " f"with {len(comp_cfgs)} component cfgs")
        return generator

    @staticmethod
    def _resolve_generator_class(gen_cfg: SceneGeneratorCFG) -> type | None:
        """Resolve the SceneGenerator class for ``gen_cfg``.

        A non-empty ``generator_type`` names a registered generator CFG symbol
        whose ``class_type`` is used; otherwise fall back to the CFG's own
        ``class_type`` (default ``SceneGenerator``). Empty ``generator_type``
        preserves the prior behavior for all existing scene orders.
        """
        generator_type = str(getattr(gen_cfg, "generator_type", "") or "").strip()
        if generator_type:
            cfg_cls = global_registry.get_cfg(generator_type)
            if cfg_cls is None:
                raise ValueError(f"Unknown scene generator_type={generator_type!r}")
            return getattr(cfg_cls, "class_type", None)
        return getattr(gen_cfg, "class_type", None)

    @staticmethod
    def _reload_components(generator: SceneGenerator, comp_cfgs: list[SceneComponentCFG]) -> None:
        """Reload translated YAML component CFGs onto the generator.

        SceneGenerator owns CFG -> component instancing so its lifecycle
        mirrors TaskGenerator: factory assembly stops at generator + CFGs.
        """
        generator.set_component_cfgs(comp_cfgs)
        generator.components = []

    # ------------------------------------------------------------------
    # Order fulfillment
    # ------------------------------------------------------------------

    def fulfill_orders(self, scene_type: str | None = None) -> list[Scene]:
        tasks = self.data_center.get(DataBucket.TASKS)
        orders = [o for o in self._orders if o[0] == scene_type] if scene_type else self._orders
        if scene_type and not orders:
            manaprint(f"WARNING: No orders for scene type '{scene_type}'")
            return []

        manaprint(f"INFO: Fulfilling {len(orders)} order(s) with {len(tasks)} task(s)...")
        all_scenes: list[Scene] = []

        for order_scene_type, gen_cfg, comp_cfgs in orders:
            manaprint(f"INFO: Processing template '{order_scene_type}'...")
            fillable = self._has_fillable_components(comp_cfgs)

            if not tasks or not fillable:
                # No task-aware fillable slots (all component tags are fixed):
                # still generate static scenes instead of skipping the order.
                if tasks and not fillable:
                    manaprint(f"INFO:   '{order_scene_type}' has no fillable components; " "generating static variants")
                all_scenes.extend(self._generate_batch(order_scene_type, gen_cfg, comp_cfgs, None, gen_cfg.nums))
                continue

            merged = self._merge_task_orders(tasks, gen_cfg.nums, self.nums_merge_policy)
            manaprint(f"INFO:   Merged {len(tasks)} task(s) into {len(merged)} group(s)")

            candidates: list[Scene] = []
            for constraints, nums in merged:
                manaprint(f"INFO:   Group constraints={constraints}, nums={nums}")
                candidates.extend(self._generate_batch(order_scene_type, gen_cfg, comp_cfgs, constraints, nums))
            unique = self._deduplicate_scenes(candidates)
            manaprint(f"INFO:   Dedup: {len(candidates)} -> {len(unique)} unique")
            all_scenes.extend(unique)

        manaprint(f"INFO: Order fulfillment complete. Total: {len(all_scenes)} scene(s)")
        return all_scenes

    def _generate_batch(
        self,
        order_scene_type: str,
        gen_cfg: SceneGeneratorCFG,
        comp_cfgs_template: list[SceneComponentCFG],
        constraints: dict[str, Any] | None,
        nums: int,
    ) -> list[Scene]:
        generator = self._generators.get(order_scene_type)
        if generator is None:
            manaprint(f"ERROR: No generator for '{order_scene_type}'")
            return []

        comp_cfgs = copy.deepcopy(comp_cfgs_template)
        if constraints:
            self._apply_task_constraints(type(generator), comp_cfgs, constraints)

        scenes: list[Scene] = []
        for i in range(nums):
            generator.config = gen_cfg
            self._reload_components(generator, comp_cfgs)
            self._apply_scene_rng_seed(getattr(gen_cfg, "seed", None), variant_index=i)
            scene = generator.generate(variant_index=i)
            if scene is not None:
                scene._generator_name = order_scene_type
                scene.scene_type = order_scene_type
                scene._manaenv_scene_generator = generator
                # Contract with scene_layout_solver (integration.py): gen_cfg.metadata
                # carries schema-vNext payloads that task finalize reads from Scene.
                # Component-written scene.metadata (e.g. navigation) wins on conflict.
                merged_metadata = dict(getattr(gen_cfg, "metadata", {}) or {})
                merged_metadata.update(getattr(scene, "metadata", {}) or {})
                scene.metadata = merged_metadata
                scene.tags = self._collect_scene_tags(comp_cfgs)
                scene._constraint_fp = constraints_fingerprint(constraints) if constraints else ""
                # Keep an outer scene-epoch producer on the product.  The
                # schema-vNext ``active_only`` topology has one concrete USD
                # slot, so changing the selected pool asset requires a fresh
                # scene/materialization instead of a reset-time slot switch.
                # The callback is intentionally private and process-local: it
                # is an execution aid, not serialized scene metadata.
                source = (getattr(scene, "metadata", {}) or {}).get("schema_vnext_scene")
                if isinstance(source, dict):
                    scene._manaenv_scene_epoch_factory = lambda episode_index=0, _scene=scene, _source=copy.deepcopy(
                        source
                    ), _scene_type=order_scene_type, _constraints=copy.deepcopy(
                        constraints
                    ): self.regenerate_scene_epoch(
                        _scene,
                        source=_source,
                        scene_type=_scene_type,
                        constraints=_constraints,
                        episode_index=int(episode_index),
                    )
                scenes.append(scene)
                manaprint(
                    f"INFO:     Variant {i + 1}/{nums} -> " f"{len(scene.assets)} assets: {list(scene.assets.keys())}"
                )
        return scenes

    def regenerate_scene_epoch(
        self,
        template_scene: Any,
        *,
        source: dict[str, Any],
        scene_type: str,
        constraints: dict[str, Any] | None = None,
        episode_index: int = 0,
    ) -> Any:
        """Materialize and generate the next schema-vNext scene epoch.

        ``active_only`` deliberately has no inactive USD slots.  Re-running
        the public vNext materializer here selects a fresh concrete asset and
        reconstructs the static layout before the outer runner calls
        :meth:`AssembledEnvironment.rebuild_scene_epoch`.
        """
        if not isinstance(source, dict):
            return None

        from manaenv.pipeline.generators.scene_layout_solver.scene_order_bridge import vnext_scene_order_from_config

        layout = source.get("layout") if isinstance(source.get("layout"), dict) else {}
        definition = source.get("definition") if isinstance(source.get("definition"), dict) else {}
        seed = layout.get("seed", definition.get("seed"))
        epoch_rng = random.Random(int(seed) + int(episode_index)) if seed is not None else None
        materialized = vnext_scene_order_from_config(source, rng=epoch_rng)
        order = _normalize_scene_order_item(materialized, index=0)
        order_scene_type, gen_cfg, comp_cfgs = self._apply_runtime_overrides(
            (scene_type, order.scene_generator, order.components)
        )
        template_generator = getattr(template_scene, "_manaenv_scene_generator", None)
        generator_cls = (
            type(template_generator) if template_generator is not None else self._resolve_generator_class(gen_cfg)
        )
        generator_cls = generator_cls or getattr(gen_cfg, "class_type", None)
        if generator_cls is None:
            from manaenv.pipeline.generators.scene_generator import SceneGenerator

            generator_cls = SceneGenerator
        if constraints:
            self._apply_task_constraints(generator_cls, comp_cfgs, constraints)
        generator = generator_cls(gen_cfg)
        generator.set_asset_registry(self.data_center.asset_registry)
        self._reload_components(generator, comp_cfgs)
        self._apply_scene_rng_seed(getattr(gen_cfg, "seed", None), variant_index=episode_index)
        new_scene = generator.generate(variant_index=episode_index)
        if new_scene is None:
            return None
        new_scene._generator_name = order_scene_type
        new_scene.scene_type = order_scene_type
        merged_metadata = dict(getattr(gen_cfg, "metadata", {}) or {})
        merged_metadata.update(getattr(new_scene, "metadata", {}) or {})
        new_scene.metadata = merged_metadata
        new_scene.tags = self._collect_scene_tags(comp_cfgs)
        new_scene._constraint_fp = constraints_fingerprint(constraints) if constraints else ""
        new_scene._manaenv_scene_generator = generator
        # Carry the producer forward so every subsequent episode can rebuild
        # from the immutable source payload, not from the previous materialized
        # asset selection.
        new_scene._manaenv_scene_epoch_factory = (
            lambda next_index=episode_index + 1, _scene=new_scene, _source=copy.deepcopy(
                source
            ), _scene_type=order_scene_type, _constraints=copy.deepcopy(constraints): self.regenerate_scene_epoch(
                _scene,
                source=_source,
                scene_type=_scene_type,
                constraints=_constraints,
                episode_index=int(next_index),
            )
        )
        return new_scene

    @staticmethod
    def _apply_scene_rng_seed(seed: int | None, *, variant_index: int = 0) -> None:
        """Seed procedural scene generation such as shelf-fill layouts."""
        from manaenv.pipeline.generators.scene_layout_solver.common.reproducibility import stable_seed
        from manaenv.pipeline.runtime import master_seed as master_seed_runtime

        if seed is not None:
            effective = int(stable_seed(int(seed), master_seed_runtime.NS_SCENE_VARIANT, int(variant_index)))
        else:
            derived = master_seed_runtime.derive(master_seed_runtime.NS_SCENE_VARIANT, int(variant_index))
            if derived is None:
                return
            effective = int(derived)
        random.seed(effective)
        manaprint(f"INFO: Scene procedural layout seed={effective}")

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_order_count(self) -> int:
        return len(self._orders)

    def get_generator(self, scene_type: str) -> SceneGenerator | None:
        return self._generators.get(scene_type)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_task_orders(tasks: list, base_nums: int, policy: str = "max") -> list[tuple[dict[str, Any], int]]:
        groups: dict[str, dict[str, Any]] = {}
        for task in tasks:
            sc = getattr(task, "scene_constraints", {}) or {}
            fp = constraints_fingerprint(sc)
            if fp not in groups:
                groups[fp] = {"constraints": sc, "count": 0}
            groups[fp]["count"] += 1
        merged = []
        for info in groups.values():
            count = info["count"]
            nums = {"sum": count * base_nums, "min": base_nums}.get(policy, max(count, base_nums))
            merged.append((info["constraints"], nums))
        return merged

    @staticmethod
    def _has_fillable_components(comp_cfgs: list[SceneComponentCFG]) -> bool:
        return any(not getattr(cfg, "tags", "") and not getattr(cfg, "gen_mode", "") for cfg in comp_cfgs)

    @staticmethod
    def _deduplicate_scenes(scenes: list) -> list:
        seen: set = set()
        unique: list = []
        for scene in scenes:
            parts: list[str] = []
            for name in sorted(scene.assets.keys()):
                asset = scene.assets[name]
                pose = getattr(asset, "initial_pose", None)
                if pose is not None:
                    pos = tuple(round(v, 3) for v in pose.position_xyz)
                    rot = tuple(round(v, 3) for v in pose.rotation_xyzw)
                    parts.append(f"{name}@{pos}{rot}")
                else:
                    parts.append(name)
            fp = tuple(parts)
            if fp not in seen:
                seen.add(fp)
                unique.append(scene)
        return unique

    @staticmethod
    def _apply_task_constraints(gen_cls: type, comp_cfgs: list[SceneComponentCFG], demands: dict[str, Any]) -> None:
        if not isinstance(demands, dict):
            manaprint(
                f"INFO:   scene_constraints is {type(demands).__name__}, "
                "skipping component-level constraint injection"
            )
            return
        extras: list[SceneComponentCFG] = []
        for cfg in comp_cfgs:
            if getattr(cfg, "asset", "") or getattr(cfg, "asset_tag", "") or getattr(cfg, "generator", ""):
                continue
            role = global_registry.role_of(cfg)
            demand = demands.get(role)
            if demand is None:
                continue
            raw_assets = demand.get("asset", demand.get("asset_tag", "")) if isinstance(demand, dict) else demand
            asset_list = (
                [raw_assets]
                if isinstance(raw_assets, str) and raw_assets
                else list(raw_assets) if not isinstance(raw_assets, str) else []
            )
            if not asset_list:
                continue
            if isinstance(demand, dict) and "asset_tag" in demand:
                cfg.asset_tag = asset_list[0]
                selector_name = "asset_tag"
            else:
                cfg.asset = asset_list[0]
                selector_name = "asset"
            manaprint(f"INFO:   Task -> [{role}] {selector_name}='{asset_list[0]}'")
            for extra_asset in asset_list[1:]:
                clone = copy.deepcopy(cfg)
                if selector_name == "asset_tag":
                    clone.asset_tag = extra_asset
                else:
                    clone.asset = extra_asset
                extras.append(clone)
                manaprint(f"INFO:   Task -> [{role}+] {selector_name}='{extra_asset}'")
        comp_cfgs.extend(extras)

    @staticmethod
    def _collect_scene_tags(comp_cfgs: list[SceneComponentCFG]) -> list[str]:
        tags: list[str] = []
        for cfg in comp_cfgs:
            for cfg_asset_field in ("asset", "asset_tag", "generator"):
                cfg_value = getattr(cfg, cfg_asset_field, None)
                if cfg_value:
                    tags.extend(cfg_value if isinstance(cfg_value, list) else [cfg_value])
            semantic_tags = getattr(cfg, "semantic_tags", None)
            if semantic_tags:
                tags.extend(tag.name for tag in normalize_semantic_tags(semantic_tags))
        return list(dict.fromkeys(tags))

    def cleanup(self):
        manaprint(f"INFO: SceneFactory cleanup - releasing {len(self._generators)} generator(s)")
        self._generators.clear()
        self._orders.clear()
