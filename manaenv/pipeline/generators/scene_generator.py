"""Scene generator that builds Arena Scene objects from components."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from manaenv.pipeline.generator import GenerationCompilation, ManaGenerator
from manaenv.pipeline.generators.scene.component import SceneComponent, SceneComponentCFG
from manaenv.utils.debug_print import manaprint

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

# Isaac Lab RenderCfg.antialiasing_mode contract.
ANTIALIASING_MODES = ("Off", "FXAA", "DLSS", "TAA", "DLAA")
# Carb ``/rtx/post/aa/op`` values confirmed by Isaac Lab render tests.
ANTIALIASING_AA_OP = {"DLSS": 3, "DLAA": 4}
DEFAULT_ANTIALIASING_MODE = "DLSS"


def normalize_antialiasing_mode(value: Any, *, default: str = DEFAULT_ANTIALIASING_MODE) -> str:
    """Return a canonical Isaac Lab antialiasing mode name."""
    raw = str(value if value is not None else default).strip()
    if not raw:
        raw = default
    lookup = {mode.lower(): mode for mode in ANTIALIASING_MODES}
    canonical = lookup.get(raw.lower())
    if canonical is None:
        manaprint(f"WARNING: unknown antialiasing_mode={raw!r}; using {default}")
        return default
    return canonical


@dataclass
class SceneGeneratorCFG:
    """Configuration for scene generator."""

    class_type: type[SceneGenerator] | None = None
    # Optional CFG-symbol name of a SceneGenerator subclass to use (e.g.
    # "NavigationSceneGenerator"). Empty -> resolve via ``class_type`` / default
    # SceneGenerator, so existing scene orders are unaffected.
    generator_type: str = ""
    scene_type: str = ""
    nums: int = 1
    sim_fps: int = 240
    render_decremental: int = 8
    # Isaac Lab env: physics substeps per env.step; None = do not override (Arena / task default).
    simulation_decimation: int | None = None
    # RTX antialiasing: Off | FXAA | DLSS | TAA | DLAA. YAML:
    # ``render_settings.antialiasing_mode``, ``render_settings_component.antialiasing_mode``,
    # or ``definition`` / ``scene_generator.antialiasing_mode``.
    antialiasing_mode: str = DEFAULT_ANTIALIASING_MODE
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = SceneGenerator
        self.antialiasing_mode = normalize_antialiasing_mode(self.antialiasing_mode)


@dataclass
class SceneGenerationCompilation(GenerationCompilation):
    """Compiled scene component CFGs for one variant."""

    component_cfgs: list[SceneComponentCFG] = field(default_factory=list)
    variant_index: int = 0
    components: list[Any] = field(default_factory=list)


class SceneGenerator(ManaGenerator):
    """Generator that combines components to build Arena Scene objects."""

    def __init__(self, config: SceneGeneratorCFG):
        super().__init__(config)
        self._asset_registry: Any | None = None
        self._component_cfgs: list[SceneComponentCFG] = []
        self._next_variant_index = 0

    @property
    def config(self) -> SceneGeneratorCFG:
        return self.cfg

    @config.setter
    def config(self, value: SceneGeneratorCFG):
        self.cfg = value

    def set_asset_registry(self, registry: Any):
        self._asset_registry = registry

    def set_component_cfgs(self, component_cfgs: list[SceneComponentCFG]) -> None:
        self._component_cfgs = list(component_cfgs or [])

    def compile(self, **kwargs) -> SceneGenerationCompilation:
        """Compile translated YAML component data into scene component CFGs."""
        variant_index = kwargs.get("variant_index")
        if variant_index is None:
            variant_index = self._next_variant_index
            self._next_variant_index += 1
        component_cfgs = kwargs.get("component_cfgs")
        if component_cfgs is None:
            component_cfgs = self._component_cfgs
        if not component_cfgs and self.components:
            component_cfgs = [
                component.cfg for component in self.sorted_components() if getattr(component, "cfg", None)
            ]
        return SceneGenerationCompilation(
            component_cfgs=self.sort_components(list(component_cfgs or [])),
            variant_index=int(variant_index),
        )

    def instancing(self, compiled: SceneGenerationCompilation, **kwargs) -> Scene:
        """Instantiate components from CFGs and return an Arena Scene."""
        del kwargs
        return self._instantiate_scene(compiled)

    def _instantiate_scene(
        self,
        compiled: SceneGenerationCompilation,
        *,
        base_assets: list[Any] | None = None,
    ) -> Scene:
        """Instantiate components and build a Scene, optionally with prebuilt assets."""
        if self._asset_registry is None:
            raise RuntimeError("SceneGenerator requires an AssetRegistry from DataCenter.")

        components = self._instantiate_components(compiled)
        self.components = list(components)

        manaprint(f"INFO: Building scene '{self.config.scene_type}' " f"with {len(components)} components")

        assets: list[Any] = list(base_assets or [])
        scene_cfg_patches: dict[str, Any] = {}
        scene_event_entries: dict[str, Any] = {}
        render_settings: dict[str, Any] = {}
        dlssg_multiplier: int | None = None
        scene_metadata_entries: dict[str, Any] = {}
        for component in components:
            try:
                set_asset_registry = getattr(component, "set_asset_registry", None)
                if callable(set_asset_registry):
                    set_asset_registry(self._asset_registry)
                component_result = self._generate_component(component, compiled)
                if component_result is not None and getattr(component_result, "_is_render_settings", False):
                    render_settings.update(component_result.render_cfg_dict)
                    component_dlssg_multiplier = getattr(component_result, "dlssg_multiplier", None)
                    if component_dlssg_multiplier is not None:
                        dlssg_multiplier = int(component_dlssg_multiplier)
                    manaprint(
                        f"INFO:   {type(component).__name__} -> {len(component_result.render_cfg_dict)} render settings"
                    )
                elif component_result is not None and getattr(component_result, "_is_scene_config", False):
                    scene_cfg_patches.update(component_result.entries)
                    if getattr(component_result, "event_entries", None):
                        scene_event_entries.update(component_result.event_entries)
                    component_scene_metadata = getattr(component_result, "scene_metadata", None)
                    if isinstance(component_scene_metadata, dict) and component_scene_metadata:
                        scene_metadata_entries.update(component_scene_metadata)
                    component_assets = getattr(component_result, "assets", None)
                    if isinstance(component_assets, list) and component_assets:
                        self._append_component_assets(assets, component, component_assets)
                    manaprint(
                        f"INFO:   {type(component).__name__} -> scene config {list(component_result.entries.keys())}"
                    )
                elif component_result is not None and isinstance(component_result, list):
                    appended = self._append_component_assets(assets, component, component_result)
                    if appended > 0:
                        manaprint(f"INFO:   {type(component).__name__} -> {appended} assets")
                elif component_result is not None and hasattr(component_result, "name"):
                    self._append_component_assets(assets, component, [component_result])
                    manaprint(f"INFO:   {type(component).__name__} -> {component_result.name}")
                elif component_result is not None:
                    manaprint(f"INFO:   {type(component).__name__} -> config")
            except Exception as exc:
                component_id = str(getattr(getattr(component, "cfg", None), "id", "") or "")
                identity = f" id={component_id!r}" if component_id else ""
                raise RuntimeError(
                    f"Scene '{self.config.scene_type}' component {type(component).__name__}{identity} failed: {exc}"
                ) from exc
        return self._build_scene(
            assets,
            scene_cfg_patches,
            scene_event_entries,
            render_settings,
            components,
            scene_metadata_entries,
            dlssg_multiplier,
        )

    def finalize(self, instance: Scene, **kwargs) -> Scene:
        """Apply late-bound scene post-processing after assembly context exists."""
        del kwargs
        return instance

    def _build_scene(
        self,
        assets: list[Any],
        scene_cfg_patches: dict[str, Any],
        scene_event_entries: dict[str, Any],
        render_settings: dict[str, Any],
        components: list[SceneComponent],
        scene_metadata_entries: dict[str, Any] | None = None,
        dlssg_multiplier: int | None = None,
    ) -> Scene:
        """Build the Arena Scene instance produced by ``instancing``."""
        from isaaclab_arena.scene.scene import Scene

        all_assets = self._deduplicate_asset_names(assets)
        # scene_layout_solver hook call
        all_assets = self._apply_scene_layout_solver(all_assets)

        scene = Scene(assets=all_assets)
        if scene_event_entries:
            from isaaclab_arena.utils.configclass import make_configclass

            fields = [(name, type(term_cfg), term_cfg) for name, term_cfg in scene_event_entries.items()]
            scene.events_cfg = make_configclass("SceneEventsCfg", fields)()
        scene.sim_fps = int(self.config.sim_fps)
        scene.render_decremental = int(self.config.render_decremental)
        if self.config.simulation_decimation is not None:
            scene.simulation_decimation = int(self.config.simulation_decimation)
        antialiasing_mode = normalize_antialiasing_mode(self.config.antialiasing_mode)
        scene.antialiasing_mode = antialiasing_mode
        scene.render_carb_settings = render_settings
        scene.dlssg_multiplier = dlssg_multiplier
        if scene_metadata_entries:
            scene.metadata = {
                **(getattr(scene, "metadata", {}) or {}),
                **dict(scene_metadata_entries),
            }

        # IsaacLab Arena 3.0 removed ``Scene.modify_env_cfg`` (and the
        # ``task/embodiment/scene.modify_env_cfg`` chain in ArenaEnvBuilder).
        # The replacement extension point is ``IsaacLabArenaEnvironment.
        # env_cfg_callback``, applied at the end of ``compose_manager_cfg()``
        # (before ``gym.make`` / ``SimulationContext.__init__``). We stash the
        # modifier on the scene under a manaenv-owned attribute; the assembly
        # factory bridges it into ``env_cfg_callback`` when it builds the
        # ``IsaacLabArenaEnvironment``. ``_orig`` chains any modifier already
        # set (e.g. by a planner wrapping this one) instead of arena's old
        # no-op hook.
        _orig = getattr(scene, "_manaenv_env_cfg_modifier", lambda env_cfg: env_cfg)

        def _patched(
            env_cfg,
            patches=scene_cfg_patches,
            orig=_orig,
            sim_fps=self.config.sim_fps,
            render_decremental=self.config.render_decremental,
            simulation_decimation=self.config.simulation_decimation,
            _render_settings=render_settings,
            _antialiasing_mode=antialiasing_mode,
        ):
            env_cfg = orig(env_cfg)
            if sim_fps:
                env_cfg.sim.dt = 1.0 / float(sim_fps)
            if render_decremental:
                env_cfg.sim.render_interval = int(render_decremental)
            if simulation_decimation is not None:
                env_cfg.decimation = int(simulation_decimation)
            if _render_settings:
                # IsaacLab 3.0 removed ``SimulationCfg.render`` (RenderCfg); global RTX
                # settings are now written straight to carb via the isaac_rtx helper.
                # This modifier runs right before gym.make, preserving the old
                # pre-SimulationContext timing.
                from isaaclab_physx.renderers import IsaacRtxRendererGlobalSettingsCfg
                from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

                apply_isaac_rtx_global_settings(IsaacRtxRendererGlobalSettingsCfg(carb_settings=_render_settings))
            for attr, val in patches.items():
                setattr(env_cfg.scene, attr, val)
            return env_cfg

        # Arena 3.0 removed ``Scene.modify_env_cfg``; the live extension point is
        # ``scene._manaenv_env_cfg_modifier`` (bridged to ``env_cfg_callback`` by
        # AssemblyFactory, and chained via ``_orig`` above). Writing the old name
        # left sim_fps/render_decremental/simulation_decimation silently ignored.
        scene._manaenv_env_cfg_modifier = _patched
        scene._post_spawn_hooks = self._collect_post_spawn_hooks(components, all_assets)

        manaprint(f"INFO: Scene '{self.config.scene_type}' built " f"with {len(all_assets)} assets")
        return scene

    def _instantiate_components(self, plan: SceneGenerationCompilation) -> list[SceneComponent]:
        """Instantiate scene components from compiled CFGs."""
        if plan.component_cfgs:
            components: list[SceneComponent] = []
            for cfg in plan.component_cfgs:
                component_class = getattr(cfg, "class_type", None)
                if component_class is None or not issubclass(component_class, SceneComponent):
                    raise TypeError(f"Invalid scene component class_type on {type(cfg).__name__}: {component_class!r}")
                components.append(component_class(cfg))
            return components
        return list(plan.components or [])

    def _generate_component(self, component: Any, plan: SceneGenerationCompilation) -> Any:
        """Generate one scene component."""
        if isinstance(component, SceneComponent):
            return component.generate_for_variant(getattr(component, "cfg", None), variant_index=plan.variant_index)
        return component.generate(getattr(component, "cfg", None))

    def _variant_seed(self, variant_index: int) -> int:
        from manaenv.pipeline.generators.scene_layout_solver.common.reproducibility import stable_seed
        from manaenv.pipeline.runtime import master_seed as master_seed_runtime

        override = self.config.seed
        if override is not None:
            return int(stable_seed(int(override), master_seed_runtime.NS_SCENE_VARIANT, int(variant_index)))
        derived = master_seed_runtime.derive(master_seed_runtime.NS_SCENE_VARIANT, int(variant_index))
        if derived is not None:
            return int(derived)
        return int(master_seed_runtime.os_entropy_seed())

    # BEGIN scene_layout_solver hook
    def _apply_scene_layout_solver(self, all_assets):
        layout_payload = (getattr(self.config, "metadata", {}) or {}).get("layout")
        if not isinstance(layout_payload, dict) or not layout_payload.get("enabled"):
            return all_assets
        from manaenv.pipeline.generators.scene_layout_solver.integration import solve_assets_for_scene_generator

        result = solve_assets_for_scene_generator(self, all_assets, layout_payload)
        if not result.success:
            details = [diag.to_json() for diag in result.diagnostics]
            raise RuntimeError(f"scene_layout_solver_failed: {details}")
        return all_assets

    # END scene_layout_solver hook

    @staticmethod
    def _append_component_assets(all_assets: list[Any], component: Any, assets: list[Any]) -> int:
        """Append named asset instances returned by one component."""
        appended = 0
        for asset in assets:
            if asset is None or not hasattr(asset, "name"):
                continue
            asset._component_type = type(component).__name__
            all_assets.append(asset)
            appended += 1
        return appended

    @staticmethod
    def _rename_asset(asset: Any, new_name: str) -> None:
        """Rename *asset* in-place so that ``asset.name``, ``prim_path`` and
        ``object_cfg`` all reflect *new_name*.

        ``LibraryObject`` subclasses define ``name`` as a **class attribute**
        which shadows the ``Asset.name`` property.  A simple
        ``asset._name = new_name`` would only set the instance's backing
        field while ``asset.name`` still returns the class attribute.
        We therefore write into the instance ``__dict__`` directly so that
        the instance value takes precedence over the class attribute.
        """
        # SHPY TODO: need fix in arena registry
        asset.__dict__["name"] = new_name
        asset._name = new_name

        if hasattr(asset, "prim_path"):
            from isaaclab_arena.assets.object_base import asset_anchor_path

            asset.prim_path = asset_anchor_path(new_name)
        if hasattr(asset, "_init_object_cfg"):
            asset.object_cfg = asset._init_object_cfg()
            from manaenv.pipeline.generators.scene.components.objects_component import reapply_kinematic_override

            reapply_kinematic_override(asset)

    @staticmethod
    def _deduplicate_asset_names(assets: list[Any]) -> list[Any]:
        """Rename duplicate asset names by appending ``_0``, ``_1``, etc.

        Assets that appear only once keep their original name unchanged.
        """
        original_names = [getattr(a, "name", None) for a in assets]
        name_counts: dict[str, int] = Counter(original_names)
        seen: dict[str, int] = {}
        for asset, orig_name in zip(assets, original_names):
            if orig_name is None:
                continue
            if name_counts[orig_name] > 1:
                idx = seen.get(orig_name, 0)
                new_name = f"{orig_name}_{idx}"
                manaprint(f"INFO:   Renaming duplicate asset '{orig_name}' -> '{new_name}'")
                SceneGenerator._rename_asset(asset, new_name)
                seen[orig_name] = idx + 1
        return assets

    @staticmethod
    def _collect_post_spawn_hooks(components: list[SceneComponent], assets: list[Any]) -> list[Any]:
        """Collect component post-spawn callbacks for AssemblyFactory."""
        asset_names = {
            getattr(asset, "name", ""): getattr(asset, "_component_type", "")
            for asset in assets
            if getattr(asset, "name", "")
        }
        hooks: list[Any] = []
        base_hook = SceneComponent.post_spawn
        for component in components:
            hook = getattr(component, "post_spawn", None)
            if hook is None or getattr(hook, "__func__", None) is base_hook:
                continue

            def _call(stage, num_envs, _hook=hook, _asset_names=asset_names):
                return _hook(stage, num_envs, _asset_names)

            hooks.append(_call)
        return hooks
