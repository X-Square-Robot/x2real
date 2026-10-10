"""Task generator base interfaces.

The Mana task pipeline runs three Generator-owned stages:

* ``compile(definition)`` produces a ``TaskGenerationCompilation`` subclass.
  The base task path uses ``TaskGenerationCompilation`` directly; specialised
  task generators may extend it with their own compile-stage fields.
* ``instancing(compiled)`` turns the compilation into either a runtime
  ``ManaTask`` or a task object that still needs scene/robot finalisation.
* ``finalize(task, scene, robot)`` performs late binding with the Arena
  ``Scene`` and ``EmbodimentBase`` once assembly has selected them, then
  returns the final ``ManaTask`` instance. ``TaskFactory.finalize_task``
  calls this hook on the originating generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from manaenv.pipeline.component import extract_roles_update
from manaenv.pipeline.generator import GenerationCompilation, ManaGenerator
from manaenv.pipeline.generators.task.ref_text import collect_text_refs, render_ref_text
from manaenv.pipeline.generators.task.role_splitter import split_component_sections_by_role_usage
from manaenv.pipeline.translator.yaml_reflection import reflect_yaml_dict

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import ManaTask


@dataclass
class TaskGenerationCompilation(GenerationCompilation):
    """Task-generation artifact split by scene-role dependency."""

    definition: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)
    instance_sections: dict[str, Any] = field(default_factory=dict)
    finalize_sections: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskGeneratorCFG:
    """Base configuration for task generators.

    Two slot mechanisms are layered on top of this base:

    * **Template payload slots** — ``ConstraintTaskGeneratorCFG`` stores
      template-specific values in ``template_payload`` and passes them to
      the resolved ``*TaskCFG`` class.
    * **YAML-injection slots** — arbitrary ``roles: { name: tag }``
      entries from yaml flow through to the runtime task. Layer-C
      components (e.g. ``SceneConstraintComponent``) may also publish
      dynamic roles via ``_roles_update`` at assembly time.

    Task-level globals (``task_type`` / ``episode_length_s`` /
    ``prompt`` / ``num_episodes`` / ``max_steps_per_episode`` /
    ``robot_constraints`` / ``benchmark``) live in :attr:`definition`;
    the runtime :class:`isaaclab_arena.tasks.task_base.ManaTask` reads
    them from there.
    """

    class_type: type[TaskGenerator] | None = None
    gen_num: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    definition: dict[str, Any] = field(default_factory=dict)
    roles: dict[str, Any] = field(default_factory=dict)
    component_sections: dict[str, Any] = field(default_factory=dict)


class TaskGenerator(ManaGenerator):
    """Base interface for task-generation workflows.

    The base implementation is the component-section compiler used by
    DIRECT and reusable by higher-level workflows. Subclasses may
    override any stage when they produce template-specific CFG artifacts.
    """

    def __init__(self, cfg: TaskGeneratorCFG, **kwargs):
        super().__init__(cfg)
        self._results: list[ManaTask] = []
        self.roles: dict[str, Any] = dict(cfg.roles) if cfg.roles else {}
        self._deferred_component_sections: dict[int, dict[str, Any]] = {}

    def generate(self, definition: dict[str, Any] | None = None, **kwargs: object) -> ManaTask:
        """Run compile + instancing; ``finalize`` is deferred to ``TaskFactory.finalize_task``."""
        definition = definition or self.cfg.definition or {}
        compiled_cfg = self.compile(definition=definition, **kwargs)
        return self.instancing(compiled_cfg)

    def compile(self, definition: dict[str, Any] | None = None, **kwargs: object) -> TaskGenerationCompilation:
        """Split component sections into instance-time and finalize-time payloads."""
        del kwargs

        resolved_definition = dict(definition or self.cfg.definition or {})
        split = split_component_sections_by_role_usage(self.cfg.component_sections)
        roles = dict(self.cfg.roles or {})
        for role_name in split.referenced_roles:
            roles.setdefault(role_name, role_name)
        for role_name in collect_text_refs(resolved_definition.get("prompt")):
            roles.setdefault(role_name, role_name)
        self.roles = dict(roles)
        return TaskGenerationCompilation(
            definition=resolved_definition,
            roles=roles,
            instance_sections=split.instance_sections,
            finalize_sections=split.finalize_sections,
        )

    def instancing(
        self,
        compiled_cfg: TaskGenerationCompilation,
        task: ManaTask | None = None,
    ) -> ManaTask:
        """Compose instance-time component sections onto a task."""
        self.components = []

        self.roles = dict(compiled_cfg.roles)
        if task is None:
            task = self._wrap_into_mana_task({}, dict(self.roles), compiled_cfg.definition)
        self._deferred_component_sections[id(task)] = dict(compiled_cfg.finalize_sections)
        return self._append_component_sections_to_task(
            task,
            compiled_cfg.instance_sections,
            roles=compiled_cfg.roles,
        )

    def finalize(self, task: ManaTask, scene: Scene, robot: EmbodimentBase, **kwargs: object) -> ManaTask:
        """Late-bind scene/robot context and return the final runtime task."""
        del kwargs
        return self._finalize_task_runtime(task, scene=scene, robot=robot)

    def _append_component_sections_to_task(
        self,
        task: ManaTask,
        sections: dict[str, Any],
        *,
        roles: dict[str, Any] | None = None,
    ) -> ManaTask:
        """Compile *sections* into components and append their output onto *task*."""
        self.components = []

        if roles:
            self.roles.update(roles)

        component_cfgs = self._compile_component_cfgs_from_sections(sections or {})
        self._inject_component_cfgs(component_cfgs)

        existing_runtime_cfgs: dict[str, Any] = task.runtime_cfgs if isinstance(task.runtime_cfgs, dict) else {}
        if not isinstance(task.runtime_cfgs, dict):
            task.runtime_cfgs = existing_runtime_cfgs

        existing_roles: dict[str, Any] = task.roles if isinstance(task.roles, dict) else {}
        if not isinstance(task.roles, dict):
            task.roles = existing_roles

        for component in self.sorted_components():
            try:
                result = component.generate(component.cfg, roles=self.roles)
            except TypeError:
                result = component.generate(component.cfg)
            update = extract_roles_update(result)
            if update:
                self.roles.update(update)
            # ``log_producers`` is the one additive runtime-cfg channel: multiple
            # components (e.g. a multi-variant instruction section + a subtask
            # log_component) each contribute specs, so extend rather than let a
            # later component's list clobber an earlier one's.
            incoming_log_producers = result.pop("log_producers", None)
            existing_runtime_cfgs.update(result)
            if incoming_log_producers:
                merged = list(existing_runtime_cfgs.get("log_producers") or [])
                merged.extend(incoming_log_producers)
                existing_runtime_cfgs["log_producers"] = merged

        existing_roles.update(self.roles)
        return task

    def _finalize_task_runtime(self, task: ManaTask, scene: Scene, robot: EmbodimentBase) -> ManaTask:
        """Apply scene/robot-dependent post-processing to an existing ``ManaTask``."""
        from isaaclab_arena.tasks.task_base import ManaTask

        from manaenv.pipeline.factories.task_factory import TaskFactory as _TaskFactory
        from manaenv.terms.scene import (
            bind_ee_contact_sensor_object_filters,
            bind_robot_ee_contact_sensor_filters,
            expand_articulation_contact_paths,
            finalize_multi_asset_contact_sensors,
            finalize_pool_cache_contact_sensors,
        )

        if not isinstance(task, ManaTask):
            raise TypeError(f"{type(self).__name__}.finalize expected a ManaTask, got {type(task).__name__}")

        if task.roles:
            old_roles = dict(task.roles)
            task.roles = self._resolve_roles_from_scene(task.roles, scene)
            tag_to_name = {
                old_val: task.roles[k]
                for k, old_val in old_roles.items()
                if isinstance(old_val, str) and old_val != task.roles.get(k)
            }
        else:
            tag_to_name = {}

        prompt = getattr(task.cfg, "prompt", None)
        if isinstance(prompt, str) and "@" in prompt and task.roles:

            task.cfg.prompt = render_ref_text(prompt, task.roles)

        deferred_sections = self._deferred_component_sections.pop(id(task), {}) or {}
        if deferred_sections:
            self._append_component_sections_to_task(task, deferred_sections, roles=task.roles)

        needs_term_expand = False
        if tag_to_name and task.runtime_cfgs:
            finalize_multi_asset_contact_sensors(task.runtime_cfgs, task.roles, tag_to_name)
            _TaskFactory._patch_runtime_cfgs(task.runtime_cfgs, tag_to_name)
            needs_term_expand = True

        runtime_cfgs = task.runtime_cfgs if isinstance(getattr(task, "runtime_cfgs", None), dict) else {}
        if runtime_cfgs:
            # Bind generic gripper_contact_sensor filters to this embodiment's EE
            # links (EX001 vs ARX) before pool-slot expansion copies them.
            bind_robot_ee_contact_sensor_filters(runtime_cfgs, task.roles, robot)
            # Pool cache slots are independent of semantic-tag multi-asset expand:
            # mount one ContactSensorCfg per slot so object_contact_destination can
            # OR forces across spares (active slot only can touch the destination).
            if finalize_pool_cache_contact_sensors(runtime_cfgs, task.roles, scene):
                needs_term_expand = True
            if needs_term_expand:
                _TaskFactory._expand_runtime_termination_terms(runtime_cfgs)
            expand_articulation_contact_paths(runtime_cfgs, scene)

        # EE object filters patch ``robot.scene_config`` and must run even when the
        # task has no scene_cfg / empty runtime_cfgs (nut_on_screw pick stages).
        bind_ee_contact_sensor_object_filters(runtime_cfgs, task.roles or {}, robot, scene)

        _TaskFactory._materialize_task_default_visualizer_cfg(task, scene)

        # scene_layout_solver task runtime compile bridge
        from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.task_runtime import (
            compile_layout_runtime_events_for_task,
        )

        compile_layout_runtime_events_for_task(task, scene)
        return task

    @staticmethod
    def _resolve_roles_from_scene(roles: dict[str, Any], scene: Any) -> dict[str, Any]:
        """Resolve component role bindings through the shared scene binder."""
        from manaenv.utils.affordance_binder import resolve

        resolved: dict[str, Any] = {}
        for role_name, value in roles.items():
            if isinstance(value, (str, list)):
                try:
                    resolved_value = resolve(value, scene, target_type=None, field_name=role_name, expect_list=True)
                except (KeyError, TypeError, ValueError):
                    resolved_value = value
                if isinstance(value, str) and isinstance(resolved_value, list) and len(resolved_value) == 1:
                    resolved_value = resolved_value[0]
                resolved[role_name] = resolved_value
            else:
                resolved[role_name] = value
        return resolved

    def _compile_component_cfgs_from_sections(self, sections: dict[str, Any]) -> list[Any]:
        """Parse YAML sections into task component CFG instances."""
        from manaenv.pipeline.generators.task.component import TaskComponentCFG

        component_cfgs: list[TaskComponentCFG] = []
        for key, section_data in sections.items():
            if isinstance(section_data, TaskComponentCFG):
                component_cfgs.append(section_data)
                continue
            if isinstance(section_data, list) and all(isinstance(item, TaskComponentCFG) for item in section_data):
                component_cfgs.extend(section_data)
                continue
            reflected = reflect_yaml_dict(
                {key: section_data},
                context="task.component_sections",
                ignore_unknown=True,
            )
            cfg_instance = reflected.get(key)
            if cfg_instance is None or not isinstance(cfg_instance, TaskComponentCFG):
                continue
            component_cfgs.append(cfg_instance)
        return component_cfgs

    def _inject_component_cfgs(self, component_cfgs: list[Any]) -> None:
        """Inject already-instantiated task component CFGs."""
        from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG

        for cfg_instance in component_cfgs:
            if not isinstance(cfg_instance, TaskComponentCFG):
                raise TypeError(f"Expected TaskComponentCFG, got {type(cfg_instance)!r}")
            impl_class = cfg_instance.class_type
            if impl_class is None or not issubclass(impl_class, TaskComponent):
                raise TypeError(f"Invalid task component class_type on {type(cfg_instance).__name__}: {impl_class!r}")
            self.add_component(impl_class(cfg_instance))

    def _wrap_into_mana_task(
        self,
        runtime_cfgs: dict[str, Any],
        roles: dict[str, Any],
        definition: dict[str, Any],
    ) -> ManaTask:
        """Pack *runtime_cfgs* into a graph-carrying ``ManaTask`` and append to ``self._results``.

        The product is a FORGED ManaTask subclass, never the bare Arena class: every task
        is a graph task, and one that declares no stages carries an EMPTY graph. Its getters
        behave exactly as a leaf's always have — the mixin short-circuits on the empty graph
        — but the TYPE is the same one a composed graph produces, so no consumer downstream
        (the stage machine, the failure attribution, the collection loops) has to ask which
        kind of task it holds. A graph's own child nodes are built through this same path,
        and their empty graphs make them inert: the composing parent supplies the one
        machine, so no nesting and no ``dag_*`` extras collision.
        """
        from isaaclab_arena.tasks.task_base import ManaTask, TaskCFG

        from manaenv.pipeline.generators.task.graph_task_forge import forge_graph_task_type
        from manaenv.pipeline.generators.task.recursive_composite import RecursiveCompositeTaskMixin
        from manaenv.pipeline.graph.mana_graph import ManaGraph

        meta_source = definition if isinstance(definition, dict) else {}
        task_cfg = TaskCFG(
            episode_length_s=meta_source.get("episode_length_s"),
            prompt=meta_source.get("prompt", ""),
            runtime_cfgs=runtime_cfgs,
            roles=dict(roles),
            definition=definition or {},
        )
        forged_cls = forge_graph_task_type(
            graph_name=str(meta_source.get("task_type") or "task"),
            base_cls=ManaTask,
            node_types=(),
            mixins=(RecursiveCompositeTaskMixin,),
        )
        task = forged_cls(cfg=task_cfg)
        # An EMPTY graph, not a missing one: `graph is None` is what let consumers keep
        # asking "is this a graph task?".
        task.graph = ManaGraph()
        task._composite_children = []
        task._composite_edges = []
        self._results.append(task)
        return task

    def get_results(self) -> list[ManaTask]:
        return self._results

    def clear_results(self):
        self._results = []
