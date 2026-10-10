"""CONSTRAINT task generator workflow.

The CONSTRAINT generator is the public, scene-aware entry point for
template-driven tasks. It owns one path only: ``cfg.template`` is the
**exact** ``*Task`` class name (e.g. ``"PickAndPlaceTask"`` /
``"OpenDoorTask"``) and is resolved through the global lazy registry
(see :mod:`template_discovery`) into the peer ``*TaskCFG`` class via
the standard CFG-Instance convention. Template-specific YAML values
live in ``cfg.template_payload`` and are interpreted by the resolved
template ``*TaskCFG``.

There is no shorthand / snake_case / suffix-rewrite for ``template``:
the value must equal the registered ``*Task`` class name byte-for-byte
so this generator stays consistent with the rest of the CFG-Instance
machinery.

For component-section / event-term workflows that don't need template
resolution, use :class:`DirectTaskGenerator` directly.

Specialised task variants are represented by template sections in YAML
and resolved to concrete ``*TaskCFG`` classes at compile time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, cast

from manaenv.pipeline.generators.task_generator import TaskGenerationCompilation, TaskGenerator, TaskGeneratorCFG
from manaenv.pipeline.registry.global_provider import global_registry
from manaenv.utils.affordance_binder import bind_typed_slots

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import ManaTask, TaskCFG  # noqa: F401


def _is_template_task(obj: object) -> bool:
    """Return ``True`` when *obj* is an arena ``ManaTask`` subclass instance."""
    try:
        from isaaclab_arena.tasks.task_base import ManaTask
    except ImportError:
        return False
    return isinstance(obj, ManaTask)


@dataclass
class ConstraintTaskGenerationCompilation(TaskGenerationCompilation):
    """CONSTRAINT task compilation with the template cfg for base instancing."""

    template_task_cfg: object | None = None


class ConstraintTaskGenerator(TaskGenerator):
    """Template-driven, scene-aware task generator.

    Lifecycle:

    * ``compile`` resolves ``cfg.template`` and stores the populated
      ``*TaskCFG`` inside ``ConstraintTaskGenerationCompilation``.
    * ``instancing`` constructs the matching ``*Task`` from the
      compilation's template cfg, then applies instance components via
      the parent task component flow.
    * ``finalize`` accepts that ``*Task`` only, requires a ``scene``,
      calls ``bind_typed_slots`` to resolve its unfinished cfg asset
      references, then delegates runtime cfg construction to
      ``task.finalize(scene, robot)``.
    """

    def compile(
        self,
        definition: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> ConstraintTaskGenerationCompilation:
        del kwargs
        cfg: ConstraintTaskGeneratorCFG = self.cfg
        self.roles = dict(cfg.roles) if cfg.roles else {}
        if not cfg.template:
            raise ValueError(
                f"{type(self).__name__} requires a non-empty `template` field; "
                "use DirectTaskGenerator for component-section workflows."
            )
        compiled = super().compile(definition=definition)
        return ConstraintTaskGenerationCompilation(
            definition=compiled.definition,
            roles=compiled.roles,
            instance_sections=compiled.instance_sections,
            finalize_sections=compiled.finalize_sections,
            template_task_cfg=self._compile_template(cfg, definition),
        )

    def instancing(
        self,
        compiled_cfg: ConstraintTaskGenerationCompilation,
    ) -> ManaTask:
        template_task_cfg = cast("TaskCFG", compiled_cfg.template_task_cfg)
        task_cls = template_task_cfg.class_type
        if task_cls is None:
            raise TypeError(
                f"Template task cfg {type(template_task_cfg).__name__} has no class_type; "
                "ensure the *TaskCFG dataclass declares its peer class_type."
            )
        task = cast("ManaTask", task_cls(replace(template_task_cfg)))
        self._results.append(task)
        return super().instancing(compiled_cfg, task=task)

    def finalize(
        self,
        task: ManaTask,
        scene: Scene,
        robot: EmbodimentBase,
        **kwargs: object,
    ) -> ManaTask:
        del kwargs
        if not _is_template_task(task):
            raise TypeError(f"{type(self).__name__}.finalize expected a template *Task, got {type(task).__name__}")
        return self._finalize_template_task(task, scene=scene, robot=robot)

    def _compile_template(
        self,
        cfg: ConstraintTaskGeneratorCFG,
        definition: dict[str, Any] | None,
    ) -> TaskCFG:
        spec = definition or cfg.definition or {}
        task_cfg_class = self._resolve_template_cfg_class(cfg.template)
        return cast("TaskCFG", cfg.build_template_task_cfg(task_cfg_class, spec))

    def _finalize_template_task(
        self,
        task: ManaTask,
        *,
        scene: Scene,
        robot: EmbodimentBase,
    ) -> ManaTask:
        if scene is None:
            raise ValueError(
                f"{type(self).__name__}.finalize requires a scene to resolve template task "
                f"asset references for {type(task).__name__}."
            )
        bind_typed_slots(task.cfg, scene)
        if not hasattr(task, "finalize"):
            finalized_task = task
        else:
            finalized_task = cast("ManaTask", task.finalize(scene=scene, robot=robot))
        component_sections = getattr(getattr(finalized_task, "cfg", None), "component_sections", {}) or {}
        if component_sections:
            self._append_component_sections_to_task(finalized_task, component_sections, roles=finalized_task.roles)
        return self._finalize_task_runtime(finalized_task, scene=scene, robot=robot)

    @staticmethod
    def _resolve_template_cfg_class(template: str) -> type:
        if not isinstance(template, str) or not template:
            raise ValueError(f"template must be a non-empty string, got {template!r}")
        cfg_class = global_registry.get_cfg(template)
        if cfg_class is None:
            raise ValueError(
                f"ConstraintTaskGenerator could not resolve template {template!r}; "
                "template must match an existing *Task class name exactly (e.g. "
                "'PickAndPlaceTask', 'OpenDoorTask') and the *Task / *TaskCFG pair "
                "must live under manaenv.tasks.* or isaaclab_arena.tasks.* so the "
                "task_template lazy provider can find it."
            )
        return cfg_class


@dataclass
class ConstraintTaskGeneratorCFG(TaskGeneratorCFG):
    """CFG for the unified CONSTRAINT task workflow.

    ``template`` is the exact ``*Task`` class name (no shorthand, no
    ``Task`` auto-suffix). It is resolved through the lazy task
    template registry into the peer ``*TaskCFG`` via the CFG-Instance
    convention. Template-specific fields live in :attr:`template_payload`
    and are consumed by the resolved ``*TaskCFG``.
    """

    class_type: type[TaskGenerator] | None = ConstraintTaskGenerator
    template: str = ""
    template_payload: dict[str, Any] = field(default_factory=dict)

    def build_template_task_cfg(self, task_cfg_class: type, spec: dict[str, Any]) -> object:
        cfg_field_names = {f.name for f in fields(task_cfg_class)}
        kwargs: dict[str, Any] = {}
        if "episode_length_s" in cfg_field_names:
            kwargs["episode_length_s"] = spec.get("episode_length_s")
        if "prompt" in cfg_field_names:
            kwargs["prompt"] = spec.get("prompt", "")
        if "runtime_cfgs" in cfg_field_names:
            kwargs["runtime_cfgs"] = {}
        if "roles" in cfg_field_names:
            kwargs["roles"] = dict(self.roles or {})
        if "definition" in cfg_field_names:
            kwargs["definition"] = dict(spec)
        if "component_sections" in cfg_field_names:
            kwargs["component_sections"] = dict(self.component_sections or {})

        kwargs.update(self.template_payload)

        return task_cfg_class(**kwargs)
