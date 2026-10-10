from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class EventComponent(TaskComponent):
    """Builds ``EventTermCfg`` objects and wraps them into a dynamic
    configclass for ``combine_configclass_instances``.

    YAML example::

        event:
          terms:
            mdp.reset_scene_to_default:
              mode: "reset"
              reset_joint_targets: true
            randomize_object_pose_from_pose_range:
              mode: "reset"
              asset_cfg: "pick_object"
              initial_pose: null
              pose_range:
                x: [-0.05, 0.05]
                y: [-0.05, 0.05]
            randomize_pickable_xy_align_reference:
              mode: "reset"
              asset_cfg: "@pickable"
              reference_cfg: "@source_container"
              pose_range:
                x: [0.0, 0.1]
                y: [-0.7, 0.15]
              z_above_reference: [0.055, 0.09]
              yaw_range: [-0.5, 0.5]

    ``mode`` is reserved (defaults to ``"reset"``).
    Everything else is forwarded as plain params.
    """

    def generate(self, cfg: EventComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab.managers import EventTermCfg
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        term_fields = []
        for func_name, spec in cfg.terms.items():
            spec = spec or {}
            target_func_name = spec.get("_func", func_name)
            # EventTermCfg-level scheduling fields are consumed by IsaacLab's
            # EventManager, NOT passed to the func — keep them OFF ``params`` so
            # manager validation doesn't reject them as phantom func arguments.
            # ``interval_range_s`` / ``is_global_time`` drive ``mode: "interval"``.
            reserved = ("mode", "_func", "interval_range_s", "is_global_time")
            params = {k: v for k, v in spec.items() if k not in reserved}
            if roles:
                params = self.resolve_roles(params, roles)
            term_kwargs: dict[str, Any] = {
                "func": resolve_task_term("event", target_func_name),
                "mode": spec.get("mode", "reset"),
                "params": params,
            }
            if "interval_range_s" in spec:
                term_kwargs["interval_range_s"] = tuple(spec["interval_range_s"])
            if "is_global_time" in spec:
                term_kwargs["is_global_time"] = bool(spec["is_global_time"])
            term_cfg = EventTermCfg(**term_kwargs)
            field_name = func_name.replace(".", "__")
            term_fields.append((field_name, type(term_cfg), term_cfg))

        EventsCfgCls = make_configclass("EventsCfg", term_fields)
        return {"event_terms": EventsCfgCls()}


@dataclass
class EventComponentCFG(TaskComponentCFG):

    class_type: type = EventComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
