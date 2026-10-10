from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class ObservationComponent(TaskComponent):
    """Builds observation configuration from grouped terms.

    YAML structure mirrors IsaacLab's two-level observation layout:
    groups (e.g. ``policy``) contain named ``ObsTerm`` entries.

    Each leaf term is resolved via ``resolve_func`` and must return an
    ``ObservationTermCfg``.  Groups are wrapped into
    ``ObservationGroupCfg`` subclasses, then the whole thing becomes a
    dynamic configclass for ``combine_configclass_instances``.

    YAML example::

        observation:
          terms:
            policy:
              concatenate_terms: false
              terms:
                joint_pos:
                  asset_cfg: "@robot"
                joint_vel:
                  asset_cfg: "@robot"
                last_action: {}
    """

    def generate(self, cfg: ObservationComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        group_fields = []

        for group_name, group_spec in cfg.terms.items():
            group_spec = group_spec or {}
            inner_terms = group_spec.get("terms", {})
            concatenate = group_spec.get("concatenate_terms", False)
            enable_corruption = group_spec.get("enable_corruption", False)

            obs_fields = []
            for func_name, spec in inner_terms.items():
                params = dict(spec or {})
                if roles:
                    params = self.resolve_roles(params, roles)
                func = resolve_task_term("observation", func_name)
                obs_term = func(**params)
                obs_fields.append((func_name, type(obs_term), obs_term))

            GroupCls = make_configclass(f"{group_name}_ObsGroup", obs_fields, bases=(ObsGroup,))
            group_instance = GroupCls()
            group_instance.concatenate_terms = concatenate
            group_instance.enable_corruption = enable_corruption
            group_fields.append((group_name, type(group_instance), group_instance))

        ObsCfgCls = make_configclass("ObservationCfg", group_fields)
        return {"observation_cfg": ObsCfgCls()}


@dataclass
class ObservationComponentCFG(TaskComponentCFG):

    class_type: type = ObservationComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
