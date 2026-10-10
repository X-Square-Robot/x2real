from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class DomainRandomizationComponent(TaskComponent):
    """Builds domain-randomization as ``EventTermCfg`` entries.

    Domain randomization in arena is realized through event terms.
    Builder functions return ``EventTermCfg`` objects; the component
    wraps them into a dynamic configclass that ``combine_configclass_instances``
    can merge with other event sources.

    YAML example::

        domain_randomization:
          terms:
            uniform_offset:
              asset_cfg: "@pickable"
              range_x: [-0.05, 0.05]
              range_y: [-0.05, 0.05]
    """

    def generate(self, cfg: DomainRandomizationComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        term_fields = []
        for func_name, spec in cfg.terms.items():
            params = dict(spec or {})
            if roles:
                params = self.resolve_roles(params, roles)
            func = resolve_task_term("domain_randomization", func_name)
            term_cfg = func(**params)
            field_name = func_name.replace(".", "__")
            term_fields.append((field_name, type(term_cfg), term_cfg))

        if not term_fields:
            return {"domain_randomization_cfg": None}

        DRCfgCls = make_configclass("DomainRandomizationCfg", term_fields)
        return {"domain_randomization_cfg": DRCfgCls()}


@dataclass
class DomainRandomizationComponentCFG(TaskComponentCFG):

    class_type: type = DomainRandomizationComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
