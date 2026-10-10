from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class CurriculumComponent(TaskComponent):
    """Builds curriculum configuration as a dynamic configclass of ``CurriculumTermCfg`` fields.

    YAML example::

        curriculum:
          terms:
            modify_reward_weight:
              term_name: "reach_target"
              weight: 2.0
              num_steps: 10000
    """

    def generate(self, cfg: CurriculumComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        term_fields = []
        for func_name, spec in cfg.terms.items():
            params = dict(spec or {})
            if roles:
                params = self.resolve_roles(params, roles)
            func = resolve_task_term("curriculum", func_name)
            term_cfg = func(**params)
            field_name = func_name.replace(".", "__")
            term_fields.append((field_name, type(term_cfg), term_cfg))

        if not term_fields:
            return {"curriculum_cfg": None}

        CurriculumCfgCls = make_configclass("CurriculumCfg", term_fields)
        return {"curriculum_cfg": CurriculumCfgCls()}


@dataclass
class CurriculumComponentCFG(TaskComponentCFG):

    class_type: type = CurriculumComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
