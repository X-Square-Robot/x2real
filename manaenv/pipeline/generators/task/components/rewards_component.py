from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class RewardsComponent(TaskComponent):
    """Builds reward configuration as a dynamic configclass of ``RewardTermCfg`` fields.

    YAML example::

        reward:
          terms:
            reach_target:
              weight: 1.0
              target_cfg: "@pickable"
    """

    def generate(self, cfg: RewardsComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from isaaclab_arena.utils.configclass import make_configclass

        roles = roles or {}
        term_fields = []
        for func_name, spec in cfg.terms.items():
            params = dict(spec or {})
            if roles:
                params = self.resolve_roles(params, roles)
            func = resolve_task_term("reward", func_name)
            term_cfg = func(**params)
            field_name = func_name.replace(".", "__")
            term_fields.append((field_name, type(term_cfg), term_cfg))

        if not term_fields:
            return {"rewards_cfg": None}

        RewardsCfgCls = make_configclass("RewardsCfg", term_fields)
        return {"rewards_cfg": RewardsCfgCls()}


@dataclass
class RewardsComponentCFG(TaskComponentCFG):

    class_type: type = RewardsComponent
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)
