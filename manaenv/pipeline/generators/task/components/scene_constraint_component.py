from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.component import ROLES_UPDATE_KEY
from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.term_discovery import resolve_task_term


class SceneConstraintComponent(TaskComponent):
    """Declares scene constraints, generates sensors, and produces a SceneCfg.

    The component serves three purposes:

    1. **constraints** — forwarded verbatim for Mana's internal scene
       generation logic. This is the component's lasting responsibility.
    2. **sensors** *(deprecated)* — derives sensor role names from a source
       asset role and a suffix, then writes them back via ``_roles_update``.
    3. **terms** *(deprecated)* — each entry is resolved to a builder function
       in ``manaenv.terms.scene`` whose return value (a dict of sensor-name
       → ``ContactSensorCfg``) is collected and wrapped into a dynamic
       ``SceneCfg`` configclass via ``make_configclass``.

    .. deprecated::
        The ``sensors`` / ``terms`` sensor responsibility has moved to the
        dedicated ``sensor_component`` (declarative single block: name +
        ``type`` / ``source`` / ``against``). New tasks should declare contact
        sensors there; ``constraints`` stays here. The legacy ``sensors`` /
        ``terms`` blocks still work (byte-equivalent output) but emit a
        ``DeprecationWarning``.

    YAML example::

        scene_constraint:
          constraints:
            - mode: 0
              tags: "buttons_contact_scene"
    """

    def generate(self, cfg: SceneConstraintComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        roles = roles or {}
        result: dict[str, Any] = {"scene_constraints": list(cfg.constraints)}

        if cfg.sensors or cfg.terms:
            warnings.warn(
                "scene_constraint_component 'sensors'/'terms' blocks are deprecated; "
                "declare contact sensors in the dedicated 'sensor_component' "
                "(single block: name + type/source/against). 'constraints' stays here.",
                DeprecationWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # sensors → derived role names (back-injected via ROLES_UPDATE_KEY)
        # ------------------------------------------------------------------
        roles_update: dict[str, Any] = {}
        for sensor in cfg.sensors:
            source_ref = sensor.get("source", "")
            suffix = sensor.get("suffix", "sensor")
            output_role = sensor.get("output_role", "")

            assets = self.resolve_roles({"_src": source_ref}, roles)["_src"] if roles else source_ref
            assets = _flatten_sensor_assets(assets)

            sensor_names = [f"{getattr(asset, 'name', asset)}_{suffix}" for asset in assets]

            if output_role:
                roles_update[output_role] = sensor_names

        if roles_update:
            result[ROLES_UPDATE_KEY] = roles_update

        # ------------------------------------------------------------------
        # terms → SceneCfg configclass
        # ------------------------------------------------------------------
        if cfg.terms:
            from isaaclab_arena.utils.configclass import make_configclass

            scene_fields: list[tuple[str, type, Any]] = []
            for func_name, spec in cfg.terms.items():
                params = dict(spec or {})
                if roles:
                    params = self.resolve_roles(params, roles)
                func = resolve_task_term("scene_constraint", func_name)
                ret = func(**params)
                if isinstance(ret, dict):
                    for field_name, cfg_obj in ret.items():
                        scene_fields.append((field_name, type(cfg_obj), cfg_obj))
                else:
                    scene_fields.append((func_name, type(ret), ret))

            if scene_fields:
                SceneCfgCls = make_configclass("SceneCfg", scene_fields)
                result["scene_cfg"] = SceneCfgCls()

        return result


@dataclass
class SceneConstraintComponentCFG(TaskComponentCFG):

    class_type: type = SceneConstraintComponent
    priority: int = 1
    constraints: list[dict[str, Any]] = field(default_factory=list)
    sensors: list[dict[str, Any]] = field(default_factory=list)
    terms: dict[str, dict[str, Any]] = field(default_factory=dict)


def _flatten_sensor_assets(value: Any) -> list[Any]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        assets: list[Any] = []
        for item in value:
            assets.extend(_flatten_sensor_assets(item))
        return assets
    return [value]
