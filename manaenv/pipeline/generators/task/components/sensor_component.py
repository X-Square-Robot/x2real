from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.component import ROLES_UPDATE_KEY
from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG


class SensorComponent(TaskComponent):
    """Declarative runtime-sensor section: one block, one declaration per sensor.

    Each ``sensors`` entry is a single-key mapping ``{sensor_name: {...}}`` where
    the key is the sensor name (also published as a role so termination can
    reference it via ``@<sensor_name>``):

    * ``type`` — sensor type, currently only ``contact``.
    * ``source`` — asset(s) to mount the sensor on (``@role``; group -> N sensors).
    * ``against`` — asset(s) / prim paths to detect contact with; asset names are
      wrapped into ``{ENV_REGEX_NS}/<name>``, raw prim paths (``{ENV_REGEX_NS}/``
      or ``/`` prefixed, e.g. robot links) pass through.

    The block produces, in one pass, both the dynamic ``SceneCfg`` (the actual
    ``ContactSensorCfg`` objects) and the ``ROLES_UPDATE_KEY`` back-injection of
    sensor names — replacing the legacy ``scene_constraint_component`` double
    declaration (``sensors`` block for names + ``terms`` block for cfgs).

    YAML example (under ``component_sections``)::

        sensor_component:
          sensors:
            - block1_on_paper:
                type: contact
                source: "@sortable_1"
                against: "@destination_1"
            - cup_contact:
                type: contact
                source: "@cups"          # group -> N sensors
                against: "@cups"

    Building reuses :func:`manaenv.terms.scene.build_contact_sensors`, so the
    finalize-time expansions (``finalize_multi_asset_contact_sensors`` /
    ``finalize_pool_cache_contact_sensors`` /
    ``expand_articulation_contact_paths``) keep working unchanged.
    """

    def generate(self, cfg: SensorComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        from manaenv.terms.scene import build_contact_sensors

        roles = roles or {}
        if not cfg.sensors:
            return {}

        scene_fields, roles_update = build_contact_sensors(cfg.sensors, roles)

        result: dict[str, Any] = {}
        if scene_fields:
            from isaaclab_arena.utils.configclass import make_configclass

            SceneCfgCls = make_configclass("SceneCfg", scene_fields)
            result["scene_cfg"] = SceneCfgCls()
        if roles_update:
            result[ROLES_UPDATE_KEY] = roles_update
        return result


@dataclass
class SensorComponentCFG(TaskComponentCFG):

    class_type: type = SensorComponent
    priority: int = 1
    sensors: list[dict[str, Any]] = field(default_factory=list)
