from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.tasks.task_base import ManaTask, TaskCFG

from manaenv.runtime_context import get_runtime_context, runtime_template_field
from manaenv.runtime_owners import mana_task
from manaenv.terms.terminations import articulation_joints_near_default_after_time, contact_sequence_reached
from manaenv.terms.viewers import look_at_object

if TYPE_CHECKING:
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.metrics.metric_base import MetricBase

CONTACT_SENSOR_SUFFIX = "contact_sensor"
VIEWER_OFFSET = [-1.0, 0.0, 1.5]
ARTICULATION_HOME_ATOL = 0.3
ARTICULATION_HOME_AFTER_SECONDS = 10.0


@dataclass
class ButtonContactTaskCFG(TaskCFG):
    class_type: type[ButtonContactTask] | None = None
    # Button assets to press; symbolic tags are resolved by the TaskGenerator before finalize().
    target_buttons: list[Asset] | str = "@touchable"
    # Number of buttons from the resolved sequence that must be contacted.
    sequence_length: int = 3
    # Minimum contact force for a button press to count as reached.
    contact_force_threshold: float = 0.1
    # Optional schema-vNext runtime context consumer settings. When set, this
    # task template owns runtime instruction and success order.
    runtime_plan_id: str = ""
    runtime_order_path: str = ""
    runtime_instruction_field: str = "instruction.instruction"
    runtime_buttons_field: str = "task.buttons"
    runtime_instruction_template: str = "Press the buttons in order: {button_names}."

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = ButtonContactTask


class ButtonContactTask(ManaTask):
    """Arena-style template task for sequential button contact."""

    def __init__(self, cfg: ButtonContactTaskCFG):
        super().__init__(cfg=cfg)
        self.cfg: ButtonContactTaskCFG = cfg
        self.target_buttons: list[Asset] = []
        self.sequence_buttons: list[Asset] = []
        self.contact_sensor_names: list[str] = []
        self.contact_against_prim_paths: list[str] = []
        self.scene_config = None
        self.events_cfg = None
        self.termination_cfg = None

    def finalize(self, scene: Any = None, robot: Any = None) -> ButtonContactTask:
        del scene
        self.target_buttons = self._require_asset_list(self.cfg.target_buttons, "target_buttons")
        if not self.target_buttons:
            raise ValueError("ButtonContactTask requires at least one target button after scene binding.")
        self.cfg.target_buttons = self.target_buttons

        self.sequence_buttons = self._select_sequence(self.target_buttons)
        self.contact_sensor_names = [self._contact_sensor_name(button) for button in self.sequence_buttons]
        self.contact_against_prim_paths = self._resolve_contact_against_prim_paths(robot)
        self.scene_config = self.make_scene_cfg()
        self.termination_cfg = self.make_termination_cfg()
        return self

    @staticmethod
    def _require_asset_list(ref: Any, field_name: str) -> list[Asset]:
        from isaaclab_arena.assets.asset import Asset

        values = ref if isinstance(ref, list) else [ref]
        assets = [item for item in values if isinstance(item, Asset)]
        if len(assets) != len(values):
            raise TypeError(
                f"ButtonContactTask requires resolved Asset instances for {field_name}; "
                f"got {type(ref).__name__}: {ref!r}."
            )
        return assets

    @staticmethod
    def _contact_sensor_name(button: Asset) -> str:
        return f"{button.name}_{CONTACT_SENSOR_SUFFIX}"

    @staticmethod
    def _resolve_contact_against_prim_paths(robot: Any) -> list[str]:
        return list(robot.get_ee_contact_prim_paths())

    def _select_sequence(self, buttons: list[Asset]) -> list[Asset]:
        count = max(1, min(int(self.cfg.sequence_length), len(buttons)))
        return list(buttons)[:count]

    def make_scene_cfg(self):
        return make_button_contact_scene_cfg(
            buttons=self.target_buttons,
            sensor_suffix=CONTACT_SENSOR_SUFFIX,
            contact_against_prim_paths=self.contact_against_prim_paths,
        )

    def make_termination_cfg(self):
        return ButtonContactTerminationsCfg()

    def get_scene_cfg(self):
        return self.scene_config

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return self.events_cfg

    def get_prompt(self):
        names = ", ".join(button.name for button in self.sequence_buttons)
        return f"Press the buttons in order: {names}."

    def make_runtime_templates(self) -> dict[str, Any]:
        if not (self.cfg.runtime_plan_id or self.cfg.runtime_order_path):
            return {}
        order_path = str(self.cfg.runtime_order_path or "selections.button_order.objects_by_slot_order")
        fields: dict[str, Any] = {}
        instruction_field = str(self.cfg.runtime_instruction_field or "instruction.instruction")
        if instruction_field:
            instruction_template = str(
                self.cfg.runtime_instruction_template or "Press the buttons in order: {button_names}."
            )
            fields[instruction_field] = {
                "template": self._runtime_instruction_template_rule(instruction_template, order_path),
            }
        buttons_field = str(self.cfg.runtime_buttons_field or "task.buttons")
        if buttons_field:
            fields[buttons_field] = {"from": order_path}
        return {"fields": fields} if fields else {}

    @staticmethod
    def _runtime_instruction_template_rule(template: str, order_path: str) -> str:
        display_path = ButtonContactTask._runtime_display_order_path(order_path)
        return str(template).replace("{button_names}", "{" + display_path + "}")

    @staticmethod
    def _runtime_display_order_path(order_path: str) -> str:
        path = str(order_path).strip()
        replacements = {
            ".objects_by_slot_order": ".display_names_by_slot_order",
            ".object_names": ".display_names",
        }
        for suffix, replacement in replacements.items():
            if path.endswith(suffix):
                return path[: -len(suffix)] + replacement
        return path

    def make_runtime_context_consumer(self):
        if not (self.cfg.runtime_plan_id or self.cfg.runtime_order_path):
            return None
        return ButtonContactRuntimeConsumer(
            plan_id=str(self.cfg.runtime_plan_id or "runtime"),
            order_path=str(self.cfg.runtime_order_path or "selections.button_order.objects_by_slot_order"),
            instruction_field=str(self.cfg.runtime_instruction_field or "instruction.instruction"),
            buttons_field=str(self.cfg.runtime_buttons_field or "task.buttons"),
            instruction_template=str(
                self.cfg.runtime_instruction_template or "Press the buttons in order: {button_names}."
            ),
        )

    def get_metrics(self) -> list[MetricBase]:
        from isaaclab_arena.metrics.success_rate import SuccessRateMetric

        return [SuccessRateMetric()]

    def get_default_visualizer_cfg(self):
        return look_at_object(
            lookat_object=self.sequence_buttons[0],
            offset=VIEWER_OFFSET,
        )

    def get_evaluation_term_cfg(self):
        from isaaclab.managers import SceneEntityCfg

        return make_button_contact_evaluation_cfg(
            contact_sensor_cfgs=[SceneEntityCfg(name) for name in self.contact_sensor_names],
            force_threshold=float(self.cfg.contact_force_threshold),
            articulation_home_atol=ARTICULATION_HOME_ATOL,
            articulation_home_after_seconds=ARTICULATION_HOME_AFTER_SECONDS,
        )


def make_button_contact_scene_cfg(
    buttons: list[Asset],
    sensor_suffix: str,
    contact_against_prim_paths: list[str],
):
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
    from isaaclab_arena.utils.configclass import make_configclass

    fields = []
    for button in buttons:
        sensor_name = f"{button.name}_{sensor_suffix}"
        fields.append(
            (
                sensor_name,
                ContactSensorCfg,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{button.name}",
                    filter_prim_paths_expr=contact_against_prim_paths,
                ),
            )
        )
    return make_configclass("ButtonContactSceneCfg", fields)()


def button_sequence_contact_reached(env) -> Any:
    """Unified checker — signature is just ``(env)``.

    Pulls "who the task controls" (the task's ``sequence_buttons`` / their contact
    sensors) plus the force threshold and instruction-order flag off the live
    ``ManaTask``, then reuses the shared ``contact_sequence_reached`` predicate.
    No params dict: every input is read from the Task at eval time.
    """
    from isaaclab.managers import SceneEntityCfg

    task = mana_task(env)
    return contact_sequence_reached(
        env,
        contact_sensor_cfgs=[SceneEntityCfg(name) for name in task.contact_sensor_names],
        sequence_length=len(task.sequence_buttons),
        force_threshold=float(task.cfg.contact_force_threshold),
        order_from_instruction=bool(task.cfg.runtime_plan_id or task.cfg.runtime_order_path),
    )


@configclass
class ButtonContactTerminationsCfg:
    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING

    def __init__(self):
        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        self.success = TerminationTermCfg(func=button_sequence_contact_reached, params={})


def make_button_contact_evaluation_cfg(
    contact_sensor_cfgs: list[SceneEntityCfg],
    force_threshold: float,
    articulation_home_atol: float,
    articulation_home_after_seconds: float,
):
    from isaaclab.managers import SceneEntityCfg
    from isaaclab_arena.utils.configclass import make_configclass

    fields = []
    for index in range(1, len(contact_sensor_cfgs) + 1):
        fields.append(
            (
                f"contact_sequence_{index}",
                TerminationTermCfg,
                TerminationTermCfg(
                    func=contact_sequence_reached,
                    params={
                        "contact_sensor_cfgs": contact_sensor_cfgs,
                        "sequence_length": index,
                        "force_threshold": force_threshold,
                        "order_from_instruction": True,
                    },
                ),
            )
        )
    fields.append(
        (
            "articulation_home_after_time",
            TerminationTermCfg,
            TerminationTermCfg(
                func=articulation_joints_near_default_after_time,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "atol": articulation_home_atol,
                    "after_seconds": articulation_home_after_seconds,
                },
            ),
        )
    )
    return make_configclass("ButtonContactEvaluationCfg", fields)()


# scene_layout_solver runtime context consumer bridge
@dataclass
class ButtonContactRuntimeConsumer:
    """Task-owned runtime consumer for schema-vNext button-order episodes."""

    plan_id: str
    order_path: str
    instruction_field: str
    buttons_field: str
    instruction_template: str

    def on_reset(self, env: Any, env_ids: Any = None) -> bool:
        applied = False
        for env_id in _env_ids(env, env_ids):
            applied = bool(self._apply_env_state(env, env_id=env_id)) or applied
        return applied

    def on_benchmark_step(
        self,
        ctx: Any,
        policy: Any,
        instruction: str,
        set_instruction: Any = None,
    ) -> bool:
        env = getattr(ctx, "gym_env", None)
        if env is None:
            return False
        runtime_instruction = self._apply_env_state(env, env_id=0)
        if callable(set_instruction) and runtime_instruction != getattr(ctx, "_mana_last_runtime_instruction", None):
            set_instruction(policy, runtime_instruction)
            setattr(ctx, "_mana_last_runtime_instruction", runtime_instruction)
        return True

    def _apply_env_state(self, env: Any, *, env_id: int) -> str:
        buttons = self._runtime_buttons(env, env_id=env_id)
        instruction = self._runtime_instruction(env, env_id=env_id, buttons=buttons)
        setattr(env, "_mana_benchmark_instruction", instruction)
        by_env = getattr(env, "_mana_button_contact_order_by_env", None)
        if not isinstance(by_env, dict):
            by_env = {}
            setattr(env, "_mana_button_contact_order_by_env", by_env)
        by_env[int(env_id)] = list(buttons)
        return instruction

    def _runtime_context(self, env: Any, *, env_id: int) -> Mapping[str, Any]:

        context = get_runtime_context(env, plan_id=self.plan_id, env_id=env_id)
        if not isinstance(context, Mapping) or not context:
            raise RuntimeError(f"button_contact_runtime_context_missing: plan_id={self.plan_id} env_id={env_id}")
        return context

    def _runtime_buttons(self, env: Any, *, env_id: int) -> list[str]:

        missing = object()
        raw = missing
        if self.buttons_field:
            raw = runtime_template_field(env, self.buttons_field, plan_id=self.plan_id, env_id=env_id, default=missing)
        if raw is missing:
            raw = _resolve_path(self._runtime_context(env, env_id=env_id), self.order_path)
        buttons = _coerce_button_order(raw)
        if not buttons:
            raise RuntimeError(f"button_contact_runtime_buttons_empty: plan_id={self.plan_id} env_id={env_id}")
        return buttons

    def _runtime_instruction(self, env: Any, *, env_id: int, buttons: Sequence[str]) -> str:
        if not self.instruction_field:
            return self._format_instruction(buttons)

        missing = object()
        value = runtime_template_field(
            env, self.instruction_field, plan_id=self.plan_id, env_id=env_id, default=missing
        )
        if value is missing:
            raise RuntimeError(
                f"button_contact_runtime_instruction_missing: plan_id={self.plan_id} env_id={env_id} field={self.instruction_field}"
            )
        instruction = str(value or "").strip()
        if not instruction:
            raise RuntimeError(
                f"button_contact_runtime_instruction_empty: plan_id={self.plan_id} env_id={env_id} field={self.instruction_field}"
            )
        return instruction

    def _format_instruction(self, buttons: Sequence[str]) -> str:
        return self.instruction_template.format(button_names=", ".join(str(button) for button in buttons))

def _coerce_button_order(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if isinstance(value.get("object_names"), list):
            value = value["object_names"]
        elif isinstance(value.get("values"), list):
            value = value["values"]
        else:
            return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        return []
    out: list[str] = []
    for item in values:
        name = str(item.get("object") if isinstance(item, Mapping) else item).strip()
        if name:
            out.append(name)
    return out


def _resolve_path(context: Any, path: str) -> Any:
    current = context
    for part in [item for item in str(path).split(".") if item]:
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise KeyError(path)
    return current


def _env_ids(env: Any, env_ids: Any) -> list[int]:
    if env_ids is None:
        return [0]
    if hasattr(env_ids, "detach"):
        values = env_ids.detach().cpu().flatten().tolist()
        return [int(value) for value in values] or [0]
    if isinstance(env_ids, slice):
        num_envs = int(getattr(env, "num_envs", 1) or 1)
        return list(range(num_envs))[env_ids]
    try:
        values = list(env_ids)
    except TypeError:
        values = [env_ids]
    return [int(value) for value in values] or [0]


ButtonContactTaskCFG.class_type = ButtonContactTask
