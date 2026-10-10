from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field
from typing import TYPE_CHECKING, Any

from isaaclab.utils.configclass import configclass
from isaaclab_arena.tasks.task_base import ManaTask, TaskCFG

from manaenv.runtime_context import get_runtime_context, schema_vnext_runtime_context_ready
from manaenv.terms.viewers import look_at_object

if TYPE_CHECKING:
    from isaaclab.managers import TerminationTermCfg
    from isaaclab_arena.assets.asset import Asset
    from isaaclab_arena.metrics.metric_base import MetricBase


@dataclass
class SchemaVNextProbeTaskCFG(TaskCFG):
    """Timeout-only task template for schema-vNext runtime/capture examples."""

    class_type: type[SchemaVNextProbeTask] | None = None
    probe_mode: str = "runtime_context_capture"
    completion_mode: str = "timeout_only"
    primary_target: Asset | list[Asset] | str = ""
    viewer_target: Asset | list[Asset] | str = ""
    viewer_offset: list[float] = field(default_factory=lambda: [-1.0, 0.0, 1.5])
    runtime_plan_id: str = ""
    runtime_templates: dict[str, Any] = field(default_factory=dict)
    required_template_fields: list[str] = field(default_factory=list)
    require_runtime_context: bool = True
    component_sections: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.class_type is None:
            self.class_type = SchemaVNextProbeTask


class SchemaVNextProbeTask(ManaTask):
    """Shared ManaEnv task-template entry for schema-vNext visual/runtime probes.

    The probe template's real task semantics are runtime-context validation:
    every episode must run through the schema-vNext reset event and publish the
    required context/template fields. It intentionally does not claim
    task-specific robot success or manipulation semantics.
    """

    def __init__(self, cfg: SchemaVNextProbeTaskCFG):
        super().__init__(cfg=cfg)
        self.cfg: SchemaVNextProbeTaskCFG = cfg
        self.primary_target: Asset | None = None
        self.viewer_target: Asset | None = None
        self.termination_cfg = None

    def finalize(self, scene: Any = None, robot: Any = None) -> SchemaVNextProbeTask:
        del scene, robot
        self._validate_boundary()
        self.primary_target = self._first_asset(self.cfg.primary_target)
        self.viewer_target = self._first_asset(self.cfg.viewer_target) or self.primary_target
        self.termination_cfg = SchemaVNextProbeTerminationsCfg(
            plan_id=self._runtime_plan_id(),
            required_template_fields=tuple(str(item) for item in self.cfg.required_template_fields),
            require_runtime_context=bool(self.cfg.require_runtime_context),
        )
        return self

    def _validate_boundary(self) -> None:
        if self.cfg.completion_mode != "timeout_only":
            raise ValueError(
                "SchemaVNextProbeTask only supports completion_mode='timeout_only'; "
                "use a dedicated task template for success or evaluation semantics."
            )
        if self.cfg.probe_mode not in {"runtime_context_capture", "visual_runtime_probe"}:
            raise ValueError(
                "SchemaVNextProbeTask probe_mode must be 'runtime_context_capture' or 'visual_runtime_probe'."
            )

    @staticmethod
    def _first_asset(value: Any) -> Asset | None:
        from isaaclab_arena.assets.asset import Asset

        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, Asset):
                return item
        return None

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_events_cfg(self):
        return None

    def get_evaluation_term_cfg(self):
        return make_schema_vnext_probe_evaluation_term_cfg(
            plan_id=self._runtime_plan_id(),
            required_template_fields=tuple(str(item) for item in self.cfg.required_template_fields),
            require_runtime_context=bool(self.cfg.require_runtime_context),
        )

    def get_metrics(self) -> list[MetricBase]:
        from isaaclab_arena.metrics.success_rate import SuccessRateMetric

        return [SuccessRateMetric()]

    def get_prompt(self) -> str:
        prompt = getattr(self.cfg, "prompt", "")
        if prompt:
            return str(prompt)
        definition = getattr(self.cfg, "definition", {}) or {}
        return str(definition.get("prompt") or "")

    def get_default_visualizer_cfg(self):
        if self.viewer_target is None:
            return None
        return look_at_object(
            lookat_object=self.viewer_target,
            offset=self.cfg.viewer_offset,
        )

    def make_runtime_templates(self) -> dict[str, Any]:
        return dict(self.cfg.runtime_templates or {})

    def make_runtime_context_consumer(self):
        plan_id = self._runtime_plan_id()
        if not plan_id:
            return None
        return SchemaVNextProbeRuntimeConsumer(
            plan_id=plan_id,
            required_template_fields=tuple(str(item) for item in self.cfg.required_template_fields),
            require_runtime_context=bool(self.cfg.require_runtime_context),
        )

    def _runtime_plan_id(self) -> str:
        if self.cfg.runtime_plan_id:
            return str(self.cfg.runtime_plan_id)
        definition = getattr(self.cfg, "definition", {}) or {}
        layout = definition.get("layout") if isinstance(definition, Mapping) else {}
        runtime = layout.get("runtime_sample") if isinstance(layout, Mapping) else {}
        if isinstance(runtime, Mapping):
            return str(runtime.get("id") or "runtime")
        return "runtime"


@configclass
class SchemaVNextProbeTerminationsCfg:
    time_out: TerminationTermCfg = MISSING
    success: TerminationTermCfg = MISSING

    def __init__(
        self,
        *,
        plan_id: str,
        required_template_fields: tuple[str, ...] = (),
        require_runtime_context: bool = True,
    ):
        import isaaclab.envs.mdp as mdp_isaac_lab
        from isaaclab.managers import TerminationTermCfg

        self.time_out = TerminationTermCfg(func=mdp_isaac_lab.time_out)
        self.success = TerminationTermCfg(
            func=schema_vnext_runtime_context_ready,
            params={
                "plan_id": str(plan_id or "runtime"),
                "required_template_fields": tuple(str(item) for item in required_template_fields),
                "require_runtime_context": bool(require_runtime_context),
            },
        )


def make_schema_vnext_probe_evaluation_term_cfg(
    *,
    plan_id: str,
    required_template_fields: tuple[str, ...] = (),
    require_runtime_context: bool = True,
) -> TerminationTermCfg:
    from isaaclab.managers import TerminationTermCfg

    params = {
        "plan_id": str(plan_id or "runtime"),
        "required_template_fields": tuple(str(item) for item in required_template_fields),
        "require_runtime_context": bool(require_runtime_context),
        "_label": "schema-vNext runtime context ready",
        "_score": 1.0,
    }
    return TerminationTermCfg(
        func=schema_vnext_runtime_context_ready,
        params={
            "_sub_term_defs": [
                {
                    "cfg": TerminationTermCfg(
                        func=schema_vnext_runtime_context_ready,
                        params=params,
                    )
                }
            ]
        },
    )


@dataclass
class SchemaVNextProbeRuntimeConsumer:
    """Task-owned runtime-context consumer for schema-vNext probe examples."""

    plan_id: str
    required_template_fields: tuple[str, ...] = ()
    require_runtime_context: bool = True

    def on_reset(self, env: Any, env_ids: Any = None) -> bool:
        applied = False
        for env_id in _env_ids(env, env_ids):
            context = self._runtime_context(env, env_id=env_id)
            self._store_context(env, env_id=env_id, context=context)
            self._validate_template_fields(context, env_id=env_id)
            instruction = self._template_instruction(context)
            if instruction:
                setattr(env, "_mana_benchmark_instruction", instruction)
            applied = True
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
        context = self._runtime_context(env, env_id=0)
        runtime_instruction = self._template_instruction(context) or instruction
        if callable(set_instruction):
            set_instruction(policy, runtime_instruction)
        setattr(env, "_mana_benchmark_instruction", runtime_instruction)
        return True

    def _runtime_context(self, env: Any, *, env_id: int) -> Mapping[str, Any]:

        context = get_runtime_context(env, plan_id=self.plan_id, env_id=env_id)
        if not isinstance(context, Mapping) or not context:
            if self.require_runtime_context:
                raise RuntimeError(
                    f"schema_vnext_probe_runtime_context_missing: plan_id={self.plan_id} env_id={env_id}"
                )
            return {}
        return context

    def _store_context(self, env: Any, *, env_id: int, context: Mapping[str, Any]) -> None:
        by_env = getattr(env, "_mana_schema_vnext_probe_context_by_env", None)
        if not isinstance(by_env, dict):
            by_env = {}
            setattr(env, "_mana_schema_vnext_probe_context_by_env", by_env)
        by_env[int(env_id)] = dict(context)

    def _validate_template_fields(self, context: Mapping[str, Any], *, env_id: int) -> None:
        fields = context.get("template_fields")
        if not self.required_template_fields:
            return
        if not isinstance(fields, Mapping):
            raise RuntimeError(f"schema_vnext_probe_template_fields_missing: plan_id={self.plan_id} env_id={env_id}")
        missing = [field for field in self.required_template_fields if field not in fields]
        if missing:
            raise RuntimeError(
                f"schema_vnext_probe_template_field_missing: plan_id={self.plan_id} env_id={env_id} fields={missing}"
            )

    @staticmethod
    def _template_instruction(context: Mapping[str, Any]) -> str:
        fields = context.get("template_fields")
        if isinstance(fields, Mapping):
            value = fields.get("instruction.instruction")
            if value:
                return str(value)
        return ""


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


SchemaVNextProbeTaskCFG.class_type = SchemaVNextProbeTask
