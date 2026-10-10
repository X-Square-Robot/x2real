from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from manaenv.pipeline.factories.rollout_factory_cfg import RolloutOrderCFG


def attach_task_runtime_consumer_event_for_task(task: Any) -> bool:
    """Attach the generic reset dispatcher for task-template runtime consumers."""

    consumers = resolve_task_runtime_consumers(task)
    if not consumers:
        return False
    runtime_cfgs = getattr(task, "runtime_cfgs", None)
    if not isinstance(runtime_cfgs, dict):
        runtime_cfgs = {}
        setattr(task, "runtime_cfgs", runtime_cfgs)
    event_terms = runtime_cfgs.get("event_terms")

    from isaaclab.managers import EventTermCfg
    from isaaclab_arena.utils.configclass import make_configclass

    field_name = "task_runtime__consumer"
    if event_terms is not None and hasattr(event_terms, field_name):
        return False
    term_cfg = EventTermCfg(
        func=task_runtime_consumer_on_reset,
        mode="reset",
        params={"consumers": consumers},
    )
    if event_terms is None:
        EventsCfg = make_configclass("TaskRuntimeConsumerEventsCfg", [(field_name, type(term_cfg), term_cfg)])
        runtime_cfgs["event_terms"] = EventsCfg()
        return True
    fields = _event_term_config_fields(event_terms)
    fields.append((field_name, type(term_cfg), term_cfg))
    EventsCfg = make_configclass("TaskRuntimeConsumerEventsCfg", fields)
    runtime_cfgs["event_terms"] = EventsCfg()
    return True


def task_runtime_consumer_on_reset(env: Any, env_ids: Any, *, consumers: Sequence[Any] | None = None) -> None:
    """IsaacLab reset event that delegates task-specific runtime logic to task templates."""

    dispatch_task_runtime_consumer_hook(
        env,
        "on_reset",
        consumers=consumers,
        env_ids=env_ids,
    )


task_runtime_consumer_on_reset._scene_layout_task_runtime_consumer_event = True


def dispatch_benchmark_runtime_consumers(
    ctx: Any,
    *,
    policy: Any,
    instruction: str,
    order: RolloutOrderCFG | None = None,
    set_instruction: Any = None,
) -> bool:
    """Dispatch benchmark-loop updates to task-template runtime consumers."""

    env = getattr(ctx, "gym_env", None)
    if env is None:
        return False

    order_dict = _order_to_mapping(order)

    results = dispatch_task_runtime_consumer_hook(
        env,
        "on_benchmark_step",
        ctx=ctx,
        policy=policy,
        instruction=instruction,
        order=order_dict,
        set_instruction=set_instruction,
    )
    return any(bool(result) for result in results)


def _order_to_mapping(order: Any) -> dict[str, Any]:
    if order is None:
        return {}
    if isinstance(order, Mapping):
        return dict(order)
    if dataclasses.is_dataclass(order):
        return dataclasses.asdict(order)
    return (
        {key: value for key, value in vars(order).items() if not key.startswith("_")}
        if hasattr(order, "__dict__")
        else {}
    )


def dispatch_task_runtime_consumer_hook(
    env: Any,
    hook_name: str,
    *,
    consumers: Sequence[Any] | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Call one named hook on all task runtime consumers available to *env*."""

    active = list(consumers or _resolve_env_task_runtime_consumers(env))
    if active:
        setattr(env, "_mana_task_runtime_consumers", active)
    out: list[Any] = []
    for consumer in active:
        hook = getattr(consumer, hook_name, None)
        if callable(hook):
            out.append(_call_with_supported_kwargs(hook, env=env, **kwargs))
    return out


def resolve_task_runtime_consumers(source: Any) -> tuple[Any, ...]:
    """Return runtime consumers exposed by a task/template/env-like object."""

    if source is None:
        return ()
    existing = getattr(source, "_mana_task_runtime_consumers", None)
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        return tuple(existing)

    task = _resolve_task(source)
    if task is None:
        return ()
    task_existing = getattr(task, "_mana_task_runtime_consumers", None)
    if isinstance(task_existing, Sequence) and not isinstance(task_existing, (str, bytes)):
        return tuple(task_existing)

    consumers: list[Any] = []
    for method_name in ("make_runtime_context_consumer", "get_runtime_context_consumer"):
        method = getattr(task, method_name, None)
        if callable(method):
            value = method()
            consumers.extend(_normalize_consumers(value))
    if not consumers and any(callable(getattr(task, name, None)) for name in ("on_reset", "on_benchmark_step")):
        consumers.append(task)
    unique = _unique_consumers(consumers)
    if unique:
        setattr(task, "_mana_task_runtime_consumers", unique)
    return unique


def _resolve_env_task_runtime_consumers(env: Any) -> tuple[Any, ...]:
    existing = getattr(env, "_mana_task_runtime_consumers", None)
    if isinstance(existing, Sequence) and not isinstance(existing, (str, bytes)):
        return tuple(existing)
    task = _resolve_task(env)
    return resolve_task_runtime_consumers(task)


def _resolve_task(source: Any) -> Any:
    if source is None:
        return None
    arena = getattr(source, "_mana_assembled_environment", None)
    if arena is not None and getattr(arena, "task", None) is not None:
        return getattr(arena, "task")
    if getattr(source, "task", None) is not None:
        return getattr(source, "task")
    arena_env = getattr(source, "_arena_env", None)
    if arena_env is not None and getattr(arena_env, "task", None) is not None:
        return getattr(arena_env, "task")
    if getattr(source, "_mana_task", None) is not None:
        return getattr(source, "_mana_task")
    return source


def _normalize_consumers(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    if isinstance(value, Iterable):
        return [item for item in value if item is not None]
    return [value]


def _unique_consumers(consumers: Sequence[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    seen: set[int] = set()
    for consumer in consumers:
        key = id(consumer)
        if key in seen:
            continue
        out.append(consumer)
        seen.add(key)
    return tuple(out)


def _call_with_supported_kwargs(func: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return func(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in params}
    return func(**accepted)


def _event_term_config_fields(event_terms: Any) -> list[tuple[str, type, Any]]:
    fields = getattr(event_terms, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        names = [str(name) for name in fields]
    else:
        names = [str(name) for name in vars(event_terms)]
    out: list[tuple[str, type, Any]] = []
    seen: set[str] = set()
    for name in names:
        if name in seen or not hasattr(event_terms, name):
            continue
        value = getattr(event_terms, name)
        out.append((name, type(value), value))
        seen.add(name)
    return out
