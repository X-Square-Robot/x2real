from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_NAMESPACE = "scene_layout"
_TEMPLATE_FIELD = re.compile(r"\{([^{}]+)\}")


def set_runtime_context(
    env: Any,
    namespace: str,
    plan_id: str,
    env_ids: Sequence[int],
    payloads: Mapping[int, Mapping[str, Any]] | Mapping[str, Any],
) -> None:
    """Store per-env runtime context on an environment.

    The storage shape is intentionally plain Python so reset events can call it
    without depending on Isaac Lab internals.
    """

    namespace_key = str(namespace or DEFAULT_NAMESPACE)
    plan_key = str(plan_id or "runtime")
    state = getattr(env, "_mana_runtime_context", None)
    if not isinstance(state, dict):
        state = {}
        setattr(env, "_mana_runtime_context", state)
    namespace_state = state.setdefault(namespace_key, {})
    if not isinstance(namespace_state, dict):
        namespace_state = {}
        state[namespace_key] = namespace_state
    plan_state = namespace_state.setdefault(plan_key, {})
    if not isinstance(plan_state, dict):
        plan_state = {}
        namespace_state[plan_key] = plan_state

    for env_id in env_ids:
        key = int(env_id)
        if key in payloads:
            payload = payloads[key]  # type: ignore[index]
        elif str(key) in payloads:
            payload = payloads[str(key)]  # type: ignore[index]
        else:
            payload = payloads
        plan_state[key] = dict(payload)

    setattr(env, "_mana_runtime_context_latest", {"namespace": namespace_key, "plan_id": plan_key})


def get_runtime_context(
    env: Any,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    plan_id: str | None = None,
    env_id: int | None = None,
) -> Any:
    namespace_key = str(namespace or DEFAULT_NAMESPACE)
    plan_key = str(plan_id or _latest_plan_id(env, namespace_key) or "runtime")

    state = getattr(env, "_mana_runtime_context", None)
    if isinstance(state, Mapping):
        namespace_state = state.get(namespace_key)
        if isinstance(namespace_state, Mapping):
            plan_state = namespace_state.get(plan_key)
            if isinstance(plan_state, Mapping):
                return dict(plan_state) if env_id is None else plan_state.get(int(env_id), {})

    return {} if env_id is not None else {}


def resolve_runtime_ref(
    env: Any,
    ref: Any,
    *,
    env_ids: Sequence[int] | None = None,
    default: Any = None,
) -> dict[int, Any] | Any:
    """Resolve a RuntimeRef-like mapping.

    Supported authored form:

    ``{"runtime_ref": "outputs.button_order.object_names", "plan_id": "button_order"}``
    """

    if not is_runtime_ref(ref):
        return default
    path = str(ref.get("runtime_ref") or ref.get("from") or ref.get("path") or "")
    namespace = str(ref.get("namespace") or DEFAULT_NAMESPACE)
    plan_id = str(ref.get("plan_id") or "")
    ids = _env_ids(env, env_ids)
    values: dict[int, Any] = {}
    for env_id in ids:
        context = get_runtime_context(env, namespace=namespace, plan_id=plan_id or None, env_id=env_id)
        try:
            values[int(env_id)] = _resolve_path(context, path)
        except KeyError:
            if "default" in ref:
                values[int(env_id)] = ref["default"]
            else:
                values[int(env_id)] = default
    if env_ids is None and len(values) == 1:
        return next(iter(values.values()))
    return values


def is_runtime_ref(value: Any) -> bool:
    return isinstance(value, Mapping) and any(key in value for key in ("runtime_ref", "from", "path"))


def runtime_template_field(
    env: Any,
    field_path: str,
    *,
    namespace: str = DEFAULT_NAMESPACE,
    plan_id: str | None = None,
    env_id: int = 0,
    default: Any = None,
) -> Any:
    context = get_runtime_context(env, namespace=namespace, plan_id=plan_id, env_id=env_id)
    fields = context.get("template_fields") if isinstance(context, Mapping) else None
    if isinstance(fields, Mapping) and field_path in fields:
        return fields[field_path]
    return default


def schema_vnext_runtime_context_ready(
    env: Any,
    *,
    plan_id: str,
    required_template_fields: Sequence[str] | None = None,
    require_runtime_context: bool = True,
):
    """Return per-env readiness of schema-vNext runtime context."""

    import torch

    num_envs = int(getattr(env, "num_envs", 1) or 1)
    device = getattr(env, "device", torch.device("cpu"))
    out = torch.zeros(num_envs, dtype=torch.bool, device=device)
    required = tuple(str(item) for item in (required_template_fields or ()))

    for env_id in range(num_envs):
        context = get_runtime_context(env, plan_id=str(plan_id or "runtime"), env_id=env_id)
        if not isinstance(context, Mapping) or not context:
            out[env_id] = not bool(require_runtime_context)
            continue
        fields = context.get("template_fields")
        if required and not isinstance(fields, Mapping):
            out[env_id] = False
            continue
        missing = [field for field in required if field not in fields]
        out[env_id] = not missing
    return out


def resolve_runtime_template_fields(
    template_spec: Mapping[str, Any] | None, context: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(template_spec, Mapping) or not template_spec:
        return {}
    raw_fields = template_spec.get("fields")
    if not isinstance(raw_fields, Mapping):
        return {}
    out: dict[str, Any] = {}
    for field_path, raw_rule in raw_fields.items():
        target = str(field_path).strip()
        if target:
            out[target] = _resolve_template_rule(raw_rule, context)
    return out


def _resolve_template_rule(raw_rule: Any, context: Mapping[str, Any]) -> Any:
    if isinstance(raw_rule, str):
        return _resolve_path(context, raw_rule)
    if not isinstance(raw_rule, Mapping):
        return raw_rule
    if "from" in raw_rule:
        return _resolve_path(context, str(raw_rule["from"]))
    if "template" in raw_rule:
        template = str(raw_rule["template"])
        return _TEMPLATE_FIELD.sub(
            lambda match: _format_template_value(_resolve_path(context, match.group(1))), template
        )
    if "value" in raw_rule:
        return raw_rule["value"]
    return dict(raw_rule)


def _format_template_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_template_value(item) for item in value)
    if isinstance(value, Mapping):
        if "display_name" in value:
            return str(value["display_name"])
        if "value" in value:
            return str(value["value"])
    return str(value)


def _latest_plan_id(env: Any, namespace: str) -> str:
    latest = getattr(env, "_mana_runtime_context_latest", None)
    if isinstance(latest, Mapping) and str(latest.get("namespace") or "") == namespace:
        return str(latest.get("plan_id") or "")
    return ""


def _env_ids(env: Any, env_ids: Sequence[int] | None) -> list[int]:
    if env_ids is not None:
        detach = getattr(env_ids, "detach", None)
        if callable(detach):
            return [int(v) for v in detach().cpu().flatten().tolist()]
        return [int(v) for v in env_ids]
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    return list(range(num_envs))


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
