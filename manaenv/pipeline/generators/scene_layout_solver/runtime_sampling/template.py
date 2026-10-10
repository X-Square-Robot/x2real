from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_TEMPLATE_FIELD = re.compile(r"\{([^{}]+)\}")


class RuntimeTemplateError(ValueError):
    pass


def resolve_runtime_template_fields(
    template_spec: Mapping[str, Any] | None, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve task field bindings from a runtime sampling context.

    The resolver is intentionally independent of ManaEnv config objects.  It
    returns a flat mapping from target field path to the value that a ManaEnv
    RuntimeRef or bridge adapter can apply.
    """

    if not isinstance(template_spec, Mapping) or not template_spec:
        return {}
    raw_fields = template_spec.get("fields")
    if not isinstance(raw_fields, Mapping):
        return {}
    out: dict[str, Any] = {}
    for field_path, raw_rule in raw_fields.items():
        target = str(field_path).strip()
        if not target:
            continue
        out[target] = resolve_runtime_template_rule(raw_rule, context)
    return out


def resolve_runtime_template_rule(raw_rule: Any, context: Mapping[str, Any]) -> Any:
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


def _resolve_path(context: Mapping[str, Any], path: str) -> Any:
    current: Any = context
    for raw_part in [part for part in str(path).split(".") if part]:
        if isinstance(current, Mapping):
            if raw_part not in current:
                raise RuntimeTemplateError(f"runtime template path not found: {path}")
            current = current[raw_part]
        elif isinstance(current, list):
            try:
                current = current[int(raw_part)]
            except (ValueError, IndexError) as exc:
                raise RuntimeTemplateError(f"runtime template list index invalid: {path}") from exc
        else:
            raise RuntimeTemplateError(f"runtime template path cannot descend into scalar: {path}")
    return current


def _format_template_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_format_template_value(item) for item in value)
    if isinstance(value, tuple):
        return ", ".join(_format_template_value(item) for item in value)
    if isinstance(value, Mapping):
        if "display_name" in value:
            return str(value["display_name"])
        if "value" in value:
            return str(value["value"])
    return str(value)
