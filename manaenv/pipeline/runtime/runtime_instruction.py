"""Render ``definition.runtime_instruction`` against schema-vNext runtime context.

Authored templates bind placeholders to reset-time context paths:

- ``{direction}`` → ``outputs.place_beside.phrase`` (beside tasks)
- ``objects`` + ``tag_mode: kind`` → ``outputs.runtime_instruction_tags.<name>``
  (species / class labels from the active asset id, e.g. peach)

Placeholders this module cannot bind (e.g. ``{nut_color}``) stay with the
bridge ``apply_runtime_instruction_for_envs`` metadata renderer; this helper
returns no ``runtime_templates`` fields for those templates.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from manaenv.runtime_context import get_runtime_context

_INSTRUCTION_FIELD = "instruction.instruction"
_OBJECT_TAG_PREFIX = "outputs.runtime_instruction_tags."
_TEMPLATE_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")

# Placeholders in authored runtime_instruction.template → context paths.
_PLACEHOLDER_BINDS: dict[str, str] = {
    "direction": "outputs.place_beside.phrase",
}

_TRAILING_INDEX = re.compile(r"^(.*?)(?:[_-]?\d+)$")


def definition_runtime_instruction(task_payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    definition = task_payload.get("definition") if isinstance(task_payload, Mapping) else None
    if not isinstance(definition, Mapping):
        return None
    spec = definition.get("runtime_instruction")
    return dict(spec) if isinstance(spec, Mapping) else None


def runtime_instruction_templates(task_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Build ``runtime_templates`` fields from ``definition.runtime_instruction``."""
    spec = definition_runtime_instruction(task_payload)
    if not spec:
        return {}
    template = str(spec.get("template") or "").strip()
    if not template:
        return {}
    definition = task_payload.get("definition") if isinstance(task_payload, Mapping) else {}
    definition = definition if isinstance(definition, Mapping) else {}
    # Direction phrases come from beside_placement; skip if that contract is absent.
    if "{direction}" in template and not isinstance(definition.get("beside_placement"), Mapping):
        return {}
    rule = _bind_placeholders(template, spec)
    # Bare tokens (no ``.``) are not context paths. Color / local-metadata
    # placeholders such as ``{nut_color}`` are rendered later by
    # ``apply_runtime_instruction_for_envs``; emitting them here raises
    # ``RuntimeTemplateError: runtime template path not found``.
    if any("." not in name for name in _TEMPLATE_PLACEHOLDER.findall(rule)):
        return {}
    return {"fields": {_INSTRUCTION_FIELD: {"template": rule}}}


def _bind_placeholders(template: str, spec: Mapping[str, Any]) -> str:
    out = str(template)
    for name, path in _PLACEHOLDER_BINDS.items():
        out = out.replace("{" + name + "}", "{" + path + "}")
    objects = spec.get("objects") if isinstance(spec.get("objects"), Mapping) else {}
    tag_mode = str(spec.get("tag_mode") or "").strip() or "kind"
    if tag_mode != "kind" or not objects:
        return out
    for placeholder in objects:
        key = str(placeholder).strip()
        if not key:
            continue
        token = "{" + key + "}"
        if token not in out:
            continue
        out = out.replace(token, "{" + _OBJECT_TAG_PREFIX + key + "}")
    return out


def _kind_label_from_asset(asset_id: str, *, fallback: str) -> str:
    """Species/class label: asset-id token first, else local taxonomy metadata."""
    kind = kind_from_asset_ref(asset_id)
    if kind:
        return kind
    fallback_label = str(fallback or "object").strip() or "object"
    if not str(asset_id or "").strip():
        return fallback_label
    try:
        from manaenv.assets.local_discovery import resolve_usd_path
        from manaenv.utils.asset_metadata import load_asset_metadata, taxonomy_kind
    except Exception:  # noqa: BLE001 — keep instruction path offline-safe
        return fallback_label
    try:
        usd_path = str(resolve_usd_path(asset_id) or "").strip()
    except Exception:  # noqa: BLE001
        usd_path = ""
    if not usd_path:
        return fallback_label
    try:
        meta = load_asset_metadata(usd_path)
        label = str(taxonomy_kind(meta, fallback=fallback_label) or "").strip()
        return label or fallback_label
    except Exception:  # noqa: BLE001 — missing metadata.json is common for pool assets
        return fallback_label


def attach_runtime_instruction_tags(
    context: dict[str, Any],
    *,
    task_payload: Mapping[str, Any] | None,
) -> None:
    """Fill ``outputs.runtime_instruction_tags`` from applied runtime assets."""
    spec = definition_runtime_instruction(task_payload)
    if not spec:
        return
    objects = spec.get("objects") if isinstance(spec.get("objects"), Mapping) else {}
    if not objects:
        return
    tag_mode = str(spec.get("tag_mode") or "").strip() or "kind"
    applied = context.get("applied") if isinstance(context.get("applied"), Mapping) else {}
    assets = applied.get("assets") if isinstance(applied.get("assets"), Mapping) else {}
    by_object = assets.get("by_object") if isinstance(assets, Mapping) else {}
    if not isinstance(by_object, Mapping):
        by_object = {}
    tags: dict[str, str] = {}
    for placeholder, object_id in objects.items():
        key = str(placeholder).strip()
        obj = str(object_id).strip()
        if not key or not obj:
            continue
        report = by_object.get(obj) if isinstance(by_object.get(obj), Mapping) else {}
        asset_id = str((report or {}).get("asset_id") or "").strip()
        if tag_mode == "kind":
            tags[key] = _kind_label_from_asset(asset_id, fallback=obj)
        else:
            tags[key] = asset_id or obj
    context.setdefault("outputs", {})["runtime_instruction_tags"] = tags


_UUID_TOKEN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SOURCE_PREFIXES = frozenset(
    {
        "omni6dpose",
        "lanejin",
        "lanejin0518",
        "benchmark",
        "blenderkit",
        "blenderkitmodels",
        "procthor",
        "models",
    }
)


def _is_uuid_token(token: str) -> bool:
    return bool(_UUID_TOKEN.match(str(token or "").strip()))


def kind_from_asset_ref(asset_ref: str) -> str:
    """Derive a short species/class label from a registry asset id or name.

    Examples:
      omni6DPose:omni6DPose_lemon_000 → lemon
      lanejin_0518__lanejin_apple_021 → apple
      benchmark__benchmark_peach_000 → peach

    BlenderKit UUID tails (``blenderkit_<uuid>``) return empty so callers can
    fall back to local taxonomy metadata or the logical object id (``fruit``).
    """
    text = str(asset_ref or "").strip()
    if not text:
        return ""
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace("__", "_")
    parts = [part for part in text.split("_") if part]
    while parts and parts[-1].isdigit():
        parts.pop()
    # Walk from the tail for a human species token; skip UUID / dataset prefixes.
    for part in reversed(parts):
        token = part.strip()
        if not token or _is_uuid_token(token):
            continue
        compact = token.replace("-", "").lower()
        if compact in _SOURCE_PREFIXES:
            continue
        if compact.startswith("lanejin") or compact.startswith("blenderkit"):
            continue
        match = _TRAILING_INDEX.match(token)
        if match and match.group(1):
            token = match.group(1)
        label = token.lower()
        if label and not _is_uuid_token(label):
            return label
    return ""


def instruction_from_template_fields(context: Mapping[str, Any] | None) -> str:
    fields = context.get("template_fields") if isinstance(context, Mapping) else None
    if not isinstance(fields, Mapping):
        return ""
    value = fields.get(_INSTRUCTION_FIELD)
    return str(value or "").strip()


def write_runtime_instruction_from_context(
    env: Any,
    *,
    env_id: int,
    context: Mapping[str, Any] | None,
) -> str:
    """Persist the rendered instruction into the per-env active store."""
    text = instruction_from_template_fields(context)
    if not text:
        return ""
    from manaenv.pipeline.runtime.active_instruction import write_active_instruction

    write_active_instruction(
        env,
        int(env_id),
        text,
        num_envs=int(getattr(env, "num_envs", 1) or 1),
        reset_owned=True,
    )
    return text


@dataclass
class RuntimeInstructionConsumer:
    """Publish rendered ``instruction.instruction`` to policy / benchmark hooks."""

    plan_id: str = "runtime"

    def on_reset(self, env: Any, env_ids: Any = None) -> bool:
        applied = False
        for env_id in _env_ids(env, env_ids):
            context = _runtime_context(env, plan_id=self.plan_id, env_id=env_id)
            text = write_runtime_instruction_from_context(env, env_id=env_id, context=context)
            if text:
                setattr(env, "_mana_benchmark_instruction", text)
                applied = True
        return applied

    def on_benchmark_step(
        self,
        ctx: Any,
        policy: Any,
        instruction: str,
        set_instruction: Any = None,
    ) -> bool:
        del instruction
        env = getattr(ctx, "gym_env", None)
        if env is None:
            return False
        context = _runtime_context(env, plan_id=self.plan_id, env_id=0)
        text = instruction_from_template_fields(context)
        if not text:
            return False
        if callable(set_instruction) and text != getattr(ctx, "_mana_last_runtime_instruction", None):
            set_instruction(policy, text)
            setattr(ctx, "_mana_last_runtime_instruction", text)
        setattr(env, "_mana_benchmark_instruction", text)
        return True


def _runtime_context(env: Any, *, plan_id: str, env_id: int) -> Mapping[str, Any]:

    context = get_runtime_context(env, plan_id=str(plan_id or "runtime"), env_id=int(env_id))
    return context if isinstance(context, Mapping) else {}


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


__all__ = [
    "RuntimeInstructionConsumer",
    "attach_runtime_instruction_tags",
    "definition_runtime_instruction",
    "instruction_from_template_fields",
    "kind_from_asset_ref",
    "runtime_instruction_templates",
    "write_runtime_instruction_from_context",
]
