"""Per-reset instruction rendering from local asset metadata for pool tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from manaenv.utils.asset_metadata import load_asset_metadata, taxonomy_kind
from manaenv.utils.asset_utils import resolve_active_asset_id, resolve_active_scene_entity_name
from manaenv.utils.debug_print import manaprint

# nut_on_screw lettered assets (metadata often lacks these shapes).
_SHAPE_LABEL_BY_ASSET_ID: dict[str, str] = {
    "nut_a": "square",
    "nut_b": "triangle",
    "nut_c": "hexagon",
    "nut_d": "cross",
    "screw_a": "square",
    "screw_b": "triangle",
    "screw_c": "hexagon",
    "screw_d": "cross",
}


def _definition_runtime_instruction(task_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    definition = task_payload.get("definition") if isinstance(task_payload.get("definition"), Mapping) else {}
    spec = definition.get("runtime_instruction") if isinstance(definition, Mapping) else None
    return dict(spec) if isinstance(spec, Mapping) else None


def _resolve_usd_path_for_asset(asset_id: str) -> str:
    from manaenv.assets.local_discovery import resolve_usd_path

    asset_name = str(asset_id).strip()
    if not asset_name:
        return ""
    try:
        return str(resolve_usd_path(asset_name) or "").strip()
    except KeyError:
        return ""


def _humanize_label(asset_id: str, logical_name: str) -> str:
    """Fallback instruction label when local metadata is unavailable."""
    from manaenv.pipeline.runtime.runtime_instruction import kind_from_asset_ref

    text = str(asset_id or logical_name or "object").strip()
    if text in _SHAPE_LABEL_BY_ASSET_ID:
        return _SHAPE_LABEL_BY_ASSET_ID[text]
    if text.startswith("bottle_"):
        return "bottle"
    kind = kind_from_asset_ref(text)
    if kind:
        return kind
    # Prefer the logical object id (fruit/bottle/…) over a raw UUID asset name.
    return str(logical_name or "object").strip() or "object"


def _placeholder_values_for_env(
    env: Any,
    *,
    env_id: int,
    plan_id: str,
    objects: Mapping[str, str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for placeholder, logical in objects.items():
        key = str(placeholder).strip()
        logical_name = str(logical).strip()
        if not key or not logical_name:
            continue
        asset_id = resolve_active_asset_id(env, logical_name, env_id=env_id, plan_id=plan_id)
        shaped = _SHAPE_LABEL_BY_ASSET_ID.get(str(asset_id or "").strip())
        if shaped:
            values[key] = shaped
            continue
        if not asset_id:
            values[key] = _humanize_label(logical_name, logical_name)
            continue
        usd_path = _resolve_usd_path_for_asset(asset_id)
        if not usd_path:
            values[key] = _humanize_label(asset_id, logical_name)
            continue
        try:
            meta = load_asset_metadata(usd_path)
            values[key] = taxonomy_kind(meta, fallback=_humanize_label(asset_id, logical_name))
        except Exception:  # noqa: BLE001 — missing metadata.json is common for pool assets
            values[key] = _humanize_label(asset_id, logical_name)
    return values


def _render_template(template: str, values: Mapping[str, str]) -> str:
    text = str(template or "")
    if not text:
        return ""

    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return ""

    rendered = text.format_map(_SafeDict(values))
    while "  " in rendered:
        rendered = rendered.replace("  ", " ")
    return rendered.replace(" ,", ",").strip()


def apply_runtime_instruction_for_envs(
    env: Any,
    env_ids: list[int],
    *,
    task_payload: Mapping[str, Any],
    runtime_instruction: Mapping[str, Any] | None = None,
    plan_id: str | None = None,
    contexts_by_env: Mapping[int, Mapping[str, Any]] | None = None,
) -> None:
    """Render task instruction from per-env pool asset metadata and write active instruction."""
    spec = (
        dict(runtime_instruction)
        if isinstance(runtime_instruction, Mapping)
        else _definition_runtime_instruction(task_payload)
    )
    if not spec:
        return

    template = str(spec.get("template") or spec.get("instruction") or "").strip()
    if not template:
        return

    objects_raw = spec.get("objects") if isinstance(spec.get("objects"), Mapping) else {}
    objects = {str(k): str(v) for k, v in objects_raw.items() if str(k).strip() and str(v).strip()}
    if not objects:
        return

    plan_key = str(plan_id or spec.get("plan_id") or "runtime").strip() or "runtime"
    num_envs = int(getattr(env, "num_envs", max(env_ids) + 1 if env_ids else 1))

    from manaenv.pipeline.runtime.active_instruction import (
        write_active_instruction,
        write_active_instruction_placeholders,
        write_active_subtasks,
    )

    subtask_templates = [str(item).strip() for item in list(spec.get("subtasks") or []) if str(item).strip()]

    for env_id in env_ids:
        try:
            values = _placeholder_values_for_env(
                env,
                env_id=int(env_id),
                plan_id=plan_key,
                objects=objects,
            )
            _merge_runtime_instruction_placeholders(
                values,
                contexts_by_env,
                env=env,
                env_id=int(env_id),
                plan_id=plan_key,
                objects=objects,
            )
            write_active_instruction_placeholders(env, int(env_id), values, num_envs=num_envs)
            rendered = _render_template(template, values)
            if rendered:
                write_active_instruction(env, int(env_id), rendered, num_envs=num_envs, reset_owned=True)
            if subtask_templates:
                write_active_subtasks(
                    env,
                    int(env_id),
                    [_render_template(item, values) for item in subtask_templates],
                    num_envs=num_envs,
                    reset_owned=True,
                )
        except Exception as exc:  # noqa: BLE001 — degrade to static instruction
            manaprint(
                f"WARNING: [runtime_instruction] env={env_id} failed to render metadata instruction: "
                f"{type(exc).__name__}: {exc}"
            )


def _merge_runtime_instruction_placeholders(
    values: dict[str, str],
    contexts_by_env: Mapping[int, Mapping[str, Any]] | None,
    *,
    env: Any,
    env_id: int,
    plan_id: str,
    objects: Mapping[str, str],
) -> None:
    if not isinstance(contexts_by_env, Mapping):
        return
    context = contexts_by_env.get(int(env_id))
    if not isinstance(context, Mapping):
        return

    outputs = context.get("outputs")
    if isinstance(outputs, Mapping):
        place_beside = outputs.get("place_beside")
        if isinstance(place_beside, Mapping):
            phrase = str(place_beside.get("phrase") or "").strip()
            if phrase:
                values.setdefault("direction", phrase)
            target = str(place_beside.get("target") or "").strip()
            if target:
                values.setdefault("direction_id", target)

    selections = context.get("selections") if isinstance(context.get("selections"), Mapping) else {}
    materials = selections.get("materials") if isinstance(selections, Mapping) else None
    if not isinstance(materials, Mapping):
        applied = context.get("applied") if isinstance(context.get("applied"), Mapping) else {}
        materials = applied.get("materials") if isinstance(applied, Mapping) else None
    if not isinstance(materials, Mapping):
        return

    colors: dict[str, str] = {}
    for entry in materials.values():
        if not isinstance(entry, Mapping):
            continue
        color = str(entry.get("candidate_id") or "").strip().lower()
        if not color:
            candidate = entry.get("candidate") if isinstance(entry.get("candidate"), Mapping) else {}
            color = str(candidate.get("id") or "").strip().lower()
        if not color:
            continue
        target = entry.get("target") if isinstance(entry.get("target"), Mapping) else {}
        for name in list(target.get("objects") or []):
            key = str(name or "").strip()
            if key:
                colors[key] = color

    for placeholder, logical in objects.items():
        key = str(placeholder).strip()
        logical_name = str(logical).strip()
        if not key or not logical_name:
            continue
        asset_id = str(resolve_active_asset_id(env, logical_name, env_id=env_id, plan_id=plan_id) or "").strip()
        slot = str(resolve_active_scene_entity_name(env, logical_name, env_id=env_id, plan_id=plan_id) or "").strip()
        color = colors.get(slot) or colors.get(asset_id) or colors.get(logical_name)
        if not color and asset_id:
            letter = asset_id.rsplit("_", 1)[-1]
            if letter in {"a", "b", "c", "d"}:
                color = colors.get(f"nut_{letter}") or colors.get(f"screw_{letter}")
        if color:
            values[f"{key}_color"] = color
            values.setdefault("color", color)


__all__ = [
    "apply_runtime_instruction_for_envs",
]
