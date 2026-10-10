from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .model import ApplyReport, MaterialEdit, RandomizationDiagnostic, RandomizationPlan, RandomizationReport


def concise_plan_report(
    plan: RandomizationPlan,
    *,
    generated_texture_dir: str | None = None,
) -> dict[str, Any]:
    """Return a compact dry-run summary for a compiled randomization plan."""

    diagnostics = [diagnostic for step in plan.steps for diagnostic in step.diagnostics]
    diagnostics.extend(plan.diagnostics)
    edits = list(plan.edits)
    payload: dict[str, Any] = {
        "schema_version": "asset_randomizer_concise_report_v1",
        "mode": "dry_run",
        "context": {
            "seed": plan.context.seed,
            "scene_id": plan.context.scene_id,
            "episode_index": plan.context.episode_index,
            "variant_index": plan.context.variant_index,
            "namespace": plan.context.namespace,
        },
        "totals": {
            "rule_step_count": len(plan.steps),
            "edit_count": len(edits),
            "diagnostic_count": len(diagnostics),
        },
        "selected_material_templates": _unique_metadata_values(edits, "material_id"),
        "selected_local_adjustments": _unique_metadata_values(edits, "adjustment_id"),
        "generated_textures": _generated_texture_summary(
            edits=edits,
            generated_texture_dir=generated_texture_dir,
        ),
        "diagnostics": _diagnostic_summary(diagnostics),
        "steps": [_step_summary(step) for step in plan.steps],
    }
    if plan.diagnostics:
        payload["plan_diagnostics"] = [_diagnostic_payload(item) for item in plan.diagnostics]
    return payload


def concise_randomization_report(
    report: RandomizationReport,
    *,
    generated_texture_dir: str | None = None,
) -> dict[str, Any]:
    """Return a compact post-apply summary suitable for phase-1 ManaEnv logs."""

    payload = concise_plan_report(report.plan, generated_texture_dir=generated_texture_dir)
    payload["mode"] = "applied"
    payload["apply"] = _apply_summary(report.apply_report)
    payload["totals"] = {
        **payload["totals"],
        "applied_edit_count": len(report.apply_report.applied_edits),
        "skipped_edit_count": len(report.apply_report.skipped_edits),
        "apply_diagnostic_count": len(report.apply_report.diagnostics),
    }
    return payload


def _step_summary(step: Any) -> dict[str, Any]:
    return {
        "rule_name": step.rule_name,
        "asset": {
            "asset_id": step.asset.asset_id,
            "name": step.asset.name,
            "prim_path": step.asset.prim_path,
        },
        "edit_count": len(step.edits),
        "diagnostic_count": len(step.diagnostics),
        "edits": [_edit_summary(edit) for edit in step.edits],
        "diagnostics": [_diagnostic_payload(diagnostic) for diagnostic in step.diagnostics],
    }


def _edit_summary(edit: MaterialEdit) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "target_slot_id": edit.target_slot_id,
        "channel": edit.channel,
        "source_type": edit.source_type,
        "operation": edit.operation,
        "policy_id": edit.policy_id,
    }
    metadata = dict(edit.metadata or {})
    selected = {
        key: metadata[key]
        for key in (
            "rule_name",
            "profile_id",
            "material_id",
            "adjustment_id",
            "template_texture_uri",
        )
        if key in metadata
    }
    if selected:
        payload["selected"] = selected
    slot_selection = metadata.get("material_slot_selection")
    if isinstance(slot_selection, Mapping):
        payload["material_slot_selection"] = {
            "mode": slot_selection.get("mode"),
            "selected_slot_count": slot_selection.get("selected_slot_count"),
            "selected_slots": tuple(slot_selection.get("selected_slots") or ()),
        }
    texture_uri = _texture_uri(edit.value)
    if texture_uri:
        payload["texture_uri"] = texture_uri
    return payload


def _apply_summary(report: ApplyReport) -> dict[str, Any]:
    return {
        "applied_edit_count": len(report.applied_edits),
        "skipped_edit_count": len(report.skipped_edits),
        "diagnostics": _diagnostic_summary(report.diagnostics),
        "skipped_edits": [_edit_summary(edit) for edit in report.skipped_edits],
    }


def _diagnostic_summary(
    diagnostics: Sequence[RandomizationDiagnostic],
) -> dict[str, Any]:
    return {
        "total": len(diagnostics),
        "by_severity": dict(sorted(Counter(diagnostic.severity for diagnostic in diagnostics).items())),
        "by_code": dict(sorted(Counter(diagnostic.code for diagnostic in diagnostics).items())),
        "items": [_diagnostic_payload(diagnostic) for diagnostic in diagnostics],
    }


def _diagnostic_payload(diagnostic: RandomizationDiagnostic) -> dict[str, Any]:
    return {
        "severity": diagnostic.severity,
        "code": diagnostic.code,
        "message": diagnostic.message,
        "asset_id": diagnostic.asset_id,
        "material_path": diagnostic.material_path,
        "rule_name": diagnostic.rule_name,
        "metadata": dict(diagnostic.metadata or {}),
    }


def _generated_texture_summary(
    *,
    edits: Sequence[MaterialEdit],
    generated_texture_dir: str | None,
) -> dict[str, Any]:
    paths = tuple(dict.fromkeys(path for edit in edits for path in _generated_texture_paths(edit)))
    payload: dict[str, Any] = {
        "paths": paths,
        "count": len(paths),
    }
    if generated_texture_dir:
        payload["directory"] = generated_texture_dir
    return payload


def _generated_texture_paths(edit: MaterialEdit) -> tuple[str, ...]:
    values = [edit.value, edit.metadata]
    paths: list[str] = []
    for value in values:
        paths.extend(
            _find_string_values(
                value,
                keys={"generated_texture", "generated_texture_path", "output_path"},
            )
        )
    return tuple(path for path in paths if path)


def _find_string_values(value: Any, *, keys: set[str]) -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in keys and isinstance(item, str):
                found.append(item)
            found.extend(_find_string_values(item, keys=keys))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.extend(_find_string_values(item, keys=keys))
    return tuple(found)


def _texture_uri(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("uri") or value.get("path") or value.get("texture") or "").strip()
    return ""


def _unique_metadata_values(edits: Sequence[MaterialEdit], key: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(edit.metadata[key]) for edit in edits if key in edit.metadata and str(edit.metadata[key]).strip()
        )
    )
