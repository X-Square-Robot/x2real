from __future__ import annotations

from collections.abc import Mapping

from ..backends.base import SceneBackend
from ..materials.channels import is_slotwise_safe_operation
from ..materials.library import RandomizationLibraryInput
from ..model import (
    AssetHandle,
    MaterialEdit,
    MaterialSlot,
    PlanStep,
    PolicySample,
    RandomizationContext,
    RandomizationDiagnostic,
    RandomizationPlan,
    RandomizationRule,
    RandomizationSpec,
)
from ..policies import create_policy
from ..rng import RandomStream

SLOTWISE_SAFE_BEHAVIOR = "slotwise_safe"
REQUIRES_EXPLICIT_SLOTS_BEHAVIOR = "requires_explicit_slots"


def compile_plan(
    *,
    backend: SceneBackend,
    spec: RandomizationSpec,
    context: RandomizationContext,
    library: RandomizationLibraryInput | None = None,
) -> RandomizationPlan:
    if not spec.enabled:
        return RandomizationPlan(context=context, schema_version=spec.schema_version)
    steps: list[PlanStep] = []
    diagnostics: list[RandomizationDiagnostic] = []
    for rule in spec.rules:
        assets = backend.resolve_targets(rule.targets)
        if not assets:
            diagnostics.append(
                RandomizationDiagnostic(
                    severity="warning",
                    code="empty_target_selection",
                    message=f"rule {rule.name} matched no assets",
                    rule_name=rule.name,
                )
            )
            continue
        for asset in assets:
            step = _compile_rule_for_asset(
                backend=backend,
                rule=rule,
                asset=asset,
                context=context,
                library=library,
            )
            steps.append(step)
    return RandomizationPlan(
        context=context,
        steps=tuple(steps),
        diagnostics=tuple(diagnostics),
        schema_version=spec.schema_version,
    )


def _compile_rule_for_asset(
    *,
    backend: SceneBackend,
    rule: RandomizationRule,
    asset: AssetHandle,
    context: RandomizationContext,
    library: RandomizationLibraryInput | None,
) -> PlanStep:
    policy = create_policy(rule.policy)
    inspected_slots = backend.inspect_material_slots(asset)
    slots, selection_diagnostics = _select_slots(inspected_slots, rule=rule, asset=asset)
    sample = (
        policy.sample(
            asset=asset,
            slots=slots,
            rng=RandomStream.from_context(context, rule.name, asset.asset_id),
            library=library,
        )
        if slots
        else PolicySample()
    )
    selection_metadata = _slot_selection_metadata(
        rule=rule,
        inspected_slots=inspected_slots,
        selected_slots=slots,
    )
    edits = tuple(_attach_apply_options(edit, rule, selection_metadata=selection_metadata) for edit in sample.edits)
    diagnostics = tuple(_with_rule_name(diag, rule.name) for diag in (*selection_diagnostics, *sample.diagnostics))
    return PlanStep(rule_name=rule.name, asset=asset, edits=edits, diagnostics=diagnostics)


def _select_slots(
    slots: tuple[MaterialSlot, ...],
    *,
    rule: RandomizationRule,
    asset: AssetHandle,
) -> tuple[tuple[MaterialSlot, ...], tuple[RandomizationDiagnostic, ...]]:
    if not slots:
        return (), (
            RandomizationDiagnostic(
                severity="warning",
                code="no_material_slots",
                message="target asset exposes no inspectable material slots",
                asset_id=asset.asset_id,
                metadata={
                    "material_slot_selection": rule.targets.material_slot_selection.to_json(),
                },
            ),
        )

    mode = rule.targets.material_slot_mode
    if mode == "all":
        return slots, ()
    if mode == "explicit":
        return _select_explicit_slots(slots, rule=rule, asset=asset)
    if mode != "auto":
        return (), (
            RandomizationDiagnostic(
                severity="warning",
                code="invalid_material_slot_selection_mode",
                message=f"invalid material slot selection mode: {mode}",
                asset_id=asset.asset_id,
                metadata={
                    "mode": mode,
                    "available_material_slots": _slot_summaries(slots),
                },
            ),
        )
    if len(slots) == 1:
        return slots, ()
    behavior = _rule_multi_slot_behavior(rule)
    if behavior == SLOTWISE_SAFE_BEHAVIOR:
        return slots, ()
    return (), (
        RandomizationDiagnostic(
            severity="warning",
            code="ambiguous_material_slot_selection",
            message=(
                "rule targets a multi-material asset with an operation that requires explicit material slot scope"
            ),
            asset_id=asset.asset_id,
            metadata={
                "material_slot_selection": rule.targets.material_slot_selection.to_json(),
                "multi_slot_behavior": behavior,
                "policy_type": str(rule.policy.get("type") or ""),
                "available_material_slots": _slot_summaries(slots),
                "required_modes": ("all", "explicit"),
            },
        ),
    )


def _select_explicit_slots(
    slots: tuple[MaterialSlot, ...],
    *,
    rule: RandomizationRule,
    asset: AssetHandle,
) -> tuple[tuple[MaterialSlot, ...], tuple[RandomizationDiagnostic, ...]]:
    allowed = set(rule.targets.material_slots)
    selected = tuple(slot for slot in slots if any(_slot_matches(slot, item) for item in allowed))
    matched = {item for item in allowed if any(_slot_matches(slot, item) for slot in selected)}
    unknown = tuple(sorted(item for item in allowed if item not in matched))
    diagnostics: list[RandomizationDiagnostic] = []
    if unknown:
        diagnostics.append(
            RandomizationDiagnostic(
                severity="warning",
                code="unknown_material_slot_selection",
                message="material slot selection contains slots that were not found on the target asset",
                asset_id=asset.asset_id,
                metadata={
                    "unknown_material_slots": unknown,
                    "available_material_slots": _slot_summaries(slots),
                },
            )
        )
    if not selected:
        diagnostics.append(
            RandomizationDiagnostic(
                severity="warning",
                code="material_slot_selection_empty",
                message="material slot selection matched no slots on the target asset",
                asset_id=asset.asset_id,
                metadata={
                    "requested_material_slots": tuple(sorted(allowed)),
                    "available_material_slots": _slot_summaries(slots),
                },
            )
        )
    return selected, tuple(diagnostics)


def _slot_matches(slot: MaterialSlot, requested: str) -> bool:
    value = str(requested or "").strip()
    if not value:
        return False
    if slot.slot_id == value or slot.material_path == value:
        return True
    normalized = value.strip("/")
    return slot.material_path.strip("/").endswith(normalized) or slot.slot_id.strip("/").endswith(normalized)


def _rule_multi_slot_behavior(rule: RandomizationRule) -> str:
    policy_type = str(rule.policy.get("type") or "")
    if policy_type == "material_local_adjustment":
        return SLOTWISE_SAFE_BEHAVIOR
    if policy_type != "channel_matrix":
        return REQUIRES_EXPLICIT_SLOTS_BEHAVIOR
    channels = rule.policy.get("channels")
    if not isinstance(channels, Mapping):
        return REQUIRES_EXPLICIT_SLOTS_BEHAVIOR
    for channel_config in channels.values():
        if not isinstance(channel_config, Mapping):
            return REQUIRES_EXPLICIT_SLOTS_BEHAVIOR
        for source_type, source_config in channel_config.items():
            if not isinstance(source_config, Mapping):
                return REQUIRES_EXPLICIT_SLOTS_BEHAVIOR
            op = str(source_config.get("op") or "").strip()
            if op and not is_slotwise_safe_operation(str(source_type), op):
                return REQUIRES_EXPLICIT_SLOTS_BEHAVIOR
    return SLOTWISE_SAFE_BEHAVIOR


def _slot_selection_metadata(
    *,
    rule: RandomizationRule,
    inspected_slots: tuple[MaterialSlot, ...],
    selected_slots: tuple[MaterialSlot, ...],
) -> dict[str, object]:
    return {
        "mode": rule.targets.material_slot_mode,
        "requested_slots": tuple(rule.targets.material_slots),
        "inspected_slot_count": len(inspected_slots),
        "selected_slot_count": len(selected_slots),
        "selected_slots": tuple(slot.slot_id for slot in selected_slots),
        "selected_material_paths": tuple(slot.material_path for slot in selected_slots),
    }


def _slot_summaries(slots: tuple[MaterialSlot, ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "slot_id": slot.slot_id,
            "material_path": slot.material_path,
            "shader_model": slot.shader_model,
            "bound_prim_path": slot.bound_prim_path,
            "channels": dict(sorted(slot.channel_sources.items())),
        }
        for slot in slots
    )


def _attach_apply_options(
    edit: MaterialEdit,
    rule: RandomizationRule,
    *,
    selection_metadata: dict[str, object],
) -> MaterialEdit:
    return edit.with_metadata(
        {
            "apply": rule.apply.to_json(),
            "rule_name": rule.name,
            "material_slot_selection": selection_metadata,
        }
    )


def _with_rule_name(diagnostic: RandomizationDiagnostic, rule_name: str) -> RandomizationDiagnostic:
    if diagnostic.rule_name:
        return diagnostic
    return RandomizationDiagnostic(
        severity=diagnostic.severity,
        code=diagnostic.code,
        message=diagnostic.message,
        asset_id=diagnostic.asset_id,
        material_path=diagnostic.material_path,
        rule_name=rule_name,
        metadata=diagnostic.metadata,
    )
