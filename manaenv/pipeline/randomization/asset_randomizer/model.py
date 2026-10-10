from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class RandomizationContext:
    seed: int
    scene_id: str | None = None
    episode_index: int | None = None
    variant_index: int | None = None
    namespace: str = "asset_randomizer"


@dataclass(frozen=True)
class MaterialSlotSelection:
    mode: str = "auto"
    slots: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "slots": list(self.slots),
        }


@dataclass(frozen=True)
class TargetQuery:
    asset_names: tuple[str, ...] = ()
    prim_paths: tuple[str, ...] = ()
    semantic_tags: tuple[str, ...] = ()
    material_slots: tuple[str, ...] = ()
    material_slot_mode: str = "auto"

    def is_empty(self) -> bool:
        return not (self.asset_names or self.prim_paths or self.semantic_tags or self.material_slots)

    @property
    def material_slot_selection(self) -> MaterialSlotSelection:
        return MaterialSlotSelection(mode=self.material_slot_mode, slots=self.material_slots)


@dataclass(frozen=True)
class AssetHandle:
    asset_id: str
    prim_path: str
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class MaterialSlot:
    slot_id: str
    asset_id: str
    material_path: str
    bound_prim_path: str
    shader_model: str
    supported_channels: frozenset[str]
    is_shared: bool = False
    channel_values: Mapping[str, Any] = field(default_factory=dict, compare=False)
    channel_sources: Mapping[str, str] = field(default_factory=dict, compare=False)
    channel_states: Mapping[str, MaterialChannelState] = field(default_factory=dict, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class MaterialTextureRef:
    input_name: str
    asset_path: str | None = None
    resolved_path: str | None = None
    connected_source: str | None = None
    source_kind: str = "direct"
    source_input_name: str | None = None
    color_space: str | None = None
    packed_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterialChannelState:
    channel: str
    source_type: str
    value: Any = None
    constant_input_name: str | None = None
    texture_input_names: tuple[str, ...] = ()
    texture_refs: tuple[MaterialTextureRef, ...] = ()
    writable_ops: frozenset[str] = field(default_factory=frozenset)
    planned_ops: frozenset[str] = field(default_factory=frozenset)
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaterialAssetRef:
    material_id: str
    shader_model: str | None = None
    usd_material_ref: str | None = None
    tags: tuple[str, ...] = ()
    channels: Mapping[str, Any] = field(default_factory=dict, compare=False)
    compatibility: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class MaterialEdit:
    target_slot_id: str
    channel: str
    source_type: str
    operation: str
    value: Any
    policy_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def op(self) -> str:
        return self.operation

    def with_metadata(self, extra: Mapping[str, Any]) -> MaterialEdit:
        merged = dict(self.metadata)
        merged.update(extra)
        return replace(self, metadata=merged)


@dataclass(frozen=True)
class RandomizationDiagnostic:
    severity: str
    code: str
    message: str
    asset_id: str | None = None
    material_path: str | None = None
    rule_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class PolicySample:
    edits: tuple[MaterialEdit, ...] = ()
    diagnostics: tuple[RandomizationDiagnostic, ...] = ()


@dataclass(frozen=True)
class ApplyOptions:
    clone_shared_materials: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "clone_shared_materials": self.clone_shared_materials,
        }


@dataclass(frozen=True)
class RandomizationRule:
    name: str
    targets: TargetQuery
    policy: Mapping[str, Any]
    apply: ApplyOptions = field(default_factory=ApplyOptions)


@dataclass(frozen=True)
class RandomizationSpec:
    enabled: bool = True
    seed_scope: str = "episode"
    rules: tuple[RandomizationRule, ...] = ()
    schema_version: int = 1


@dataclass(frozen=True)
class PlanStep:
    rule_name: str
    asset: AssetHandle
    edits: tuple[MaterialEdit, ...]
    diagnostics: tuple[RandomizationDiagnostic, ...] = ()


@dataclass(frozen=True)
class RandomizationPlan:
    context: RandomizationContext
    steps: tuple[PlanStep, ...] = ()
    diagnostics: tuple[RandomizationDiagnostic, ...] = ()
    schema_version: int = 1

    @property
    def edits(self) -> tuple[MaterialEdit, ...]:
        return tuple(edit for step in self.steps for edit in step.edits)


@dataclass(frozen=True)
class ApplyReport:
    applied_edits: tuple[MaterialEdit, ...] = ()
    skipped_edits: tuple[MaterialEdit, ...] = ()
    diagnostics: tuple[RandomizationDiagnostic, ...] = ()


@dataclass(frozen=True)
class RandomizationReport:
    plan: RandomizationPlan
    apply_report: ApplyReport
