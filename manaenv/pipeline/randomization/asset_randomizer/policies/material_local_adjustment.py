from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..materials.adjustments import MaterialLocalAdjustment, MaterialLocalAdjustmentLibrary
from ..materials.channels import is_slotwise_safe_operation
from ..materials.library import RandomizationLibraryInput, adjustment_library_from_input
from ..model import AssetHandle, MaterialEdit, MaterialSlot, PolicySample, RandomizationDiagnostic
from ..rng import RandomStream
from .channel_matrix import ChannelMatrixPolicy


@dataclass(frozen=True)
class _AdjustmentOperationConfig:
    channel: str
    source_type: str
    config: Mapping[str, Any]


@dataclass(frozen=True)
class MaterialLocalAdjustmentPolicy:
    policy_id: str
    library_id: str
    adjustment_ids: tuple[str, ...] = ()
    adjustment_tags_any: tuple[str, ...] = ()
    adjustment_tags_all: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MaterialLocalAdjustmentPolicy:
        adjustment_ids = _string_tuple(
            config.get("adjustment_ids")
            or config.get("adjustment_id")
            or config.get("local_adjustment_ids")
            or config.get("local_adjustment_id")
        )
        return cls(
            policy_id=str(config.get("id") or config.get("name") or "material_local_adjustment"),
            library_id=str(config.get("library") or config.get("library_id") or ""),
            adjustment_ids=adjustment_ids,
            adjustment_tags_any=_string_tuple(config.get("adjustment_tags_any") or config.get("tags_any")),
            adjustment_tags_all=_string_tuple(config.get("adjustment_tags_all") or config.get("tags_all")),
        )

    def sample(
        self,
        *,
        asset: AssetHandle,
        slots: tuple[MaterialSlot, ...],
        rng: RandomStream,
        library: RandomizationLibraryInput | None = None,
    ) -> PolicySample:
        adjustment_library = adjustment_library_from_input(library)
        if adjustment_library is None:
            return PolicySample(
                diagnostics=(
                    self._diagnostic(
                        "warning",
                        "missing_material_local_adjustment_library",
                        "material_local_adjustment requires an adjustment library",
                        asset=asset,
                        metadata={"library_id": self.library_id},
                    ),
                )
            )
        if self.library_id and adjustment_library.library_id != self.library_id:
            return PolicySample(
                diagnostics=(
                    self._diagnostic(
                        "warning",
                        "material_local_adjustment_library_mismatch",
                        f"policy requested library {self.library_id}, got {adjustment_library.library_id}",
                        asset=asset,
                        metadata={
                            "expected_library_id": self.library_id,
                            "actual_library_id": adjustment_library.library_id,
                        },
                    ),
                )
            )
        candidates = self._select_candidates(adjustment_library)
        if not candidates:
            return PolicySample(
                diagnostics=(
                    self._diagnostic(
                        "warning",
                        "empty_material_local_adjustment_selection",
                        "no material local adjustment matched policy selection",
                        asset=asset,
                        metadata={
                            "library_id": adjustment_library.library_id,
                            "adjustment_ids": self.adjustment_ids,
                            "adjustment_tags_any": self.adjustment_tags_any,
                            "adjustment_tags_all": self.adjustment_tags_all,
                        },
                    ),
                )
            )

        adjustment = rng.choice(candidates)
        operation_configs, diagnostics = self._operation_configs(asset=asset, adjustment=adjustment, rng=rng)
        if diagnostics:
            return PolicySample(diagnostics=tuple(diagnostics))

        edits: list[MaterialEdit] = []
        sample_diagnostics: list[RandomizationDiagnostic] = []
        for index, operation_config in enumerate(operation_configs):
            policy = ChannelMatrixPolicy(
                policy_id=self.policy_id,
                channels={
                    operation_config.channel: {
                        operation_config.source_type: operation_config.config,
                    }
                },
            )
            sample = policy.sample(
                asset=asset,
                slots=slots,
                rng=rng.spawn(self.policy_id, adjustment.adjustment_id, index),
            )
            edits.extend(self._with_adjustment_metadata(edit, adjustment) for edit in sample.edits)
            sample_diagnostics.extend(
                self._diagnostic_with_adjustment_metadata(diagnostic, adjustment) for diagnostic in sample.diagnostics
            )
        return PolicySample(edits=tuple(edits), diagnostics=tuple(sample_diagnostics))

    def _select_candidates(self, library: MaterialLocalAdjustmentLibrary) -> tuple[MaterialLocalAdjustment, ...]:
        candidates = library.filter_by_ids(self.adjustment_ids)
        if self.adjustment_tags_all:
            allowed_ids = {
                adjustment.adjustment_id for adjustment in library.filter_by_tags_all(self.adjustment_tags_all)
            }
            candidates = tuple(adjustment for adjustment in candidates if adjustment.adjustment_id in allowed_ids)
        if self.adjustment_tags_any:
            allowed_ids = {
                adjustment.adjustment_id for adjustment in library.filter_by_tags_any(self.adjustment_tags_any)
            }
            candidates = tuple(adjustment for adjustment in candidates if adjustment.adjustment_id in allowed_ids)
        return candidates

    def _operation_configs(
        self,
        *,
        asset: AssetHandle,
        adjustment: MaterialLocalAdjustment,
        rng: RandomStream,
    ) -> tuple[tuple[_AdjustmentOperationConfig, ...], tuple[RandomizationDiagnostic, ...]]:
        diagnostics: list[RandomizationDiagnostic] = []
        if not adjustment.identity_preserving:
            diagnostics.append(
                self._unsafe_diagnostic(
                    asset=asset,
                    adjustment=adjustment,
                    message="material local adjustment must preserve material identity",
                    metadata={"identity_preserving": adjustment.identity_preserving},
                )
            )
        if not adjustment.slotwise_safe:
            diagnostics.append(
                self._unsafe_diagnostic(
                    asset=asset,
                    adjustment=adjustment,
                    message="material local adjustment must be slotwise-safe in V1",
                    metadata={"slotwise_safe": adjustment.slotwise_safe},
                )
            )
        operation_configs = tuple(
            _expand_operation_configs(
                channels=adjustment.channels,
                rng=rng.spawn(self.policy_id, adjustment.adjustment_id, "recipe"),
            )
        )
        if not operation_configs:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "empty_material_local_adjustment_recipe",
                    "material local adjustment defines no channel operations",
                    asset=asset,
                    metadata=self._adjustment_metadata(adjustment),
                )
            )
        for operation_config in operation_configs:
            op = str(operation_config.config.get("op") or "").strip()
            if not op:
                diagnostics.append(
                    self._diagnostic(
                        "warning",
                        "invalid_material_local_adjustment_recipe",
                        "material local adjustment operation requires op",
                        asset=asset,
                        metadata={
                            **self._adjustment_metadata(adjustment),
                            "channel": operation_config.channel,
                            "source_type": operation_config.source_type,
                        },
                    )
                )
                continue
            if not is_slotwise_safe_operation(operation_config.source_type, op):
                diagnostics.append(
                    self._unsafe_diagnostic(
                        asset=asset,
                        adjustment=adjustment,
                        message=f"operation {op} is not allowed in material_local_adjustment V1",
                        metadata={
                            "channel": operation_config.channel,
                            "source_type": operation_config.source_type,
                            "operation": op,
                        },
                    )
                )
        return operation_configs, tuple(diagnostics)

    def _unsafe_diagnostic(
        self,
        *,
        asset: AssetHandle,
        adjustment: MaterialLocalAdjustment,
        message: str,
        metadata: Mapping[str, Any],
    ) -> RandomizationDiagnostic:
        return self._diagnostic(
            "warning",
            "unsafe_material_local_adjustment",
            message,
            asset=asset,
            metadata={**self._adjustment_metadata(adjustment), **dict(metadata)},
        )

    def _with_adjustment_metadata(
        self,
        edit: MaterialEdit,
        adjustment: MaterialLocalAdjustment,
    ) -> MaterialEdit:
        return edit.with_metadata(self._adjustment_metadata(adjustment))

    def _diagnostic_with_adjustment_metadata(
        self,
        diagnostic: RandomizationDiagnostic,
        adjustment: MaterialLocalAdjustment,
    ) -> RandomizationDiagnostic:
        return RandomizationDiagnostic(
            severity=diagnostic.severity,
            code=diagnostic.code,
            message=diagnostic.message,
            asset_id=diagnostic.asset_id,
            material_path=diagnostic.material_path,
            rule_name=diagnostic.rule_name,
            metadata={
                **dict(diagnostic.metadata),
                **self._adjustment_metadata(adjustment),
            },
        )

    @staticmethod
    def _adjustment_metadata(adjustment: MaterialLocalAdjustment) -> dict[str, Any]:
        metadata = {
            "adjustment_id": adjustment.adjustment_id,
            "adjustment_tags": adjustment.tags,
            "identity_preserving": adjustment.identity_preserving,
            "slotwise_safe": adjustment.slotwise_safe,
        }
        if adjustment.family:
            metadata["adjustment_family"] = adjustment.family
        return metadata

    @staticmethod
    def _diagnostic(
        severity: str,
        code: str,
        message: str,
        *,
        asset: AssetHandle,
        metadata: Mapping[str, Any] | None = None,
    ) -> RandomizationDiagnostic:
        return RandomizationDiagnostic(
            severity=severity,
            code=code,
            message=message,
            asset_id=asset.asset_id,
            metadata=dict(metadata or {}),
        )


def _expand_operation_configs(
    *,
    channels: Mapping[str, Any],
    rng: RandomStream,
) -> tuple[_AdjustmentOperationConfig, ...]:
    operations: list[_AdjustmentOperationConfig] = []
    for channel, channel_config in channels.items():
        if not isinstance(channel_config, Mapping):
            operations.append(_AdjustmentOperationConfig(str(channel), "", {}))
            continue
        for source_type, source_config in channel_config.items():
            if not isinstance(source_config, Mapping):
                operations.append(_AdjustmentOperationConfig(str(channel), str(source_type), {}))
                continue
            op_items = source_config.get("ops")
            if isinstance(op_items, Sequence) and not isinstance(op_items, (str, bytes, bytearray)):
                defaults = {key: value for key, value in source_config.items() if key != "ops"}
                for index, item in enumerate(op_items):
                    config = dict(defaults)
                    if isinstance(item, Mapping):
                        config.update(item)
                    operations.append(
                        _AdjustmentOperationConfig(
                            str(channel),
                            str(source_type),
                            _sample_range_fields(config, rng=rng.spawn(channel, source_type, index)),
                        )
                    )
                continue
            operations.append(
                _AdjustmentOperationConfig(
                    str(channel),
                    str(source_type),
                    _sample_range_fields(source_config, rng=rng.spawn(channel, source_type)),
                )
            )
    return tuple(operations)


def _sample_range_fields(config: Mapping[str, Any], *, rng: RandomStream) -> dict[str, Any]:
    result = dict(config)
    if "value" not in result and "value_range" in result:
        result["value"] = _sample_range(result["value_range"], rng=rng.spawn("value_range"))
    return result


def _sample_range(value: Any, *, rng: RandomStream) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 2:
        low, high = value[0], value[1]
        if _is_number(low) and _is_number(high):
            return rng.uniform(float(low), float(high))
        if _is_numeric_sequence(low) and _is_numeric_sequence(high):
            return tuple(rng.uniform(float(lo), float(hi)) for lo, hi in zip(low, high, strict=True))
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _is_numeric_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and all(_is_number(item) for item in value)
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)
