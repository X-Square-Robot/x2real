from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..materials.channels import ChannelSource, MaterialChannel, RandomizationOp, implemented_ops, planned_ops
from ..materials.library import RandomizationLibraryInput
from ..materials.packed_orm import (
    packed_orm_disable_values_payload,
    packed_orm_replace_payload,
    packed_orm_set_channel_value_payload,
    packed_orm_transform_channel_payload,
)
from ..model import AssetHandle, MaterialEdit, MaterialSlot, PolicySample, RandomizationDiagnostic
from ..rng import RandomStream
from .value_ops import (
    normal_strength_scale_value,
    normalize_channel_value,
    normalize_scalar,
    sample_constant_value,
    texture_bias_value,
    texture_replace_value,
    texture_scale_value,
)


def _source_config(config: Mapping[str, Any], source_type: str) -> Mapping[str, Any] | None:
    item = config.get(source_type)
    return item if isinstance(item, Mapping) else None


@dataclass(frozen=True)
class ChannelMatrixPolicy:
    policy_id: str
    channels: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> ChannelMatrixPolicy:
        channels = config.get("channels")
        return cls(
            policy_id=str(config.get("id") or config.get("name") or "channel_matrix"),
            channels=channels if isinstance(channels, Mapping) else {},
        )

    def sample(
        self,
        *,
        asset: AssetHandle,
        slots: tuple[MaterialSlot, ...],
        rng: RandomStream,
        library: RandomizationLibraryInput | None = None,
    ) -> PolicySample:
        del library
        edits: list[MaterialEdit] = []
        diagnostics: list[RandomizationDiagnostic] = []
        for slot in slots:
            for channel, channel_config in self.channels.items():
                if not isinstance(channel_config, Mapping):
                    diagnostics.append(
                        self._diagnostic(
                            "warning",
                            "invalid_channel_config",
                            f"channel config must be a mapping: {channel}",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            metadata={"config_type": type(channel_config).__name__},
                        )
                    )
                    continue
                state = slot.channel_states.get(str(channel))
                if state is None or state.source_type in {
                    ChannelSource.MISSING,
                    ChannelSource.UNSUPPORTED,
                }:
                    diagnostics.append(
                        self._diagnostic(
                            "warning",
                            "unsupported_or_missing_channel",
                            f"slot does not expose channel {channel}",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            source_type=getattr(state, "source_type", ChannelSource.MISSING),
                        )
                    )
                    continue
                config = _source_config(channel_config, state.source_type)
                if config is None:
                    diagnostics.append(
                        self._diagnostic(
                            "warning",
                            "unhandled_channel_source",
                            f"no operation configured for {channel}.{state.source_type}",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            source_type=state.source_type,
                            metadata={"configured_sources": sorted(str(key) for key in channel_config.keys())},
                        )
                    )
                    continue
                op = str(config.get("op") or "").strip()
                if not op:
                    diagnostics.append(
                        self._diagnostic(
                            "warning",
                            "missing_operation",
                            f"operation is required for {channel}.{state.source_type}",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            source_type=state.source_type,
                        )
                    )
                    continue
                if op not in planned_ops(str(channel), state.source_type):
                    diagnostics.append(
                        self._diagnostic(
                            "warning",
                            "unsupported_operation",
                            f"operation {op} is not valid for {channel}.{state.source_type}",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            source_type=state.source_type,
                            operation=op,
                            metadata={
                                "planned_ops": sorted(planned_ops(str(channel), state.source_type)),
                                "implemented_ops": sorted(implemented_ops(str(channel), state.source_type)),
                            },
                        )
                    )
                    continue
                if op not in implemented_ops(str(channel), state.source_type):
                    diagnostics.append(
                        self._diagnostic(
                            "warning",
                            "unimplemented_operation",
                            f"operation {op} is planned but not implemented for {channel}.{state.source_type}",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            source_type=state.source_type,
                            operation=op,
                            metadata={
                                "operation_status": "planned",
                                "planned_ops": sorted(planned_ops(str(channel), state.source_type)),
                                "implemented_ops": sorted(implemented_ops(str(channel), state.source_type)),
                                "required_ops": sorted(implemented_ops(str(channel), state.source_type)),
                            },
                        )
                    )
                    continue
                edit = self._sample_edit(
                    slot=slot,
                    channel=str(channel),
                    source_type=state.source_type,
                    op=op,
                    config=config,
                    rng=rng.spawn(self.policy_id, slot.slot_id, channel, state.source_type),
                )
                if edit is None:
                    diagnostics.append(
                        self._diagnostic(
                            "info",
                            "explicit_channel_skip",
                            f"operation {op} intentionally leaves {channel}.{state.source_type} unchanged",
                            asset=asset,
                            slot=slot,
                            channel=str(channel),
                            source_type=state.source_type,
                            operation=op,
                            metadata={
                                "texture_input_names": tuple(state.texture_input_names),
                                "texture_refs": tuple(ref.input_name for ref in state.texture_refs),
                            },
                        )
                    )
                    continue
                edits.append(edit)
        return PolicySample(edits=tuple(edits), diagnostics=tuple(diagnostics))

    def _sample_edit(
        self,
        *,
        slot: MaterialSlot,
        channel: str,
        source_type: str,
        op: str,
        config: Mapping[str, Any],
        rng: RandomStream,
    ) -> MaterialEdit | None:
        if op == RandomizationOp.SKIP:
            return None
        if source_type == ChannelSource.CONSTANT:
            return self._sample_constant_edit(slot=slot, channel=channel, op=op, config=config, rng=rng)
        if source_type == ChannelSource.TEXTURE:
            return self._sample_texture_edit(slot=slot, channel=channel, op=op, config=config)
        if source_type == ChannelSource.PACKED_TEXTURE:
            return self._sample_packed_texture_edit(slot=slot, channel=channel, op=op, config=config)
        raise NotImplementedError(f"implemented operation has no sampler: {channel}.{source_type}.{op}")

    def _sample_constant_edit(
        self,
        *,
        slot: MaterialSlot,
        channel: str,
        op: str,
        config: Mapping[str, Any],
        rng: RandomStream,
    ) -> MaterialEdit:
        return MaterialEdit(
            target_slot_id=slot.slot_id,
            channel=channel,
            source_type=ChannelSource.CONSTANT,
            operation=RandomizationOp.SET_VALUE,
            value=sample_constant_value(slot=slot, channel=channel, op=op, config=config, rng=rng),
            policy_id=self.policy_id,
            metadata={"requested_operation": op},
        )

    def _sample_texture_edit(
        self,
        *,
        slot: MaterialSlot,
        channel: str,
        op: str,
        config: Mapping[str, Any],
    ) -> MaterialEdit:
        if op == RandomizationOp.REPLACE:
            value = texture_replace_value(config)
        elif op == RandomizationOp.SCALE:
            value = texture_scale_value(channel=channel, config=config)
        elif op == RandomizationOp.BIAS:
            value = texture_bias_value(channel=channel, config=config)
        elif op == RandomizationOp.DISABLE_AND_SET_VALUE:
            if "value" not in config and "color" not in config:
                raise ValueError(f"{channel}.{op} requires value")
            value = normalize_channel_value(channel, config.get("value", config.get("color")))
        elif op == RandomizationOp.SCALE_STRENGTH:
            value = normal_strength_scale_value(config)
        else:
            raise NotImplementedError(f"implemented texture operation has no sampler: {channel}.{op}")
        return MaterialEdit(
            target_slot_id=slot.slot_id,
            channel=channel,
            source_type=ChannelSource.TEXTURE,
            operation=op,
            value=value,
            policy_id=self.policy_id,
            metadata={"requested_operation": op},
        )

    def _sample_packed_texture_edit(
        self,
        *,
        slot: MaterialSlot,
        channel: str,
        op: str,
        config: Mapping[str, Any],
    ) -> MaterialEdit:
        if channel == MaterialChannel.ORM and op == RandomizationOp.REPLACE:
            value = packed_orm_replace_payload(config)
        elif (
            channel in {MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC} and op == RandomizationOp.SET_CHANNEL_VALUE
        ):
            if "value" not in config:
                raise ValueError(f"{channel}.{ChannelSource.PACKED_TEXTURE}.{op} requires value")
            value = packed_orm_set_channel_value_payload(
                channel=channel,
                value=normalize_scalar(config["value"]),
                layout=config.get("layout") or config.get("packing"),
            )
        elif (
            channel in {MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC} and op == RandomizationOp.TRANSFORM_CHANNEL
        ):
            transform = str(config.get("transform") or config.get("operation") or config.get("mode") or "").strip()
            if not transform:
                raise ValueError(f"{channel}.{ChannelSource.PACKED_TEXTURE}.{op} requires transform")
            transform_value = _packed_transform_config_value(config=config, transform=transform)
            value = packed_orm_transform_channel_payload(
                channel=channel,
                transform=transform,
                value=transform_value,
                layout=config.get("layout") or config.get("packing"),
            )
        elif channel == MaterialChannel.ORM and op == RandomizationOp.DISABLE_AND_SET_VALUES:
            value = packed_orm_disable_values_payload(config)
        else:
            raise NotImplementedError(f"implemented packed texture operation has no sampler: {channel}.{op}")
        return MaterialEdit(
            target_slot_id=slot.slot_id,
            channel=channel,
            source_type=ChannelSource.PACKED_TEXTURE,
            operation=op,
            value=value,
            policy_id=self.policy_id,
            metadata={"requested_operation": op},
        )

    @staticmethod
    def _diagnostic(
        severity: str,
        code: str,
        message: str,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        channel: str | None = None,
        source_type: str | None = None,
        operation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RandomizationDiagnostic:
        payload = dict(metadata or {})
        if channel is not None:
            payload["channel"] = channel
        if source_type is not None:
            payload["source_type"] = source_type
        if operation is not None:
            payload["operation"] = operation
        return RandomizationDiagnostic(
            severity=severity,
            code=code,
            message=message,
            asset_id=asset.asset_id,
            material_path=slot.material_path,
            metadata=payload,
        )


def _packed_transform_config_value(*, config: Mapping[str, Any], transform: str) -> Any:
    if transform == RandomizationOp.CLAMP:
        if "range" in config:
            return config["range"]
        return {"min": config.get("min", 0.0), "max": config.get("max", 1.0)}
    if transform == RandomizationOp.SCALE:
        return config.get("value", config.get("scale", 1.0))
    if transform == RandomizationOp.BIAS:
        return config.get("value", config.get("bias", 0.0))
    return config.get("value")
