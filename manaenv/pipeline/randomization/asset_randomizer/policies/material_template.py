from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..materials.channels import ChannelSource, MaterialChannel, RandomizationOp, implemented_ops
from ..materials.library import MaterialAsset, MaterialLibrary, RandomizationLibraryInput, material_library_from_input
from ..materials.packed_orm import packed_orm_replace_payload
from ..materials.templates import MaterialChannelBinding, material_channel_bindings
from ..model import AssetHandle, MaterialEdit, MaterialSlot, PolicySample, RandomizationDiagnostic
from ..rng import RandomStream
from .value_ops import normalize_channel_value


@dataclass(frozen=True)
class MaterialTemplatePolicy:
    policy_id: str
    library_id: str
    template_ids: tuple[str, ...] = ()
    material_tags_any: tuple[str, ...] = ()
    material_tags_all: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> MaterialTemplatePolicy:
        template_ids = _string_tuple(
            config.get("template_ids")
            or config.get("material_ids")
            or config.get("template_id")
            or config.get("material_id")
        )
        return cls(
            policy_id=str(config.get("id") or config.get("name") or "material_template"),
            library_id=str(config.get("library") or config.get("library_id") or ""),
            template_ids=template_ids,
            material_tags_any=_string_tuple(config.get("template_tags_any") or config.get("material_tags_any")),
            material_tags_all=_string_tuple(config.get("template_tags_all") or config.get("material_tags_all")),
        )

    def sample(
        self,
        *,
        asset: AssetHandle,
        slots: tuple[MaterialSlot, ...],
        rng: RandomStream,
        library: RandomizationLibraryInput | None = None,
    ) -> PolicySample:
        material_library = material_library_from_input(library)
        if material_library is None:
            return PolicySample(
                diagnostics=(
                    self._diagnostic(
                        "warning",
                        "missing_material_library",
                        "material_template requires a material library",
                        asset=asset,
                        metadata={"library_id": self.library_id},
                    ),
                )
            )
        if self.library_id and material_library.library_id != self.library_id:
            return PolicySample(
                diagnostics=(
                    self._diagnostic(
                        "warning",
                        "material_library_mismatch",
                        f"policy requested library {self.library_id}, got {material_library.library_id}",
                        asset=asset,
                        metadata={
                            "expected_library_id": self.library_id,
                            "actual_library_id": material_library.library_id,
                        },
                    ),
                )
            )
        candidates = self._select_candidates(material_library)
        if not candidates:
            return PolicySample(
                diagnostics=(
                    self._diagnostic(
                        "warning",
                        "empty_material_template_selection",
                        "no material template matched policy selection",
                        asset=asset,
                        metadata={
                            "library_id": material_library.library_id,
                            "template_ids": self.template_ids,
                            "material_tags_any": self.material_tags_any,
                            "material_tags_all": self.material_tags_all,
                        },
                    ),
                )
            )

        material = rng.choice(candidates)
        edits: list[MaterialEdit] = []
        diagnostics: list[RandomizationDiagnostic] = []
        diagnostics.extend(self._material_reference_diagnostics(asset=asset, material=material))
        bindings = material_channel_bindings(material)
        for slot in slots:
            slot_diagnostic = self._slot_compatibility_diagnostic(asset=asset, slot=slot, material=material)
            if slot_diagnostic is not None:
                diagnostics.append(slot_diagnostic)
                continue
            packed_orm_conflict = self._packed_orm_template_conflict(
                asset=asset,
                slot=slot,
                material=material,
                bindings=bindings,
            )
            if packed_orm_conflict is not None:
                diagnostics.append(packed_orm_conflict)
                continue
            if _uv_required(material, bindings) and slot.metadata.get("has_uv_primvars") is False:
                diagnostics.append(
                    self._diagnostic(
                        "warning",
                        "uv_required_template_without_uv",
                        "material template requires UV primvars but the selected slot has no detected UVs",
                        asset=asset,
                        slot=slot,
                        metadata={"material_id": material.material_id},
                    )
                )
                continue
            slot_bindability_diagnostic = self._slot_bindability_diagnostic(
                asset=asset,
                slot=slot,
                material=material,
                bindings=bindings,
            )
            if slot_bindability_diagnostic is not None:
                diagnostics.append(slot_bindability_diagnostic)
                continue
            for binding in bindings:
                edit = self._compile_binding(
                    asset=asset,
                    slot=slot,
                    material=material,
                    binding=binding,
                    diagnostics=diagnostics,
                )
                if edit is not None:
                    edits.append(edit)
        return PolicySample(edits=tuple(edits), diagnostics=tuple(diagnostics))

    def _select_candidates(self, library: MaterialLibrary) -> tuple[MaterialAsset, ...]:
        candidates = library.filter_by_ids(self.template_ids)
        if self.material_tags_all:
            allowed_ids = {material.material_id for material in library.filter_by_tags_all(self.material_tags_all)}
            candidates = tuple(material for material in candidates if material.material_id in allowed_ids)
        if self.material_tags_any:
            allowed_ids = {material.material_id for material in library.filter_by_tags_any(self.material_tags_any)}
            candidates = tuple(material for material in candidates if material.material_id in allowed_ids)
        return candidates

    def _material_reference_diagnostics(
        self,
        *,
        asset: AssetHandle,
        material: MaterialAsset,
    ) -> list[RandomizationDiagnostic]:
        diagnostics: list[RandomizationDiagnostic] = []
        if material.usd_material_ref:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "usd_material_ref_not_directly_bound",
                    "material_template compiles channel bindings; direct USD material binding is not used",
                    asset=asset,
                    metadata={
                        "material_id": material.material_id,
                        "usd_material_ref": material.usd_material_ref,
                    },
                )
            )
        mdl_ref = _compatibility_value(material, "mdl_material_ref")
        if mdl_ref:
            diagnostics.append(
                self._diagnostic(
                    "info",
                    "mdl_material_ref_metadata_only",
                    "MDL material reference is recorded as metadata; V1 applies extracted channel bindings only",
                    asset=asset,
                    metadata={
                        "material_id": material.material_id,
                        "mdl_material_ref": mdl_ref,
                    },
                )
            )
        return diagnostics

    def _slot_compatibility_diagnostic(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
    ) -> RandomizationDiagnostic | None:
        supported = _supported_target_shader_families(material)
        if supported and slot.shader_model not in supported:
            return self._diagnostic(
                "warning",
                "unsupported_target_shader_family",
                "material template does not support the target slot shader family",
                asset=asset,
                slot=slot,
                metadata={
                    "material_id": material.material_id,
                    "target_shader_model": slot.shader_model,
                    "supported_target_shader_families": tuple(sorted(supported)),
                },
            )
        return None

    def _slot_bindability_diagnostic(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        bindings: tuple[MaterialChannelBinding, ...],
    ) -> RandomizationDiagnostic | None:
        if not bindings:
            return None
        if any(self._binding_can_target_slot(slot=slot, binding=binding) for binding in bindings):
            return None
        return self._diagnostic(
            "warning",
            "unsupported_template_target_slot_skipped",
            "target slot exposes no writable channel requested by the material template; slot skipped",
            asset=asset,
            slot=slot,
            metadata={
                "material_id": material.material_id,
                "requested_channels": tuple(binding.channel for binding in bindings),
                "target_shader_model": slot.shader_model,
                "target_supported_channels": tuple(sorted(slot.supported_channels)),
                "target_channel_sources": {
                    binding.channel: getattr(
                        slot.channel_states.get(binding.channel),
                        "source_type",
                        ChannelSource.MISSING,
                    )
                    for binding in bindings
                },
            },
        )

    def _binding_can_target_slot(
        self,
        *,
        slot: MaterialSlot,
        binding: MaterialChannelBinding,
    ) -> bool:
        if binding.source == ChannelSource.CONSTANT:
            return _constant_binding_can_target_slot(slot=slot, binding=binding)
        if binding.source == ChannelSource.TEXTURE:
            return _texture_binding_can_target_slot(slot=slot, binding=binding)
        if binding.source == ChannelSource.PACKED_TEXTURE:
            return _packed_texture_binding_can_target_slot(slot=slot, binding=binding)
        return False

    def _compile_binding(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        binding: MaterialChannelBinding,
        diagnostics: list[RandomizationDiagnostic],
    ) -> MaterialEdit | None:
        source = binding.source
        if source == ChannelSource.CONSTANT:
            return self._compile_constant_binding(
                asset=asset,
                slot=slot,
                material=material,
                binding=binding,
                diagnostics=diagnostics,
            )
        if source == ChannelSource.TEXTURE:
            return self._compile_texture_binding(
                asset=asset,
                slot=slot,
                material=material,
                binding=binding,
                diagnostics=diagnostics,
            )
        if source == ChannelSource.PACKED_TEXTURE:
            return self._compile_packed_texture_binding(
                asset=asset,
                slot=slot,
                material=material,
                binding=binding,
                diagnostics=diagnostics,
            )
        diagnostics.append(
            self._diagnostic(
                "warning",
                "unsupported_template_binding_source",
                f"unsupported material template binding source: {source}",
                asset=asset,
                slot=slot,
                channel=binding.channel,
                source_type=source,
                metadata={"material_id": material.material_id},
            )
        )
        return None

    def _compile_constant_binding(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        binding: MaterialChannelBinding,
        diagnostics: list[RandomizationDiagnostic],
    ) -> MaterialEdit | None:
        state = slot.channel_states.get(binding.channel)
        if state is None or state.source_type in {
            ChannelSource.MISSING,
            ChannelSource.UNSUPPORTED,
        }:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "unsupported_template_constant_channel",
                    "target slot does not expose a writable constant field for template binding",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=getattr(state, "source_type", ChannelSource.MISSING),
                    metadata={"material_id": material.material_id},
                )
            )
            return None
        if state.source_type == ChannelSource.PACKED_TEXTURE:
            diagnostics.append(
                self._packed_target_diagnostic(asset=asset, slot=slot, material=material, binding=binding)
            )
            return None
        try:
            value = normalize_channel_value(binding.channel, binding.value)
        except Exception as exc:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "invalid_template_constant_value",
                    f"invalid material template constant value for {binding.channel}: {exc}",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=ChannelSource.CONSTANT,
                    metadata={
                        "material_id": material.material_id,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            return None
        if state.source_type == ChannelSource.CONSTANT:
            return MaterialEdit(
                target_slot_id=slot.slot_id,
                channel=binding.channel,
                source_type=ChannelSource.CONSTANT,
                operation=RandomizationOp.SET_VALUE,
                value=value,
                policy_id=self.policy_id,
                metadata=self._edit_metadata(
                    material=material,
                    binding=binding,
                    desired_source=ChannelSource.CONSTANT,
                ),
            )
        if state.source_type == ChannelSource.TEXTURE and RandomizationOp.DISABLE_AND_SET_VALUE in implemented_ops(
            binding.channel, ChannelSource.TEXTURE
        ):
            return MaterialEdit(
                target_slot_id=slot.slot_id,
                channel=binding.channel,
                source_type=ChannelSource.TEXTURE,
                operation=RandomizationOp.DISABLE_AND_SET_VALUE,
                value=value,
                policy_id=self.policy_id,
                metadata=self._edit_metadata(
                    material=material,
                    binding=binding,
                    desired_source=ChannelSource.CONSTANT,
                ),
            )
        diagnostics.append(
            self._diagnostic(
                "warning",
                "unsupported_template_constant_channel",
                "target channel cannot be converted to the requested constant template binding",
                asset=asset,
                slot=slot,
                channel=binding.channel,
                source_type=state.source_type,
                metadata={"material_id": material.material_id},
            )
        )
        return None

    def _compile_texture_binding(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        binding: MaterialChannelBinding,
        diagnostics: list[RandomizationDiagnostic],
    ) -> MaterialEdit | None:
        state = slot.channel_states.get(binding.channel)
        if state is None or not state.texture_input_names:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "unsupported_template_texture_channel",
                    "target slot does not expose an official texture field for template binding",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=getattr(state, "source_type", ChannelSource.UNSUPPORTED),
                    metadata={"material_id": material.material_id, "uri": binding.uri},
                )
            )
            return None
        if state.source_type == ChannelSource.PACKED_TEXTURE:
            diagnostics.append(
                self._packed_target_diagnostic(asset=asset, slot=slot, material=material, binding=binding)
            )
            return None
        if not binding.uri:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "missing_template_texture_uri",
                    "texture material template binding requires a texture URI",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=ChannelSource.TEXTURE,
                    metadata={"material_id": material.material_id},
                )
            )
            return None
        local_path = binding.local_texture_path
        if local_path is not None and not local_path.is_file():
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "missing_template_texture_file",
                    "local material template texture file does not exist",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=ChannelSource.TEXTURE,
                    metadata={"material_id": material.material_id, "uri": binding.uri},
                )
            )
            return None
        return MaterialEdit(
            target_slot_id=slot.slot_id,
            channel=binding.channel,
            source_type=ChannelSource.TEXTURE,
            operation=RandomizationOp.REPLACE,
            value=binding.texture_payload,
            policy_id=self.policy_id,
            metadata={
                **self._edit_metadata(
                    material=material,
                    binding=binding,
                    desired_source=ChannelSource.TEXTURE,
                ),
                "allow_source_switch": True,
                "reset_texture_channel_defaults": True,
            },
        )

    def _compile_packed_texture_binding(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        binding: MaterialChannelBinding,
        diagnostics: list[RandomizationDiagnostic],
    ) -> MaterialEdit | None:
        if binding.channel != MaterialChannel.ORM:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "packed_texture_template_binding_requires_orm_channel",
                    "packed ORM template bindings must be declared on the orm channel",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=binding.source,
                    metadata={"material_id": material.material_id, "uri": binding.uri},
                )
            )
            return None
        state = slot.channel_states.get(MaterialChannel.ORM)
        if state is None or not state.texture_input_names:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "unsupported_template_packed_orm_channel",
                    "target slot does not expose an official packed ORM texture field",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=getattr(state, "source_type", ChannelSource.UNSUPPORTED),
                    metadata={"material_id": material.material_id, "uri": binding.uri},
                )
            )
            return None
        if not binding.uri:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "missing_template_packed_orm_uri",
                    "packed ORM material template binding requires a texture URI",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=ChannelSource.PACKED_TEXTURE,
                    metadata={"material_id": material.material_id},
                )
            )
            return None
        local_path = binding.local_texture_path
        if local_path is not None and not local_path.is_file():
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "missing_template_packed_orm_file",
                    "local material template packed ORM file does not exist",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=ChannelSource.PACKED_TEXTURE,
                    metadata={"material_id": material.material_id, "uri": binding.uri},
                )
            )
            return None
        try:
            value = packed_orm_replace_payload({"uri": binding.uri, **dict(binding.metadata)})
        except Exception as exc:
            diagnostics.append(
                self._diagnostic(
                    "warning",
                    "invalid_template_packed_orm_binding",
                    f"invalid packed ORM template binding: {exc}",
                    asset=asset,
                    slot=slot,
                    channel=binding.channel,
                    source_type=ChannelSource.PACKED_TEXTURE,
                    metadata={
                        "material_id": material.material_id,
                        "error_type": type(exc).__name__,
                    },
                )
            )
            return None
        return MaterialEdit(
            target_slot_id=slot.slot_id,
            channel=MaterialChannel.ORM,
            source_type=ChannelSource.PACKED_TEXTURE,
            operation=RandomizationOp.REPLACE,
            value=value,
            policy_id=self.policy_id,
            metadata={
                **self._edit_metadata(
                    material=material,
                    binding=binding,
                    desired_source=ChannelSource.PACKED_TEXTURE,
                ),
                "allow_source_switch": True,
            },
        )

    def _packed_orm_template_conflict(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        bindings: tuple[MaterialChannelBinding, ...],
    ) -> RandomizationDiagnostic | None:
        has_packed_orm = any(
            binding.channel == MaterialChannel.ORM and binding.source == ChannelSource.PACKED_TEXTURE
            for binding in bindings
        )
        if not has_packed_orm:
            return None
        conflicting_channels = tuple(
            binding.channel
            for binding in bindings
            if binding.channel in {MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC}
        )
        if not conflicting_channels:
            return None
        return self._diagnostic(
            "warning",
            "packed_orm_template_channel_conflict",
            "material template cannot mix orm.packed_texture with independent roughness/metallic bindings",
            asset=asset,
            slot=slot,
            channel=MaterialChannel.ORM,
            source_type=ChannelSource.PACKED_TEXTURE,
            metadata={
                "material_id": material.material_id,
                "conflicting_channels": conflicting_channels,
            },
        )

    def _packed_target_diagnostic(
        self,
        *,
        asset: AssetHandle,
        slot: MaterialSlot,
        material: MaterialAsset,
        binding: MaterialChannelBinding,
    ) -> RandomizationDiagnostic:
        return self._diagnostic(
            "warning",
            "packed_texture_target_unsupported",
            "target channel is driven by packed ORM; use orm.packed_texture bindings or channel_matrix packed ORM operations",
            asset=asset,
            slot=slot,
            channel=binding.channel,
            source_type=ChannelSource.PACKED_TEXTURE,
            metadata={
                "material_id": material.material_id,
                "requested_source": binding.source,
                "uri": binding.uri,
            },
        )

    def _edit_metadata(
        self,
        *,
        material: MaterialAsset,
        binding: MaterialChannelBinding,
        desired_source: str,
    ) -> dict[str, Any]:
        metadata = {
            "material_id": material.material_id,
            "template_id": material.material_id,
            "template_tags": material.tags,
            "template_channel": binding.channel,
            "desired_source_binding": desired_source,
        }
        for key in ("family", "variant", "shader_family"):
            value = _compatibility_value(material, key)
            if value:
                metadata[f"template_{key}"] = value
        for key in ("uv_tile_size_m", "texture_scale"):
            value = _compatibility_value(material, key)
            if value is not None:
                metadata[key] = value
        if binding.uri:
            metadata["template_texture_uri"] = binding.uri
        return metadata

    def _diagnostic(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        asset: AssetHandle,
        slot: MaterialSlot | None = None,
        channel: str | None = None,
        source_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RandomizationDiagnostic:
        payload = dict(metadata or {})
        if channel is not None:
            payload["channel"] = channel
        if source_type is not None:
            payload["source_type"] = source_type
        return RandomizationDiagnostic(
            severity=severity,
            code=code,
            message=message,
            asset_id=asset.asset_id,
            material_path=slot.material_path if slot is not None else None,
            metadata=payload,
        )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item) for item in value)
    return (str(value),)


def _compatibility_value(material: MaterialAsset, key: str) -> Any:
    if key in material.compatibility:
        return material.compatibility[key]
    requirements = material.compatibility.get("requirements")
    if isinstance(requirements, Mapping) and key in requirements:
        return requirements[key]
    metadata = material.compatibility.get("metadata")
    if isinstance(metadata, Mapping) and key in metadata:
        return metadata[key]
    return None


def _supported_target_shader_families(material: MaterialAsset) -> set[str]:
    value = (
        _compatibility_value(material, "supported_target_shader_families")
        or _compatibility_value(material, "supported_target_shader_models")
        or ()
    )
    return set(_string_tuple(value))


def _uv_required(material: MaterialAsset, bindings: tuple[MaterialChannelBinding, ...]) -> bool:
    value = _compatibility_value(material, "uv_required")
    if value is not None:
        return bool(value)
    return any(binding.source in {ChannelSource.TEXTURE, ChannelSource.PACKED_TEXTURE} for binding in bindings)


def _constant_binding_can_target_slot(*, slot: MaterialSlot, binding: MaterialChannelBinding) -> bool:
    state = slot.channel_states.get(binding.channel)
    if state is None or state.source_type in {
        ChannelSource.MISSING,
        ChannelSource.UNSUPPORTED,
        ChannelSource.PACKED_TEXTURE,
    }:
        return False
    if state.source_type == ChannelSource.CONSTANT:
        return True
    return state.source_type == ChannelSource.TEXTURE and RandomizationOp.DISABLE_AND_SET_VALUE in implemented_ops(
        binding.channel,
        ChannelSource.TEXTURE,
    )


def _texture_binding_can_target_slot(*, slot: MaterialSlot, binding: MaterialChannelBinding) -> bool:
    state = slot.channel_states.get(binding.channel)
    if state is None or state.source_type == ChannelSource.PACKED_TEXTURE:
        return False
    return bool(state.texture_input_names)


def _packed_texture_binding_can_target_slot(*, slot: MaterialSlot, binding: MaterialChannelBinding) -> bool:
    if binding.channel != MaterialChannel.ORM:
        return False
    state = slot.channel_states.get(MaterialChannel.ORM)
    if state is None:
        return False
    return bool(state.texture_input_names)
