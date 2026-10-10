from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...errors import BackendUnavailableError
from ...materials.channels import ChannelSource, MaterialChannel, RandomizationOp, implemented_ops, planned_ops
from ...materials.packed_orm import (
    normalize_packed_orm_binding,
    packed_orm_disable_values_payload,
    write_packed_orm_channel_transform_texture,
    write_packed_orm_channel_value_texture,
)
from ...model import (
    ApplyReport,
    AssetHandle,
    MaterialChannelState,
    MaterialEdit,
    MaterialSlot,
    MaterialTextureRef,
    RandomizationDiagnostic,
    TargetQuery,
)
from .apply import channel_value_type, infer_usd_value
from .oqc_bridge import inspect_material_inventory, path_within, semantic_tags_for_prim
from .shader_io import (
    canonical_shader_model,
    channel_input_names,
    channel_io,
    channel_texture_input_names,
    input_asset_path,
    input_connected_source_path,
    input_has_connected_source,
    input_resolved_asset_path,
    read_input_value,
)
from .texture_ops import DEFAULT_GENERATED_TEXTURE_DIR, write_value_transformed_texture


def _require_pxr() -> tuple[Any, Any]:
    try:
        from pxr import Sdf, UsdShade
    except ImportError as exc:
        raise BackendUnavailableError("pxr is required for UsdSceneBackend") from exc
    return UsdShade, Sdf


class UsdSceneBackend:
    def __init__(
        self,
        stage: Any,
        *,
        asset_usd_uri: str | None = None,
        generated_texture_dir: str | Path | None = None,
    ):
        self.stage = stage
        self.asset_usd_uri = asset_usd_uri
        self.generated_texture_dir = (
            Path(generated_texture_dir) if generated_texture_dir else DEFAULT_GENERATED_TEXTURE_DIR
        )
        self._slot_cache: dict[str, MaterialSlot] = {}
        self._clone_cache: dict[str, MaterialSlot] = {}

    def resolve_targets(self, query: TargetQuery) -> tuple[AssetHandle, ...]:
        _require_pxr()
        targets: list[AssetHandle] = []
        seen: set[str] = set()
        for path in query.prim_paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                handle = self._asset_handle_from_prim(prim)
                targets.append(handle)
                seen.add(handle.prim_path)
        if query.asset_names or query.semantic_tags:
            wanted_names = set(query.asset_names)
            wanted_tags = set(query.semantic_tags)
            for prim in self.stage.Traverse():
                path = str(prim.GetPath())
                if path in seen:
                    continue
                name_match = bool(wanted_names and prim.GetName() in wanted_names)
                tag_match = bool(wanted_tags and wanted_tags.intersection(self._semantic_tags(prim)))
                if name_match or tag_match:
                    handle = self._asset_handle_from_prim(prim)
                    targets.append(handle)
                    seen.add(handle.prim_path)
        return tuple(targets)

    def inspect_material_slots(self, asset: AssetHandle) -> tuple[MaterialSlot, ...]:
        UsdShade, _ = _require_pxr()
        asset_prim = self.stage.GetPrimAtPath(asset.prim_path)
        if not asset_prim or not asset_prim.IsValid():
            return ()
        inventory = inspect_material_inventory(
            stage=self.stage,
            asset_prim_path=asset.prim_path,
            asset_usd_uri=self._asset_usd_uri(),
        )
        consumer_map = self._material_consumers()
        slots: dict[str, MaterialSlot] = {}
        for material_path, record in inventory.material_records.items():
            material = UsdShade.Material(self.stage.GetPrimAtPath(material_path))
            if not material:
                continue
            shader = self._surface_shader(material)
            if not shader:
                continue
            if material_path in slots:
                continue
            input_names = tuple(shader_input.GetBaseName() for shader_input in shader.GetInputs())
            shader_id = shader.GetIdAttr().Get() or ""
            shader_model = canonical_shader_model(
                str(shader_id),
                input_names,
                effective_shader_type=getattr(record, "effective_shader_type", None),
            )
            inputs = channel_input_names(shader_model)
            texture_inputs = channel_texture_input_names(shader_model)
            material_mesh_paths = tuple(
                mesh_path
                for mesh_path, material_paths in inventory.mesh_to_material_paths.items()
                if material_path in material_paths
            )
            bound_prim_paths = (
                tuple(path for path in consumer_map.get(material_path, ()) if path_within(path, asset.prim_path))
                or material_mesh_paths
                or (inventory.object_root_path,)
            )
            is_shared = any(not path_within(path, asset.prim_path) for path in consumer_map.get(material_path, ()))
            slot_id = self._slot_id(asset.asset_id, material_path)
            channel_states = self._channel_states(
                shader=shader,
                constant_inputs=inputs,
                texture_inputs=texture_inputs,
                oqc_texture_refs=tuple(getattr(record, "texture_refs", ()) or ()),
            )
            slot = MaterialSlot(
                slot_id=slot_id,
                asset_id=asset.asset_id,
                material_path=material_path,
                bound_prim_path=bound_prim_paths[0],
                shader_model=shader_model,
                supported_channels=frozenset(
                    channel
                    for channel, state in channel_states.items()
                    if state.source_type not in {ChannelSource.MISSING, ChannelSource.UNSUPPORTED}
                ),
                is_shared=is_shared,
                channel_values={
                    channel: state.value for channel, state in channel_states.items() if state.value is not None
                },
                channel_sources={channel: state.source_type for channel, state in channel_states.items()},
                channel_states=channel_states,
                metadata={
                    "asset_prim_path": asset.prim_path,
                    "object_root_path": inventory.object_root_path,
                    "object_root_source": inventory.object_root_source,
                    "surface_mesh_paths": inventory.surface_mesh_paths,
                    "shader_path": str(shader.GetPath()),
                    "channel_inputs": {
                        key: self._first_existing_input_name(shader, value) for key, value in inputs.items() if value
                    },
                    "channel_texture_inputs": {
                        key: self._existing_input_names(shader, value) for key, value in texture_inputs.items() if value
                    },
                    "bound_prim_paths": bound_prim_paths,
                    "has_uv_primvars": self._has_uv_primvars(material_mesh_paths or bound_prim_paths),
                    "has_base_color_texture": channel_states.get(
                        MaterialChannel.BASE_COLOR,
                        MaterialChannelState(
                            channel=MaterialChannel.BASE_COLOR,
                            source_type=ChannelSource.MISSING,
                        ),
                    ).source_type
                    == ChannelSource.TEXTURE,
                    "oqc_material": {
                        "effective_shader_type": record.effective_shader_type,
                        "issue_codes": tuple(record.issue_codes),
                        "texture_input_names": tuple(ref.input_name for ref in record.texture_refs),
                        "bound_mesh_count": int(record.bound_mesh_count),
                    },
                },
            )
            slots[material_path] = slot
            self._slot_cache[slot.slot_id] = slot
        return tuple(slots.values())

    def apply_material_edits(self, edits: tuple[MaterialEdit, ...]) -> ApplyReport:
        UsdShade, _ = _require_pxr()
        applied: list[MaterialEdit] = []
        skipped: list[MaterialEdit] = []
        diagnostics: list[RandomizationDiagnostic] = []
        for edit in edits:
            slot = self._slot_cache.get(edit.target_slot_id)
            if slot is None:
                skipped.append(edit)
                diagnostics.append(
                    RandomizationDiagnostic(
                        severity="warning",
                        code="unknown_material_slot",
                        message=f"slot was not inspected before apply: {edit.target_slot_id}",
                        metadata={
                            "target_slot_id": edit.target_slot_id,
                            "channel": edit.channel,
                            "source_type": edit.source_type,
                            "operation": edit.operation,
                        },
                    )
                )
                continue
            apply_options = edit.metadata.get("apply", {})
            clone_shared = not isinstance(apply_options, dict) or bool(
                apply_options.get("clone_shared_materials", True)
            )
            if slot.is_shared and clone_shared:
                slot = self._clone_cache.get(slot.slot_id) or self._clone_material_for_slot(slot)
                self._clone_cache[edit.target_slot_id] = slot
            channel_state = slot.channel_states.get(edit.channel)
            if channel_state is None:
                channel_state = MaterialChannelState(channel=edit.channel, source_type=ChannelSource.UNSUPPORTED)
            if edit.source_type != channel_state.source_type:
                switched_state = self._source_switch_channel_state(slot=slot, channel_state=channel_state, edit=edit)
                if switched_state is not None:
                    channel_state = switched_state
                else:
                    skipped.append(edit)
                    diagnostics.append(
                        RandomizationDiagnostic(
                            severity="warning",
                            code="channel_source_mismatch",
                            message=(
                                f"edit expects {edit.channel}.{edit.source_type}, "
                                f"but inspected source is {channel_state.source_type}"
                            ),
                            asset_id=slot.asset_id,
                            material_path=slot.material_path,
                            metadata={
                                "channel": edit.channel,
                                "expected_source_type": edit.source_type,
                                "actual_source_type": channel_state.source_type,
                                "operation": edit.operation,
                            },
                        )
                    )
                    continue
            if edit.operation not in implemented_ops(edit.channel, edit.source_type):
                skipped.append(edit)
                diagnostics.append(
                    RandomizationDiagnostic(
                        severity="warning",
                        code="unimplemented_operation",
                        message=f"USD backend does not implement operation {edit.operation}",
                        asset_id=slot.asset_id,
                        material_path=slot.material_path,
                        metadata={
                            "channel": edit.channel,
                            "source_type": edit.source_type,
                            "operation": edit.operation,
                            "implemented_ops": sorted(implemented_ops(edit.channel, edit.source_type)),
                            "planned_ops": sorted(planned_ops(edit.channel, edit.source_type)),
                        },
                    )
                )
                continue
            shader_path = slot.metadata.get("shader_path")
            if not shader_path:
                skipped.append(edit)
                diagnostics.append(
                    RandomizationDiagnostic(
                        severity="warning",
                        code="missing_shader_path",
                        message=f"slot does not expose a shader path for {edit.channel} operation {edit.operation}",
                        asset_id=slot.asset_id,
                        material_path=slot.material_path,
                        metadata={
                            "channel": edit.channel,
                            "source_type": edit.source_type,
                            "operation": edit.operation,
                        },
                    )
                )
                continue
            shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(shader_path)))
            diagnostic = self._apply_edit_to_shader(slot=slot, shader=shader, channel_state=channel_state, edit=edit)
            if diagnostic is not None:
                skipped.append(edit)
                diagnostics.append(diagnostic)
                continue
            applied.append(edit)
        return ApplyReport(
            applied_edits=tuple(applied),
            skipped_edits=tuple(skipped),
            diagnostics=tuple(diagnostics),
        )

    def _source_switch_channel_state(
        self,
        *,
        slot: MaterialSlot,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> MaterialChannelState | None:
        if not edit.metadata.get("allow_source_switch"):
            return None
        if edit.operation != RandomizationOp.REPLACE:
            return None
        if channel_state.source_type not in {
            ChannelSource.CONSTANT,
            ChannelSource.MISSING,
        }:
            return None
        if edit.source_type == ChannelSource.TEXTURE:
            io = channel_io(slot.shader_model, edit.channel)
            if io is None or not io.texture_inputs:
                return None
            texture_input_names = channel_state.texture_input_names or tuple(io.texture_inputs)
            return replace(
                channel_state,
                source_type=ChannelSource.TEXTURE,
                texture_input_names=tuple(texture_input_names),
                writable_ops=implemented_ops(edit.channel, ChannelSource.TEXTURE),
                planned_ops=planned_ops(edit.channel, ChannelSource.TEXTURE),
            )
        if edit.source_type == ChannelSource.PACKED_TEXTURE and edit.channel == MaterialChannel.ORM:
            io = channel_io(slot.shader_model, MaterialChannel.ORM)
            if io is None or not io.texture_inputs:
                return None
            texture_input_names = channel_state.texture_input_names or tuple(io.texture_inputs)
            return replace(
                channel_state,
                source_type=ChannelSource.PACKED_TEXTURE,
                texture_input_names=tuple(texture_input_names),
                writable_ops=implemented_ops(MaterialChannel.ORM, ChannelSource.PACKED_TEXTURE),
                planned_ops=planned_ops(MaterialChannel.ORM, ChannelSource.PACKED_TEXTURE),
            )
        return None

    def _apply_edit_to_shader(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        try:
            if edit.source_type == ChannelSource.CONSTANT:
                return self._apply_constant_edit(
                    slot=slot,
                    shader=shader,
                    channel_state=channel_state,
                    edit=edit,
                )
            if edit.source_type == ChannelSource.TEXTURE:
                return self._apply_texture_edit(
                    slot=slot,
                    shader=shader,
                    channel_state=channel_state,
                    edit=edit,
                )
            if edit.source_type == ChannelSource.PACKED_TEXTURE:
                return self._apply_packed_texture_edit(
                    slot=slot,
                    shader=shader,
                    channel_state=channel_state,
                    edit=edit,
                )
        except Exception as exc:
            return self._diagnostic(
                "warning",
                "apply_operation_failed",
                f"failed to apply {edit.channel}.{edit.source_type}.{edit.operation}: {exc}",
                slot=slot,
                edit=edit,
                metadata={"error_type": type(exc).__name__},
            )
        return self._diagnostic(
            "warning",
            "unsupported_apply_source_type",
            f"USD backend cannot apply source type {edit.source_type}",
            slot=slot,
            edit=edit,
        )

    def _apply_constant_edit(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        input_name = channel_state.constant_input_name
        if not input_name:
            return self._missing_writable_input(slot=slot, channel_state=channel_state, edit=edit)
        # OmniPBR final color is typically constant * diffuse_tint. Assets like
        # nut/screw ship with diffuse_tint=(1,0,0); writing only
        # diffuse_color_constant then yields black for green/blue (and muted
        # yellow). Neutralize multipliers before SET_VALUE.
        io = channel_io(slot.shader_model, edit.channel)
        if io is not None:
            for value_name, value in io.disable_values.items():
                if value_name == input_name:
                    continue
                self._set_input_value(shader=shader, input_name=value_name, value=value)
        self._set_channel_value(shader=shader, channel=edit.channel, input_name=input_name, value=edit.value)
        return None

    def _apply_texture_edit(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        if edit.operation == RandomizationOp.REPLACE:
            return self._replace_texture(slot=slot, shader=shader, channel_state=channel_state, edit=edit)
        if edit.operation in {RandomizationOp.SCALE, RandomizationOp.BIAS}:
            return self._apply_texture_value_transform(
                slot=slot,
                shader=shader,
                channel_state=channel_state,
                edit=edit,
            )
        if edit.operation == RandomizationOp.DISABLE_AND_SET_VALUE:
            return self._disable_texture_and_set_value(
                slot=slot,
                shader=shader,
                channel_state=channel_state,
                edit=edit,
            )
        if edit.operation == RandomizationOp.SCALE_STRENGTH:
            input_name = channel_state.constant_input_name
            if not input_name:
                return self._missing_writable_input(slot=slot, channel_state=channel_state, edit=edit)
            self._set_input_value(shader=shader, input_name=input_name, value=float(edit.value))
            return None
        return self._diagnostic(
            "warning",
            "unimplemented_operation",
            f"USD backend does not implement operation {edit.operation}",
            slot=slot,
            edit=edit,
            metadata={
                "implemented_ops": sorted(implemented_ops(edit.channel, edit.source_type)),
                "planned_ops": sorted(planned_ops(edit.channel, edit.source_type)),
            },
        )

    def _apply_packed_texture_edit(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        if edit.channel == MaterialChannel.ORM and edit.operation == RandomizationOp.REPLACE:
            return self._replace_packed_orm_texture(
                slot=slot,
                shader=shader,
                channel_state=channel_state,
                edit=edit,
            )
        if edit.channel in {
            MaterialChannel.ROUGHNESS,
            MaterialChannel.METALLIC,
        } and edit.operation in {
            RandomizationOp.SET_CHANNEL_VALUE,
            RandomizationOp.TRANSFORM_CHANNEL,
        }:
            return self._apply_packed_orm_channel_transform(
                slot=slot,
                shader=shader,
                channel_state=channel_state,
                edit=edit,
            )
        if edit.channel == MaterialChannel.ORM and edit.operation == RandomizationOp.DISABLE_AND_SET_VALUES:
            return self._disable_packed_orm_and_set_values(
                slot=slot,
                shader=shader,
                channel_state=channel_state,
                edit=edit,
            )
        return self._diagnostic(
            "warning",
            "unimplemented_operation",
            f"USD backend does not implement operation {edit.operation}",
            slot=slot,
            edit=edit,
            metadata={
                "implemented_ops": sorted(implemented_ops(edit.channel, edit.source_type)),
                "planned_ops": sorted(planned_ops(edit.channel, edit.source_type)),
            },
        )

    def _replace_packed_orm_texture(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        binding = normalize_packed_orm_binding(edit.value)
        target_ref = self._target_texture_ref(channel_state)
        if target_ref is None and not channel_state.texture_input_names:
            return self._missing_writable_input(slot=slot, channel_state=channel_state, edit=edit)
        self._set_texture_asset(
            shader=shader,
            channel_state=channel_state,
            texture_uri=binding.uri,
            target_ref=target_ref,
        )
        self._enable_texture_binding(slot=slot, shader=shader, edit=edit)
        return None

    def _apply_packed_orm_channel_transform(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        source_path, target_ref = self._source_texture_path(channel_state)
        if source_path is None or target_ref is None:
            return self._diagnostic(
                "warning",
                self._missing_packed_orm_texture_code(channel_state),
                f"cannot rewrite packed ORM texture for {edit.channel}; no local resolved ORM texture was found",
                slot=slot,
                edit=edit,
                metadata={
                    "texture_input_names": tuple(channel_state.texture_input_names),
                    "texture_refs": tuple(ref.input_name for ref in channel_state.texture_refs),
                },
            )
        payload = dict(edit.value) if isinstance(edit.value, dict) else {"value": edit.value}
        layout = payload.get("layout") or "orm_rgb"
        if edit.operation == RandomizationOp.SET_CHANNEL_VALUE:
            output_path = write_packed_orm_channel_value_texture(
                source_path=source_path,
                output_dir=self.generated_texture_dir,
                channel=edit.channel,
                value=payload.get("value"),
                layout=layout,
            )
        elif edit.operation == RandomizationOp.TRANSFORM_CHANNEL:
            transform = str(payload.get("transform") or "").strip()
            output_path = write_packed_orm_channel_transform_texture(
                source_path=source_path,
                output_dir=self.generated_texture_dir,
                channel=edit.channel,
                transform=transform,
                value=payload.get("value"),
                layout=layout,
            )
        else:
            return self._diagnostic(
                "warning",
                "unimplemented_operation",
                f"USD backend does not implement packed ORM operation {edit.operation}",
                slot=slot,
                edit=edit,
            )
        orm_state = self._orm_channel_state_for_slot(slot, fallback=channel_state)
        self._set_texture_asset(
            shader=shader,
            channel_state=orm_state,
            texture_uri=str(output_path),
            target_ref=target_ref,
        )
        self._enable_texture_binding(
            slot=slot,
            shader=shader,
            edit=replace(edit, channel=MaterialChannel.ORM),
        )
        return None

    def _disable_packed_orm_and_set_values(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        values = packed_orm_disable_values_payload(edit.value)
        scalar_inputs: dict[str, tuple[str, Any]] = {}
        for scalar_channel, scalar_value in values.items():
            scalar_state = slot.channel_states.get(scalar_channel)
            scalar_io = channel_io(slot.shader_model, scalar_channel)
            scalar_input_names = tuple(scalar_io.constant_inputs) if scalar_io is not None else ()
            input_name = (
                scalar_state.constant_input_name
                if scalar_state is not None and scalar_state.constant_input_name
                else (self._first_existing_input_name(shader, scalar_input_names) if scalar_input_names else "")
            )
            if not input_name:
                return self._diagnostic(
                    "warning",
                    "missing_packed_orm_scalar_input",
                    f"slot does not expose writable scalar input for {scalar_channel}",
                    slot=slot,
                    edit=edit,
                    metadata={"scalar_channel": scalar_channel},
                )
            scalar_inputs[scalar_channel] = (input_name, scalar_io)

        for texture_ref in channel_state.texture_refs:
            self._clear_texture_asset(shader=shader, texture_ref=texture_ref)
        for texture_input_name in channel_state.texture_input_names:
            if texture_input_name not in {ref.input_name for ref in channel_state.texture_refs}:
                self._clear_texture_asset(
                    shader=shader,
                    texture_ref=MaterialTextureRef(input_name=texture_input_name),
                )
        orm_io = channel_io(slot.shader_model, MaterialChannel.ORM)
        if orm_io is not None:
            for flag_name, flag_value in orm_io.disable_flags.items():
                self._set_input_value(shader=shader, input_name=flag_name, value=flag_value)
            for value_name, value in orm_io.disable_values.items():
                self._set_input_value(shader=shader, input_name=value_name, value=value)

        for scalar_channel, scalar_value in values.items():
            input_name, scalar_io = scalar_inputs[scalar_channel]
            if scalar_io is not None:
                for flag_name, flag_value in scalar_io.disable_flags.items():
                    self._set_input_value(shader=shader, input_name=flag_name, value=flag_value)
                for value_name, value in scalar_io.disable_values.items():
                    self._set_input_value(shader=shader, input_name=value_name, value=value)
            self._set_channel_value(
                shader=shader,
                channel=scalar_channel,
                input_name=input_name,
                value=scalar_value,
            )
        return None

    def _replace_texture(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        texture_uri = self._texture_uri(edit.value)
        if not texture_uri:
            return self._diagnostic(
                "warning",
                "missing_texture_uri",
                f"texture.replace requires a texture uri for {edit.channel}",
                slot=slot,
                edit=edit,
            )
        target_ref = self._target_texture_ref(channel_state)
        if target_ref is None and not channel_state.texture_input_names:
            return self._missing_writable_input(slot=slot, channel_state=channel_state, edit=edit)
        self._set_texture_asset(
            shader=shader,
            channel_state=channel_state,
            texture_uri=texture_uri,
            target_ref=target_ref,
        )
        self._apply_template_texture_scale(shader=shader, edit=edit)
        if (
            edit.metadata.get("reset_texture_channel_defaults")
            or edit.metadata.get("desired_source_binding") == ChannelSource.TEXTURE
        ):
            self._enable_texture_binding(slot=slot, shader=shader, edit=edit)
        return None

    def _apply_template_texture_scale(self, *, shader: Any, edit: MaterialEdit) -> None:
        scale = _texture_scale_from_metadata(edit.metadata)
        if scale is None:
            return
        _, Sdf = _require_pxr()
        try:
            from pxr import Gf
        except ImportError:
            return
        value = Gf.Vec2f(float(scale[0]), float(scale[1]))
        for input_name in ("texture_scale", "uv_scale", "project_uvw_scale"):
            existing = shader.GetInput(input_name)
            if existing or input_name == "texture_scale":
                shader.CreateInput(input_name, Sdf.ValueTypeNames.Float2).Set(value)

    def _apply_texture_value_transform(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        if self._apply_native_texture_value_transform(slot=slot, shader=shader, edit=edit):
            return None
        source_path, target_ref = self._source_texture_path(channel_state)
        if source_path is None or target_ref is None:
            return self._diagnostic(
                "warning",
                "missing_resolved_texture",
                f"cannot generate texture transform for {edit.channel}; no local resolved texture was found",
                slot=slot,
                edit=edit,
                metadata={
                    "texture_input_names": tuple(channel_state.texture_input_names),
                    "texture_refs": tuple(ref.input_name for ref in channel_state.texture_refs),
                },
            )
        output_path = write_value_transformed_texture(
            source_path=source_path,
            output_dir=self.generated_texture_dir,
            channel=edit.channel,
            operation=edit.operation,
            value=edit.value,
        )
        self._set_texture_asset(
            shader=shader,
            channel_state=channel_state,
            texture_uri=str(output_path),
            target_ref=target_ref,
        )
        return None

    def _apply_native_texture_value_transform(self, *, slot: MaterialSlot, shader: Any, edit: MaterialEdit) -> bool:
        io = channel_io(slot.shader_model, edit.channel)
        if io is None:
            return False
        if edit.operation == RandomizationOp.SCALE and io.native_scale_inputs:
            if isinstance(edit.value, tuple):
                input_name = self._first_existing_input_name(shader, tuple(io.native_scale_inputs))
                if input_name == "albedo_brightness":
                    return False
                self._set_input_value(shader=shader, input_name=input_name, value=edit.value)
                return True
            if "albedo_brightness" in io.native_scale_inputs:
                self._set_input_value(
                    shader=shader,
                    input_name="albedo_brightness",
                    value=float(edit.value),
                )
                return True
            input_name = self._first_existing_input_name(shader, tuple(io.native_scale_inputs))
            self._set_input_value(shader=shader, input_name=input_name, value=float(edit.value))
            return True
        if edit.operation == RandomizationOp.BIAS and io.native_bias_inputs:
            if isinstance(edit.value, tuple):
                if not (
                    abs(float(edit.value[0]) - float(edit.value[1])) < 1e-6
                    and abs(float(edit.value[1]) - float(edit.value[2])) < 1e-6
                ):
                    return False
                value = float(edit.value[0])
            else:
                value = float(edit.value)
            input_name = self._first_existing_input_name(shader, tuple(io.native_bias_inputs))
            self._set_input_value(shader=shader, input_name=input_name, value=value)
            return True
        return False

    def _disable_texture_and_set_value(
        self,
        *,
        slot: MaterialSlot,
        shader: Any,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic | None:
        input_name = channel_state.constant_input_name
        if not input_name:
            return self._missing_writable_input(slot=slot, channel_state=channel_state, edit=edit)
        for texture_ref in channel_state.texture_refs:
            self._clear_texture_asset(shader=shader, texture_ref=texture_ref)
        for texture_input_name in channel_state.texture_input_names:
            if texture_input_name not in {ref.input_name for ref in channel_state.texture_refs}:
                self._clear_texture_asset(
                    shader=shader,
                    texture_ref=MaterialTextureRef(input_name=texture_input_name),
                )
        io = channel_io(slot.shader_model, edit.channel)
        if io is not None:
            for flag_name, flag_value in io.disable_flags.items():
                self._set_input_value(shader=shader, input_name=flag_name, value=flag_value)
            for value_name, value in io.disable_values.items():
                self._set_input_value(shader=shader, input_name=value_name, value=value)
        if edit.channel == MaterialChannel.OPACITY:
            self._set_input_value(
                shader=shader,
                input_name="enable_opacity",
                value=float(edit.value) < 0.999,
            )
        self._set_channel_value(shader=shader, channel=edit.channel, input_name=input_name, value=edit.value)
        return None

    def _set_texture_asset(
        self,
        *,
        shader: Any,
        channel_state: MaterialChannelState,
        texture_uri: str,
        target_ref: MaterialTextureRef | None,
    ) -> None:
        _, Sdf = _require_pxr()
        if target_ref and target_ref.connected_source:
            source_shader = self._shader_at_path(target_ref.connected_source)
            if source_shader is not None:
                source_input_name = target_ref.source_input_name or "file"
                source_shader.CreateInput(source_input_name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_uri))
                return
        input_name = target_ref.input_name if target_ref else channel_state.texture_input_names[0]
        shader.CreateInput(input_name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_uri))

    def _enable_texture_binding(self, *, slot: MaterialSlot, shader: Any, edit: MaterialEdit) -> None:
        io = channel_io(slot.shader_model, edit.channel)
        if io is None:
            return
        for flag_name, flag_value in io.enable_flags.items():
            self._set_input_value(shader=shader, input_name=flag_name, value=flag_value)
        for value_name, value in io.enable_values.items():
            self._set_input_value(shader=shader, input_name=value_name, value=value)
        if edit.metadata.get("reset_texture_channel_defaults"):
            for value_name, value in io.disable_values.items():
                if value_name in io.enable_values:
                    continue
                self._set_input_value(shader=shader, input_name=value_name, value=value)
            for flag_name, flag_value in io.disable_flags.items():
                if flag_name in io.enable_flags:
                    continue
                self._set_input_value(shader=shader, input_name=flag_name, value=flag_value)

    def _clear_texture_asset(self, *, shader: Any, texture_ref: MaterialTextureRef) -> None:
        _, Sdf = _require_pxr()
        if texture_ref.connected_source:
            source_shader = self._shader_at_path(texture_ref.connected_source)
            if source_shader is not None:
                source_shader.CreateInput(texture_ref.source_input_name or "file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath("")
                )
        shader_input = shader.GetInput(texture_ref.input_name)
        if shader_input:
            try:
                shader_input.GetAttr().ClearConnections()
            except Exception:
                pass
            shader.CreateInput(texture_ref.input_name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(""))

    def _set_channel_value(self, *, shader: Any, channel: str, input_name: str, value: Any) -> None:
        shader.CreateInput(str(input_name), channel_value_type(channel)).Set(infer_usd_value(value))

    def _set_input_value(self, *, shader: Any, input_name: str, value: Any) -> None:
        _, Sdf = _require_pxr()
        if isinstance(value, bool):
            value_type = Sdf.ValueTypeNames.Bool
            usd_value = value
        elif isinstance(value, tuple) and len(value) == 3:
            value_type = Sdf.ValueTypeNames.Color3f
            usd_value = infer_usd_value(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            value_type = Sdf.ValueTypeNames.Int
            usd_value = int(value)
        else:
            value_type = Sdf.ValueTypeNames.Float
            usd_value = float(value)
        shader.CreateInput(str(input_name), value_type).Set(usd_value)

    def _shader_at_path(self, path: str) -> Any | None:
        UsdShade, _ = _require_pxr()
        prim = self.stage.GetPrimAtPath(str(path))
        if not prim or not prim.IsValid():
            return None
        shader = UsdShade.Shader(prim)
        return shader if shader and shader.GetPrim().IsValid() else None

    def _source_texture_path(
        self, channel_state: MaterialChannelState
    ) -> tuple[Path | None, MaterialTextureRef | None]:
        for texture_ref in channel_state.texture_refs:
            for candidate in (texture_ref.resolved_path, texture_ref.asset_path):
                if not candidate:
                    continue
                path = Path(str(candidate))
                if path.is_file():
                    return path, texture_ref
        return None, None

    @staticmethod
    def _missing_packed_orm_texture_code(channel_state: MaterialChannelState) -> str:
        for texture_ref in channel_state.texture_refs:
            for candidate in (texture_ref.resolved_path, texture_ref.asset_path):
                if candidate and "://" in str(candidate):
                    return "remote_packed_orm_transform_requires_resolver"
        return "missing_resolved_packed_orm_texture"

    def _orm_channel_state_for_slot(
        self,
        slot: MaterialSlot,
        *,
        fallback: MaterialChannelState,
    ) -> MaterialChannelState:
        state = slot.channel_states.get(MaterialChannel.ORM)
        if state is not None and state.texture_input_names:
            return state
        io = channel_io(slot.shader_model, MaterialChannel.ORM)
        if io is None:
            return fallback
        return replace(
            fallback,
            channel=MaterialChannel.ORM,
            source_type=ChannelSource.PACKED_TEXTURE,
            texture_input_names=fallback.texture_input_names or tuple(io.texture_inputs),
            writable_ops=implemented_ops(MaterialChannel.ORM, ChannelSource.PACKED_TEXTURE),
            planned_ops=planned_ops(MaterialChannel.ORM, ChannelSource.PACKED_TEXTURE),
        )

    @staticmethod
    def _target_texture_ref(
        channel_state: MaterialChannelState,
    ) -> MaterialTextureRef | None:
        return channel_state.texture_refs[0] if channel_state.texture_refs else None

    @staticmethod
    def _texture_uri(value: Any) -> str:
        if isinstance(value, dict):
            return str(
                value.get("uri") or value.get("path") or value.get("texture") or value.get("value") or ""
            ).strip()
        return str(value or "").strip()

    def _missing_writable_input(
        self,
        *,
        slot: MaterialSlot,
        channel_state: MaterialChannelState,
        edit: MaterialEdit,
    ) -> RandomizationDiagnostic:
        return self._diagnostic(
            "warning",
            "missing_writable_input",
            f"slot does not expose writable input for {edit.channel} operation {edit.operation}",
            slot=slot,
            edit=edit,
            metadata={
                "constant_input_name": channel_state.constant_input_name,
                "texture_input_names": tuple(channel_state.texture_input_names),
            },
        )

    @staticmethod
    def _diagnostic(
        severity: str,
        code: str,
        message: str,
        *,
        slot: MaterialSlot,
        edit: MaterialEdit,
        metadata: dict[str, Any] | None = None,
    ) -> RandomizationDiagnostic:
        payload = dict(metadata or {})
        payload.update(
            {
                "channel": edit.channel,
                "source_type": edit.source_type,
                "operation": edit.operation,
            }
        )
        return RandomizationDiagnostic(
            severity=severity,
            code=code,
            message=message,
            asset_id=slot.asset_id,
            material_path=slot.material_path,
            metadata=payload,
        )

    def _channel_states(
        self,
        *,
        shader: Any,
        constant_inputs: dict[str, tuple[str, ...]],
        texture_inputs: dict[str, tuple[str, ...]],
        oqc_texture_refs: tuple[Any, ...],
    ) -> dict[str, MaterialChannelState]:
        channels = set(constant_inputs) | set(texture_inputs)
        if MaterialChannel.ORM in texture_inputs:
            channels.add(MaterialChannel.ORM)
        orm_refs = (
            self._texture_refs_for_channel(
                shader=shader,
                channel=MaterialChannel.ORM,
                texture_input_names=texture_inputs.get(MaterialChannel.ORM, ()),
                oqc_texture_refs=oqc_texture_refs,
            )
            if self._bool_shader_input(shader, "enable_ORM_texture", False)
            else ()
        )
        states: dict[str, MaterialChannelState] = {}
        for channel in sorted(channels):
            constant_names = constant_inputs.get(channel, ())
            texture_names = texture_inputs.get(channel, ())
            constant_input = self._first_existing_input_name(shader, constant_names) if constant_names else None
            texture_refs = (
                orm_refs
                if channel in {MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC} and orm_refs
                else self._texture_refs_for_channel(
                    shader=shader,
                    channel=channel,
                    texture_input_names=texture_names,
                    oqc_texture_refs=oqc_texture_refs,
                )
            )
            value = read_input_value(shader, constant_names) if constant_names else None
            source_type = self._channel_source_type(channel=channel, value=value, texture_refs=texture_refs)
            if (
                channel in {MaterialChannel.ROUGHNESS, MaterialChannel.METALLIC}
                and texture_refs == orm_refs
                and orm_refs
            ):
                source_type = ChannelSource.PACKED_TEXTURE
            if (
                source_type == ChannelSource.MISSING
                and constant_input is not None
                and not texture_refs
                and channel != MaterialChannel.NORMAL
            ):
                source_type = ChannelSource.CONSTANT
            if source_type == ChannelSource.MISSING and constant_input is None and not texture_names:
                source_type = ChannelSource.UNSUPPORTED
            states[channel] = MaterialChannelState(
                channel=channel,
                source_type=source_type,
                value=value,
                constant_input_name=constant_input,
                texture_input_names=tuple(ref.input_name for ref in texture_refs) or texture_names,
                texture_refs=texture_refs,
                writable_ops=implemented_ops(channel, source_type),
                planned_ops=planned_ops(channel, source_type),
            )
        return states

    def _texture_refs_for_channel(
        self,
        *,
        shader: Any,
        channel: str,
        texture_input_names: tuple[str, ...],
        oqc_texture_refs: tuple[Any, ...],
    ) -> tuple[MaterialTextureRef, ...]:
        refs: list[MaterialTextureRef] = []
        seen: set[str] = set()
        wanted = set(texture_input_names)
        for ref in oqc_texture_refs:
            input_name = str(getattr(ref, "input_name", "") or "")
            if not input_name or input_name not in wanted:
                continue
            refs.append(
                MaterialTextureRef(
                    input_name=input_name,
                    asset_path=str(getattr(ref, "authored_path", "") or "") or None,
                    resolved_path=str(getattr(ref, "resolved_path", "") or "") or None,
                    connected_source=str(getattr(ref, "source_shader_path", "") or "") or None,
                    source_kind=str(getattr(ref, "source_kind", "") or "direct"),
                    source_input_name=str(getattr(ref, "source_input_name", "") or "") or None,
                    packed_channels=(("occlusion", "roughness", "metallic") if channel == MaterialChannel.ORM else ()),
                )
            )
            seen.add(input_name)
        for input_name in texture_input_names:
            if input_name in seen:
                continue
            shader_input = shader.GetInput(input_name)
            if not shader_input:
                continue
            connected_source = (
                input_connected_source_path(shader_input) if input_has_connected_source(shader_input) else None
            )
            asset_path = input_asset_path(shader_input)
            resolved_path = input_resolved_asset_path(shader_input)
            if connected_source or asset_path:
                refs.append(
                    MaterialTextureRef(
                        input_name=input_name,
                        asset_path=asset_path,
                        resolved_path=resolved_path,
                        connected_source=connected_source,
                        source_kind="connected" if connected_source else "direct",
                        source_input_name="file" if connected_source else None,
                        packed_channels=(
                            ("occlusion", "roughness", "metallic") if channel == MaterialChannel.ORM else ()
                        ),
                    )
                )
        return tuple(refs)

    @staticmethod
    def _bool_shader_input(shader: Any, input_name: str, default: bool) -> bool:
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            return bool(default)
        value = shader_input.Get()
        return value if isinstance(value, bool) else bool(default)

    @staticmethod
    def _channel_source_type(
        *,
        channel: str,
        value: Any,
        texture_refs: tuple[MaterialTextureRef, ...],
    ) -> str:
        if channel == MaterialChannel.ORM and texture_refs:
            return ChannelSource.PACKED_TEXTURE
        if texture_refs:
            return ChannelSource.TEXTURE
        if channel == MaterialChannel.NORMAL:
            return ChannelSource.MISSING
        if value is not None:
            return ChannelSource.CONSTANT
        return ChannelSource.MISSING

    def _asset_handle_from_prim(self, prim: Any) -> AssetHandle:
        path = str(prim.GetPath())
        return AssetHandle(
            asset_id=path,
            prim_path=path,
            name=prim.GetName(),
            metadata={"semantic_tags": self._semantic_tags(prim)},
        )

    @staticmethod
    def _semantic_tags(prim: Any) -> tuple[str, ...]:
        return semantic_tags_for_prim(prim)

    def _surface_shader(self, material: Any) -> Any | None:
        UsdShade, _ = _require_pxr()
        for output in material.GetOutputs():
            try:
                sources, _invalid = output.GetConnectedSources()
            except Exception:
                continue
            for source in sources:
                connectable = source.source
                if connectable and connectable.GetPrim().IsValid():
                    return UsdShade.Shader(connectable.GetPrim())
        try:
            shader = material.ComputeSurfaceSource()[0]
        except Exception:
            shader = None
        return shader

    def _material_consumers(self) -> dict[str, tuple[str, ...]]:
        UsdShade, _ = _require_pxr()
        consumers: dict[str, list[str]] = {}
        for prim in self.stage.Traverse():
            material, _relationship = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            if not material:
                continue
            consumers.setdefault(str(material.GetPath()), []).append(str(prim.GetPath()))
        return {path: tuple(sorted(set(paths))) for path, paths in consumers.items()}

    def _has_uv_primvars(self, prim_paths: tuple[str, ...]) -> bool:
        try:
            from pxr import UsdGeom
        except ImportError:
            return False
        for prim_path in prim_paths:
            prim = self.stage.GetPrimAtPath(str(prim_path))
            if not prim or not prim.IsValid():
                continue
            for candidate in self._iter_mesh_prims(prim):
                for primvar in UsdGeom.PrimvarsAPI(candidate).GetPrimvars() or ():
                    name = str(primvar.GetPrimvarName() or primvar.GetName() or "").lower()
                    if name not in {"st", "uv", "uvmap", "map1"} and not name.endswith(":st"):
                        continue
                    if _has_authored_sequence_value(primvar.Get()):
                        return True
        return False

    @staticmethod
    def _iter_mesh_prims(root: Any) -> tuple[Any, ...]:
        from pxr import Usd, UsdGeom

        prims: list[Any] = []
        if root.IsA(UsdGeom.Mesh):
            prims.append(root)
        for prim in Usd.PrimRange(root):
            if prim == root:
                continue
            if prim.IsA(UsdGeom.Mesh):
                prims.append(prim)
        return tuple(prims)

    @staticmethod
    def _first_existing_input_name(shader: Any, input_names: tuple[str, ...]) -> str:
        for input_name in input_names:
            if shader.GetInput(input_name):
                return input_name
        return input_names[0]

    @staticmethod
    def _existing_input_names(shader: Any, input_names: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(input_name for input_name in input_names if shader.GetInput(input_name))

    @staticmethod
    def _slot_id(asset_id: str, material_path: str) -> str:
        return f"{asset_id}:{material_path}"

    def _asset_usd_uri(self) -> str:
        if self.asset_usd_uri:
            return self.asset_usd_uri
        root_layer = self.stage.GetRootLayer()
        return str(getattr(root_layer, "identifier", "") or "")

    def _clone_material_for_slot(self, slot: MaterialSlot) -> MaterialSlot:
        UsdShade, _ = _require_pxr()
        material = UsdShade.Material(self.stage.GetPrimAtPath(slot.material_path))
        shader_path = slot.metadata.get("shader_path")
        source_shader = UsdShade.Shader(self.stage.GetPrimAtPath(str(shader_path))) if shader_path else None
        if not material or not source_shader:
            return slot
        asset_prim_path = str(slot.metadata.get("asset_prim_path") or slot.asset_id)
        clone_path = self._clone_material_path(asset_prim_path, slot.material_path)
        self.stage.DefinePrim(str(clone_path.GetParentPath()), "Scope")
        clone_material = UsdShade.Material.Define(self.stage, clone_path)
        clone_shader_path = clone_path.AppendChild(_valid_identifier(source_shader.GetPrim().GetName() or "Shader"))
        clone_shader = UsdShade.Shader.Define(self.stage, clone_shader_path)
        shader_id = source_shader.GetIdAttr().Get()
        if shader_id:
            clone_shader.CreateIdAttr(shader_id)
        for attr_name in (
            "info:implementationSource",
            "info:mdl:sourceAsset",
            "info:mdl:sourceAsset:subIdentifier",
        ):
            source_attr = source_shader.GetPrim().GetAttribute(attr_name)
            if not source_attr or not source_attr.IsValid():
                continue
            value = source_attr.Get()
            if value is None:
                continue
            clone_shader.GetPrim().CreateAttribute(
                attr_name,
                source_attr.GetTypeName(),
                variability=source_attr.GetVariability(),
            ).Set(value)
        for shader_input in source_shader.GetInputs():
            clone_input = clone_shader.CreateInput(shader_input.GetBaseName(), shader_input.GetTypeName())
            value = shader_input.Get()
            if value is not None:
                clone_input.Set(value)
            connections = shader_input.GetAttr().GetConnections()
            if connections:
                clone_input.GetAttr().SetConnections(connections)
        for output in material.GetOutputs():
            clone_output = clone_material.CreateOutput(output.GetBaseName(), output.GetTypeName())
            clone_output.ConnectToSource(clone_shader.CreateOutput("out", output.GetTypeName()))
        bound_prim_paths = tuple(slot.metadata.get("bound_prim_paths", (slot.bound_prim_path,)))
        for bound_prim_path in bound_prim_paths:
            prim = self.stage.GetPrimAtPath(str(bound_prim_path))
            if prim and prim.IsValid():
                UsdShade.MaterialBindingAPI(prim).Bind(clone_material)
        metadata = dict(slot.metadata)
        metadata["shader_path"] = str(clone_shader_path)
        metadata["cloned_from_material_path"] = slot.material_path
        channel_states = {
            channel: replace(state, constant_input_name=state.constant_input_name)
            for channel, state in slot.channel_states.items()
        }
        cloned = MaterialSlot(
            slot_id=slot.slot_id,
            asset_id=slot.asset_id,
            material_path=str(clone_path),
            bound_prim_path=slot.bound_prim_path,
            shader_model=slot.shader_model,
            supported_channels=slot.supported_channels,
            is_shared=False,
            channel_values=slot.channel_values,
            channel_sources=slot.channel_sources,
            channel_states=channel_states,
            metadata=metadata,
        )
        self._slot_cache[slot.slot_id] = cloned
        return cloned

    @staticmethod
    def _clone_material_path(asset_prim_path: str, material_path: str) -> Any:
        from pxr import Sdf

        digest = hashlib.blake2b(material_path.encode("utf-8"), digest_size=4).hexdigest()
        name = _valid_identifier(f"{material_path.rstrip('/').split('/')[-1]}_{digest}")
        return Sdf.Path(asset_prim_path).AppendChild("RandomizedLooks").AppendChild(name)


def _valid_identifier(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not value or value[0].isdigit():
        value = f"_{value}"
    return value


def _texture_scale_from_metadata(metadata: Any) -> tuple[float, float] | None:
    if not isinstance(metadata, Mapping):
        return None
    raw_scale = metadata.get("texture_scale")
    if isinstance(raw_scale, (list, tuple)) and len(raw_scale) == 2:
        try:
            return (float(raw_scale[0]), float(raw_scale[1]))
        except (TypeError, ValueError):
            return None
    if raw_scale is not None:
        try:
            parsed = float(raw_scale)
        except (TypeError, ValueError):
            parsed = 0.0
        if parsed > 0.0:
            return (parsed, parsed)
    raw_tile_size = metadata.get("uv_tile_size_m")
    try:
        tile_size = float(raw_tile_size)
    except (TypeError, ValueError):
        return None
    if tile_size <= 0.0:
        return None
    scale = 1.0 / tile_size
    return (scale, scale)


def _has_authored_sequence_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True
