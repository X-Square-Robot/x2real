from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import cache
from typing import Any

from ...errors import BackendUnavailableError

_OQC_PACKAGE_CANDIDATES = (
    "object_quality_checker",
    "manaenv.scripts.asset_validation.object_quality_checker",
)


@dataclass(frozen=True)
class OqcMaterialInventory:
    object_root_path: str
    object_root_source: str
    surface_mesh_paths: tuple[str, ...]
    mesh_to_material_paths: dict[str, tuple[str, ...]]
    material_records: dict[str, Any]
    surface_mesh_count: int


@dataclass(frozen=True)
class NativeMaterialRecord:
    material_path: str
    effective_shader_type: str | None
    texture_refs: tuple[Any, ...] = ()
    issue_codes: tuple[str, ...] = ()
    bound_mesh_count: int = 0


@cache
def _import_oqc_submodule(suffix: str) -> Any:
    errors: list[str] = []
    for package_name in _OQC_PACKAGE_CANDIDATES:
        module_name = f"{package_name}.{suffix}"
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
    raise BackendUnavailableError(
        "object_quality_checker is required for USD asset inspection; "
        + "tried "
        + ", ".join(_OQC_PACKAGE_CANDIDATES)
        + ". Import errors: "
        + " | ".join(errors)
    )


def path_within(path_text: str, root_text: str) -> bool:
    path = str(path_text or "").rstrip("/")
    root = str(root_text or "").rstrip("/")
    if not path or not root:
        return False
    return path == root or path.startswith(root + "/")


def resolve_object_root_path(stage: Any, asset_prim_path: str) -> tuple[str, str]:
    package_root = stage.GetPrimAtPath(asset_prim_path) if stage is not None else None
    if not package_root or not package_root.IsValid():
        return asset_prim_path, "asset_prim_invalid"
    try:
        stage_open = _import_oqc_submodule("usd.stage_open")
        resolution = stage_open.resolve_object_root_prim(stage, package_root=package_root)
        if resolution and resolution.prim and resolution.prim.IsValid():
            return resolution.prim.GetPath().pathString, str(resolution.source)
    except BackendUnavailableError:
        return asset_prim_path, "native_fallback_asset_prim"
    return asset_prim_path, "fallback_asset_prim"


def inspect_material_inventory(
    *,
    stage: Any,
    asset_prim_path: str,
    asset_usd_uri: str,
) -> OqcMaterialInventory:
    object_root_path, object_root_source = resolve_object_root_path(stage, asset_prim_path)
    try:
        mesh_selection_mod = _import_oqc_submodule("usd.mesh_selection")
        materials_mod = _import_oqc_submodule("usd.materials")
        mesh_selection = mesh_selection_mod.build_mesh_selection(stage, object_root_path)
        inspection = materials_mod.inspect_materials(
            stage=stage,
            object_root_path=object_root_path,
            asset_usd_uri=asset_usd_uri,
            mesh_selection=mesh_selection,
        )
        return OqcMaterialInventory(
            object_root_path=object_root_path,
            object_root_source=object_root_source,
            surface_mesh_paths=tuple(mesh_selection.surface_mesh_paths),
            mesh_to_material_paths=dict(inspection.mesh_to_material_paths),
            material_records={item.material_path: item for item in inspection.materials},
            surface_mesh_count=int(inspection.surface_mesh_count),
        )
    except BackendUnavailableError:
        return _inspect_material_inventory_native(
            stage=stage,
            object_root_path=object_root_path,
            object_root_source=object_root_source,
        )


def semantic_tags_for_prim(prim: Any) -> tuple[str, ...]:
    try:
        semantics = _import_oqc_submodule("usd.semantics")
        inspection = semantics.inspect_embedded_semantics(prim)
        merged = semantics.merge_embedded_semantics_payloads([inspection])
        downstream = semantics.normalize_semantics_for_downstream(merged)

        tags: list[str] = []
        primary_category = downstream.get("primary_category")
        if primary_category:
            tags.append(str(primary_category))
        for key in ("taxonomy_tags", "placement_tags", "affordance_tags"):
            tags.extend(str(value) for value in tuple(downstream.get(key) or ()))
    except BackendUnavailableError:
        tags = _semantic_tags_native(prim)

    # Keep support for module-local ad hoc tags while OQC remains the canonical
    # extractor for asset_semantics and UsdSemantics metadata.
    custom_data = prim.GetCustomData() or {}
    for key in ("semantic_tags", "tags"):
        tags.extend(_string_values(custom_data.get(key)))
    return tuple(dict.fromkeys(item for item in tags if item))


def _inspect_material_inventory_native(
    *,
    stage: Any,
    object_root_path: str,
    object_root_source: str,
) -> OqcMaterialInventory:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    except ImportError as exc:
        raise BackendUnavailableError("pxr is required for native USD material inspection fallback") from exc

    root = stage.GetPrimAtPath(object_root_path) if stage is not None else None
    if not root or not root.IsValid():
        return OqcMaterialInventory(
            object_root_path=object_root_path,
            object_root_source=object_root_source,
            surface_mesh_paths=(),
            mesh_to_material_paths={},
            material_records={},
            surface_mesh_count=0,
        )

    surface_meshes: list[Any] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(prim)
        visibility = str(imageable.ComputeVisibility() or "").strip() if imageable else ""
        purpose = str(imageable.ComputePurpose() or "").strip().lower() if imageable else ""
        if visibility == str(UsdGeom.Tokens.invisible):
            continue
        if purpose and purpose not in {"default", "render", "proxy"}:
            continue
        surface_meshes.append(prim)

    mesh_to_material_paths: dict[str, tuple[str, ...]] = {}
    records: dict[str, NativeMaterialRecord] = {}
    bound_counts: dict[str, int] = {}
    for mesh in surface_meshes:
        materials = _bound_render_materials(mesh, UsdPhysics=UsdPhysics, UsdShade=UsdShade)
        if not materials:
            continue
        material_paths = tuple(material.GetPath().pathString for material in materials)
        mesh_to_material_paths[mesh.GetPath().pathString] = material_paths
        for material in materials:
            material_path = material.GetPath().pathString
            bound_counts[material_path] = bound_counts.get(material_path, 0) + 1
            if material_path in records:
                continue
            shader = _surface_shader(material, UsdShade=UsdShade)
            if shader is None:
                records[material_path] = NativeMaterialRecord(
                    material_path=material_path,
                    effective_shader_type=None,
                    issue_codes=("material_shader_missing",),
                )
                continue
            input_names = tuple(shader_input.GetBaseName() for shader_input in shader.GetInputs())
            shader_id = shader.GetIdAttr().Get() or ""
            from .shader_io import canonical_shader_model

            records[material_path] = NativeMaterialRecord(
                material_path=material_path,
                effective_shader_type=canonical_shader_model(str(shader_id), input_names),
            )

    material_records = {
        path: NativeMaterialRecord(
            material_path=record.material_path,
            effective_shader_type=record.effective_shader_type,
            texture_refs=record.texture_refs,
            issue_codes=record.issue_codes,
            bound_mesh_count=bound_counts.get(path, record.bound_mesh_count),
        )
        for path, record in records.items()
    }
    return OqcMaterialInventory(
        object_root_path=object_root_path,
        object_root_source=object_root_source,
        surface_mesh_paths=tuple(mesh.GetPath().pathString for mesh in surface_meshes),
        mesh_to_material_paths=dict(sorted(mesh_to_material_paths.items())),
        material_records=dict(sorted(material_records.items())),
        surface_mesh_count=len(surface_meshes),
    )


def _bound_render_materials(prim: Any, *, UsdPhysics: Any, UsdShade: Any) -> tuple[Any, ...]:
    materials: list[Any] = []
    seen: set[str] = set()

    def append(material: Any) -> None:
        if not material or not material.GetPrim().IsValid():
            return
        material_prim = material.GetPrim()
        path_text = material_prim.GetPath().pathString.lower()
        if material_prim.HasAPI(UsdPhysics.MaterialAPI) or "physicsmaterial" in path_text:
            return
        path = material.GetPath().pathString
        if path in seen:
            return
        seen.add(path)
        materials.append(material)

    binding_api = UsdShade.MaterialBindingAPI(prim)
    append(binding_api.ComputeBoundMaterial()[0])
    for subset in binding_api.GetMaterialBindSubsets():
        append(UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0])
    return tuple(materials)


def _surface_shader(material: Any, *, UsdShade: Any) -> Any | None:
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
    return shader if shader and shader.GetPrim().IsValid() else None


def _semantic_tags_native(prim: Any) -> list[str]:
    tags: list[str] = []
    if not prim or not prim.IsValid():
        return tags
    payload = prim.GetCustomDataByKey("asset_semantics")
    if isinstance(payload, dict):
        primary_category = payload.get("primary_category")
        if primary_category:
            tags.append(str(primary_category))
        for key in ("taxonomy_tags", "placement_tags", "affordance_tags"):
            tags.extend(_string_values(payload.get(key)))
    try:
        from pxr import UsdSemantics

        for taxonomy in (
            "taxonomy_primary_category",
            "taxonomy_tags",
            "placement_tags",
            "affordance_tags",
        ):
            api = UsdSemantics.LabelsAPI.Get(prim, taxonomy)
            if api:
                labels = api.GetLabelsAttr().Get() or ()
                tags.extend(_string_values(labels))
    except Exception:
        pass
    return tags


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]
