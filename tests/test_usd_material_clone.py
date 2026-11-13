from pathlib import Path

import pytest
from pxr import Sdf, Usd, UsdGeom, UsdShade

from manaenv.pipeline.randomization.asset_randomizer.backends.usd.backend import UsdSceneBackend
from manaenv.pipeline.randomization.asset_randomizer.materials.channels import (
    ChannelSource,
    MaterialChannel,
    RandomizationOp,
)
from manaenv.pipeline.randomization.asset_randomizer.model import AssetHandle, MaterialEdit


@pytest.fixture
def shared_material_scene(tmp_path: Path):
    for relative in ("textures/albedo.png", "textures/normal.png", "materials/local.mdl"):
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"Asset-resolution test fixture")

    source = Usd.Stage.CreateNew(str(tmp_path / "source.usda"))
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Asset").GetPrim())
    mesh = UsdGeom.Mesh.Define(source, "/Asset/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    material = UsdShade.Material.Define(source, "/Asset/Looks/Painted")
    shader = UsdShade.Shader.Define(source, "/Asset/Looks/Painted/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("./materials/local.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("./textures/albedo.png"))
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/normal.png"))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("texture_set", Sdf.ValueTypeNames.AssetArray).Set(
        Sdf.AssetPathArray(
            [Sdf.AssetPath("./textures/albedo.png"), Sdf.AssetPath("textures/normal.png"), Sdf.AssetPath("")]
        )
    )
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    source.GetRootLayer().Save()

    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim("/Asset", "Xform").GetReferences().AddReference(str(tmp_path / "source.usda"))
    other = UsdGeom.Mesh.Define(stage, "/Other")
    UsdShade.MaterialBindingAPI.Apply(other.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(material.GetPath())))
    backend = UsdSceneBackend(stage)
    slots = backend.inspect_material_slots(AssetHandle(asset_id="asset", prim_path="/Asset"))
    assert len(slots) == 1
    assert slots[0].is_shared
    return stage, backend, slots[0], tmp_path


def edit_shared_material(scene):
    stage, backend, slot, _ = scene
    report = backend.apply_material_edits(
        (
            MaterialEdit(
                target_slot_id=slot.slot_id,
                channel=MaterialChannel.ROUGHNESS,
                source_type=ChannelSource.CONSTANT,
                operation=RandomizationOp.SET_VALUE,
                value=0.7,
                policy_id="clone-regression",
            ),
        )
    )
    assert len(report.applied_edits) == 1
    assert not report.skipped_edits
    material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Asset/Mesh")).ComputeBoundMaterial()
    assert str(material.GetPath()) != slot.material_path
    other_material, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Other")).ComputeBoundMaterial()
    assert str(other_material.GetPath()) == slot.material_path
    return material.ComputeSurfaceSource("mdl")[0]


def test_shared_material_edit_preserves_relative_textures(shared_material_scene):
    stage, _, slot, root = shared_material_scene
    source_shader = UsdShade.Shader(stage.GetPrimAtPath(slot.metadata["shader_path"]))
    before = source_shader.GetInput("diffuse_texture").Get()
    clone_shader = edit_shared_material(shared_material_scene)
    for name, relative in (("diffuse_texture", "textures/albedo.png"), ("normalmap_texture", "textures/normal.png")):
        value = clone_shader.GetInput(name).Get()
        assert value.resolvedPath == str(root / relative)
    assert source_shader.GetInput("diffuse_texture").Get() == before
    assert clone_shader.GetInput("reflection_roughness_constant").Get() == pytest.approx(0.7)


def test_clone_preserves_relative_mdl_source(shared_material_scene):
    clone_shader = edit_shared_material(shared_material_scene)
    source = clone_shader.GetSourceAsset("mdl")
    assert source.resolvedPath == str(shared_material_scene[3] / "materials/local.mdl")


def test_clone_preserves_asset_array_paths(shared_material_scene):
    clone_shader = edit_shared_material(shared_material_scene)
    values = clone_shader.GetInput("texture_set").Get()
    root = shared_material_scene[3]
    assert [value.resolvedPath for value in values] == [
        str(root / "textures/albedo.png"),
        str(root / "textures/normal.png"),
        "",
    ]
    assert values[-1].path == ""


def test_clone_preserves_builtin_mdl_module_name(shared_material_scene):
    stage, _, slot, _ = shared_material_scene
    source_shader = UsdShade.Shader(stage.GetPrimAtPath(slot.metadata["shader_path"]))
    source_shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    clone_shader = edit_shared_material(shared_material_scene)
    assert clone_shader.GetSourceAsset("mdl").path == "OmniPBR.mdl"
