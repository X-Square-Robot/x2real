"""Small USD authoring helpers for ManaEnv-owned PhysX runtime behavior."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_PARTICLE_API_NAMES = (
    "PhysxParticleSystemAPI",
    "PhysxParticleClothAPI",
    "PhysxParticleSetAPI",
)


class ManaPhysxContractError(ValueError):
    """Raised when a PhysX runtime setting is incompatible with the scene."""


def is_cuda_physx_device(device: Any) -> bool:
    """True when ``device`` selects a CUDA PhysX pipeline (enables GPU dynamics)."""
    return str(device or "").strip().lower().startswith("cuda")


def _iter_tag_names(raw: Any) -> Iterable[str]:
    if raw is None:
        return
    if isinstance(raw, str):
        text = raw.strip()
        if text:
            yield text
        return
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            key_text = str(key).strip()
            if key_text:
                yield key_text
            yield from _iter_tag_names(value)
        return
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        for entry in raw:
            if isinstance(entry, (tuple, list)) and len(entry) >= 2 and entry[0] == "class":
                text = str(entry[1]).strip()
                if text:
                    yield text
                continue
            name = getattr(entry, "name", None)
            if name is not None:
                text = str(name).strip()
                if text:
                    yield text
                continue
            yield from _iter_tag_names(entry)
        return


def collect_cloth_semantic_evidence(scene: Any) -> list[str]:
    """Return evidence strings when scene assets declare the ``cloth`` semantic tag.

    Particle-cloth tasks (fold_towel / towel_to_basket / tshirt_to_basket) author
    ``semantic_tags: [cloth, ...]``. That tag is the pre-spawn signal that PhysX
    particles will be loaded and therefore require GPU dynamics.
    """
    assets = getattr(scene, "assets", None)
    if not isinstance(assets, Mapping):
        return []
    evidence: list[str] = []
    for name, asset in assets.items():
        tag_names: set[str] = set()
        for source in (
            getattr(asset, "semantic_tags", None),
            getattr(asset, "tags", None),
            getattr(getattr(getattr(asset, "object_cfg", None), "spawn", None), "semantic_tags", None),
        ):
            tag_names.update(_iter_tag_names(source))
        if "cloth" in tag_names:
            evidence.append(f"scene.assets[{name!r}] semantic tag 'cloth'")
    return evidence


def find_particle_feature_prim_paths(stage: Any) -> list[str]:
    """Return prim paths that carry PhysX particle APIs on a composed stage."""
    if stage is None:
        return []
    paths: list[str] = []
    traverse = getattr(stage, "TraverseAll", None) or getattr(stage, "Traverse", None)
    if traverse is None:
        return []
    for prim in traverse():
        schemas = {str(schema) for schema in (prim.GetAppliedSchemas() if hasattr(prim, "GetAppliedSchemas") else [])}
        has_api = False
        for api_name in _PARTICLE_API_NAMES:
            checker = getattr(prim, "HasAPI", None)
            if callable(checker) and checker(api_name):
                has_api = True
                break
            if api_name in schemas:
                has_api = True
                break
        if has_api:
            path = prim.GetPath() if hasattr(prim, "GetPath") else None
            paths.append(str(path) if path is not None else str(prim))
    return paths


def require_gpu_dynamics_for_particles(
    *,
    device: Any,
    evidence: Sequence[str],
) -> None:
    """Reject CPU PhysX when the scene/stage needs the particles feature.

    PhysX only supports particles (including cloth) with GPU dynamics enabled.
    ManaEnv enables GPU dynamics when ``pipeline_runtime.device`` starts with
    ``cuda``.
    """
    if not evidence:
        return
    if is_cuda_physx_device(device):
        return
    shown = ", ".join(str(item) for item in evidence[:8])
    suffix = " ..." if len(evidence) > 8 else ""
    raise ManaPhysxContractError(
        "PhysX particles/cloth require GPU dynamics "
        f"(got device={device!r}). Set pipeline_runtime.device to a CUDA "
        f"device (for example 'cuda:0'). Evidence: {shown}{suffix}"
    )


def enable_ccd_for_all_rigid_bodies(
    stage: Any,
    *,
    swept: bool,
    speculative: bool,
    usd_physics: Any = None,
    physx_schema: Any = None,
) -> int:
    """Author selected CCD modes on every non-kinematic prim with ``RigidBodyAPI``.

    ``TraverseAll`` includes inactive prims, unlike the default stage traversal.
    PhysX does not support CCD on kinematic rigid bodies, so those prims are
    deliberately left unchanged.

    Args:
        stage: Fully composed USD stage.
        swept: Whether to enable sweep-based CCD for linear motion.
        speculative: Whether to enable speculative CCD for linear and angular motion.
        usd_physics: Optional injected ``pxr.UsdPhysics`` module for tests.
        physx_schema: Optional injected ``pxr.PhysxSchema`` module for tests.

    Returns:
        Number of rigid-body prims updated.
    """
    if stage is None:
        raise ValueError("stage is required to enable rigid-body CCD")
    if not swept and not speculative:
        return 0
    if usd_physics is None or physx_schema is None:
        from pxr import PhysxSchema, UsdPhysics

        usd_physics = UsdPhysics if usd_physics is None else usd_physics
        physx_schema = PhysxSchema if physx_schema is None else physx_schema

    applied = 0
    for prim in stage.TraverseAll():
        if not prim.HasAPI(usd_physics.RigidBodyAPI):
            continue
        rigid_body_api = usd_physics.RigidBodyAPI(prim)
        kinematic_attr = rigid_body_api.GetKinematicEnabledAttr()
        if kinematic_attr and bool(kinematic_attr.Get()):
            continue
        physx_api = physx_schema.PhysxRigidBodyAPI(prim)
        if not physx_api:
            physx_api = physx_schema.PhysxRigidBodyAPI.Apply(prim)
        if not physx_api:
            raise RuntimeError(f"failed to apply PhysxRigidBodyAPI to {prim.GetPath()}")
        if swept:
            enable_ccd_attr = physx_api.CreateEnableCCDAttr(True)
            if not enable_ccd_attr or enable_ccd_attr.Set(True) is False:
                raise RuntimeError(f"failed to enable swept CCD on {prim.GetPath()}")
        if speculative:
            enable_speculative_ccd_attr = physx_api.CreateEnableSpeculativeCCDAttr(True)
            if not enable_speculative_ccd_attr or enable_speculative_ccd_attr.Set(True) is False:
                raise RuntimeError(f"failed to enable speculative CCD on {prim.GetPath()}")
        applied += 1
    return applied


def enable_swept_ccd_for_all_rigid_bodies(
    stage: Any,
    *,
    usd_physics: Any = None,
    physx_schema: Any = None,
) -> int:
    """Author swept CCD on every composed USD prim that has ``RigidBodyAPI``."""
    return enable_ccd_for_all_rigid_bodies(
        stage,
        swept=True,
        speculative=False,
        usd_physics=usd_physics,
        physx_schema=physx_schema,
    )
