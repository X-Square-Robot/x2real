"""Safe in-place pose updates for live USD assets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def asset_supports_sim_root_pose(asset: Any) -> bool:
    """Return whether an asset exposes a simulation-owned root-pose API."""
    return callable(getattr(asset, "write_root_pose_to_sim_index", None)) or (
        callable(getattr(asset, "write_root_pose_to_sim", None))
        and callable(getattr(asset, "write_root_velocity_to_sim", None))
    )


def should_sync_runtime_pose_to_usd(asset: Any) -> bool:
    """Return whether reset-time pose writes may safely target USD.

    PhysX/Fabric-backed roots are authoritative in the simulation tensor API.
    Mirroring the reset pose into USD creates a competing transform source on
    the first post-reset tick, so only assets without a simulation root-pose
    interface use the USD bridge.
    """
    return not asset_supports_sim_root_pose(asset)


def should_author_pose_owned_visibility_to_usd(asset: Any) -> bool:
    """Return whether pose-owned activation may also author USD visibility.

    PhysX/Fabric roots stay authored visible because their active/inactive state
    is already represented by the simulation-owned root pose.
    """
    return should_sync_runtime_pose_to_usd(asset)


def set_existing_pose_xform_ops(
    xformable: Any,
    *,
    position_xyz: Sequence[float],
    quaternion_xyzw: Sequence[float],
    gf: Any,
    usd_geom: Any,
) -> bool:
    """Update an existing pose stack from ManaEnv's XYZW convention."""
    # Clearing or adding ops on a tensor-managed rigid prim can make PhysX
    # rebuild nested bodies and permanently invalidate the shared SimulationView.
    ops = xformable.GetOrderedXformOps()
    translate_op = next(
        (
            op
            for op in ops
            if op.GetOpType() == usd_geom.XformOp.TypeTranslate and op.GetOpName() == "xformOp:translate"
        ),
        None,
    )
    orient_op = next(
        (op for op in ops if op.GetOpType() == usd_geom.XformOp.TypeOrient and op.GetOpName() == "xformOp:orient"),
        None,
    )
    if translate_op is None or orient_op is None:
        transform_type = getattr(usd_geom.XformOp, "TypeTransform", None)
        transform_op = next(
            (op for op in ops if op.GetOpType() == transform_type and op.GetOpName() == "xformOp:transform"),
            None,
        )
        if transform_op is None:
            return False

        current_matrix = transform_op.Get()
        if current_matrix is None:
            return False

        px, py, pz = position_xyz
        qx, qy, qz, qw = quaternion_xyzw
        transform = gf.Transform(current_matrix)
        transform.SetTranslation(gf.Vec3d(px, py, pz))
        transform.SetRotation(gf.Rotation(gf.Quatd(qw, qx, qy, qz)))
        return bool(transform_op.Set(transform.GetMatrix()))

    float_precision = getattr(usd_geom.XformOp, "PrecisionFloat", None)
    half_precision = getattr(usd_geom.XformOp, "PrecisionHalf", None)
    translate_precision = translate_op.GetPrecision()
    orient_precision = orient_op.GetPrecision()
    px, py, pz = position_xyz
    qx, qy, qz, qw = quaternion_xyzw

    if half_precision is not None and translate_precision == half_precision:
        translate_value = gf.Vec3h(px, py, pz)
    elif float_precision is not None and translate_precision == float_precision:
        translate_value = gf.Vec3f(px, py, pz)
    else:
        translate_value = gf.Vec3d(px, py, pz)

    if half_precision is not None and orient_precision == half_precision:
        orient_value = gf.Quath(qw, qx, qy, qz)
    elif float_precision is not None and orient_precision == float_precision:
        orient_value = gf.Quatf(qw, qx, qy, qz)
    else:
        orient_value = gf.Quatd(qw, qx, qy, qz)

    translate_written = bool(translate_op.Set(translate_value))
    orient_written = bool(orient_op.Set(orient_value))
    return translate_written and orient_written
