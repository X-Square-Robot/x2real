"""Internal validation helpers for the public X2Real ID evaluator.

The supported executable is :mod:`manaenv.scripts.x2real_eval`.  This module
keeps suite parsing, distribution-boundary checks, and asset validation out of
the command entry point without exposing a second launcher.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

# Keep direct source-tree imports bound to this checkout.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from manaenv.assets.resolver import (
    AssetResolutionError,
    asset_root,
    is_asset_reference,
    resolve_asset_path,
)
from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
from manaenv.pipeline.translator.reflective_translator import (
    expand_includes,
    reflect_section,
)

_X2REAL_CONFIG = _REPO_ROOT / "manaenv" / "defination" / "x2real.yaml"
_PUBLIC_CONFIG_SPLIT = "id"
_EXPECTED_CASE_COUNT = 73
_BENCHMARK_ONLY_CAPABILITY = "benchmark"
_PUBLIC_ROBOTS = (
    "artixon_arm6a",
    "quanta_x1",
    "arx_r5",
    "franka_panda",
)
_IMPLICIT_ENV_GRIPPERS = {"franka_panda": "panda"}
_LEGACY_ROBOT_ALIASES = {
    "ex001": "artixon_arm6a",
    "ex001_6r": "artixon_arm6a",
    "x2robot": "artixon_arm6a",
    "ex001wb": "quanta_x1",
    "ex001_wholebody": "quanta_x1",
    "arx": "arx_r5",
    "franka": "franka_panda",
}
_LEGACY_ENV_PREFIXES = {
    "ex001_": "artixon_arm6a_",
    "ex001wb_": "quanta_x1_",
    "arx_default_": "arx_r5_default_",
}
_WHOLEBODY_MOBILE_EMBODIMENT = "quanta_x1"
_WHOLEBODY_MOBILE_GENERATOR = "EX001-WB-G"
_PUBLIC_DEFINITION_DIRS = (
    _REPO_ROOT / "manaenv" / "defination" / "scene",
    _REPO_ROOT / "manaenv" / "defination" / "task",
    _REPO_ROOT / "manaenv" / "defination" / "x2real" / "scene",
    _REPO_ROOT / "manaenv" / "defination" / "x2real" / "task",
)
_PUBLIC_ASSET_POOL_ROOT = _REPO_ROOT / "assets" / "pools"
_PUBLIC_ROBOT_ASSET_FILES = (
    "assets/simready/robots/ex001/v00.04.04/ex001_6r_legacy.usd",
    "assets/simready/robots/ex001/v00.05.05/ex001_6r.usd",
    "assets/simready/robots/ex001/v00.05.05/ex001.usd",
    "assets/art_assets/urdf/optimized/ex001/v00.05.03/ex001_6r_legacy.urdf",
    "assets/robots/arx_r5/arx_r5.usd",
    "assets/robots/franka/franka.usd",
)
_USD_SUFFIXES = {".usd", ".usda", ".usdc"}
_BUILTIN_USD_ASSETS = {
    "OmniGlass.mdl",
    "OmniPBR.mdl",
    "OmniPBR_ClearCoat.mdl",
    "OmniSurface.mdl",
    "OmniSurfacePresets.mdl",
    "core_definitions.blend_colors.png",
    "core_definitions.file_texture.png",
    "nvidia/aux_definitions.mdl",
    "nvidia/core_definitions.mdl",
}

# Expected assembly robot_generator_name for (robot, gripper).
_ROBOT_GENERATOR = {
    ("artixon_arm6a", "H"): "EX001-6R-H",
    ("artixon_arm6a", "G"): "EX001-6R-G",
    ("quanta_x1", "G"): _WHOLEBODY_MOBILE_GENERATOR,
    ("arx_r5", "default"): "ARX-R5-default",
    ("franka_panda", "panda"): "franka-panda",
}
_POLICY_IMPLEMENTATION = {
    "artixon_arm6a": "policy_space",
    "quanta_x1": "policy_space",
    "arx_r5": "policy_space",
    "franka_panda": "policy_space",
}
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_OOD_PATH_TOKEN = re.compile(r"(^|[/_.-])ood([/_.-]|$)", re.IGNORECASE)


@dataclass(frozen=True)
class X2RealCase:
    robot: str
    gripper: str
    task: str
    split: str
    env: str
    robot_generator_name: str = ""
    task_generator_name: str = ""
    scene_generator_name: str = ""
    embodiment: str = ""
    capabilities: tuple[str, ...] = ()
    robot_generator_identity: str = ""


def _as_str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        item = value.strip()
        return (item,) if item else ()
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for raw in value:
            item = str(raw or "").strip()
            if item:
                out.append(item)
        return tuple(out)
    item = str(value).strip()
    return (item,) if item else ()


def _extract_x2real_metadata(assembly: dict[str, Any]) -> dict[str, Any]:
    """Read assembly ``x2real:`` or ``metadata.x2real:`` block."""
    direct = assembly.get("x2real")
    if isinstance(direct, dict):
        return direct
    metadata = assembly.get("metadata")
    if isinstance(metadata, dict):
        nested = metadata.get("x2real")
        if isinstance(nested, dict):
            return nested
    return {}


def is_benchmark_only(case: X2RealCase) -> bool:
    """True when capabilities is exactly ``[benchmark]``."""
    return set(case.capabilities) == {_BENCHMARK_ONLY_CAPABILITY}


def _check_mobile_identity(case: X2RealCase, errors: list[str]) -> None:
    """Validate the frozen public whole-body mobile identity."""
    if case.robot != "quanta_x1":
        return
    if case.embodiment != _WHOLEBODY_MOBILE_EMBODIMENT:
        errors.append(
            f"{case.env}: metadata.x2real.embodiment must be "
            f"{_WHOLEBODY_MOBILE_EMBODIMENT!r} (got {case.embodiment!r})"
        )
    if not is_benchmark_only(case):
        errors.append(
            f"{case.env}: metadata.x2real.capabilities must be "
            f"[{_BENCHMARK_ONLY_CAPABILITY!r}] (got {list(case.capabilities)!r})"
        )
    if case.robot_generator_identity != _WHOLEBODY_MOBILE_GENERATOR:
        errors.append(
            f"{case.env}: metadata.x2real.robot_generator_identity must be "
            f"{_WHOLEBODY_MOBILE_GENERATOR!r} (got {case.robot_generator_identity!r})"
        )


def canonical_robot_name(name: str) -> str:
    """Return the paper-facing robot key for a public or legacy selector."""

    normalized = str(name or "").strip().lower().replace("-", "_")
    return _LEGACY_ROBOT_ALIASES.get(normalized, normalized)


def canonical_env_name(env: str) -> str:
    """Translate a legacy public environment ID to its paper-facing ID."""

    normalized = str(env or "").strip()
    for legacy, canonical in _LEGACY_ENV_PREFIXES.items():
        if normalized.startswith(legacy):
            return canonical + normalized[len(legacy) :]
    return normalized


def parse_env_name(env: str) -> X2RealCase:
    """Parse one canonical paper-facing suite environment name."""
    if not env.endswith("_env"):
        raise ValueError(f"env name must end with _env: {env!r}")
    body = env[: -len("_env")]
    robot = next(
        (
            candidate
            for candidate in sorted(_PUBLIC_ROBOTS, key=len, reverse=True)
            if body.startswith(f"{candidate}_")
        ),
        None,
    )
    if robot is None:
        raise ValueError(
            f"env name must start with a paper robot key "
            f"({', '.join(_PUBLIC_ROBOTS)}; got {env!r})"
        )
    parts = body[len(robot) + 1 :].split("_")
    implicit_gripper = _IMPLICIT_ENV_GRIPPERS.get(robot)
    minimum_parts = 2 if implicit_gripper else 3
    if len(parts) < minimum_parts:
        raise ValueError(f"env name must be robot_gripper_task_split_env (got {env!r})")
    split = parts[-1].lower()
    if split != _PUBLIC_CONFIG_SPLIT:
        raise ValueError(f"env split must be id (got {split!r} in {env!r})")
    gripper = implicit_gripper or parts[0]
    task = "_".join(parts[:-1] if implicit_gripper else parts[1:-1])
    if not task:
        raise ValueError(f"env name missing task segment: {env!r}")
    return X2RealCase(robot=robot, gripper=gripper, task=task, split=split, env=env)


def load_x2real_document(path: Path | None = None) -> dict[str, Any]:
    cfg = path or _X2REAL_CONFIG
    if not cfg.is_file():
        raise FileNotFoundError(f"x2real config not found: {cfg}")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"x2real config is not a mapping: {cfg}")
    expanded = expand_includes(data, cfg.resolve().parent)
    if not isinstance(expanded, dict):
        raise TypeError(f"expanded x2real config is not a mapping: {cfg}")
    return expanded


def load_cases(path: Path | None = None) -> list[X2RealCase]:
    data = load_x2real_document(path)
    cases: list[X2RealCase] = []
    for order in data.get("assembly_orders") or []:
        if not isinstance(order, dict):
            continue
        ag = order.get("assembly_generator") or order
        if not isinstance(ag, dict):
            continue
        name = str(ag.get("name") or "").strip()
        if not name:
            continue
        parsed = parse_env_name(name)
        meta = _extract_x2real_metadata(ag)
        identity = str(meta.get("robot_generator_identity") or "").strip()
        cases.append(
            X2RealCase(
                robot=parsed.robot,
                gripper=parsed.gripper,
                task=parsed.task,
                split=parsed.split,
                env=parsed.env,
                robot_generator_name=str(ag.get("robot_generator_name") or ""),
                task_generator_name=str(ag.get("task_generator_name") or ""),
                scene_generator_name=str(ag.get("scene_generator_name") or ""),
                embodiment=str(meta.get("embodiment") or "").strip(),
                capabilities=_as_str_tuple(meta.get("capabilities")),
                robot_generator_identity=identity,
            )
        )
    return cases


def _robot_orders_index(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for order in data.get("robot_orders") or []:
        if not isinstance(order, dict):
            continue
        rg = order.get("robot_generator") or order
        if not isinstance(rg, dict):
            continue
        name = str(rg.get("name") or "").strip()
        if name:
            index[name] = rg
    return index


def _walk_strings(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        out: list[str] = []
        for value in node.values():
            out.extend(_walk_strings(value))
        return out
    if isinstance(node, list):
        out: list[str] = []
        for value in node:
            out.extend(_walk_strings(value))
        return out
    return []


def _walk_asset_references(node: Any) -> set[str]:
    if isinstance(node, str) and is_asset_reference(node):
        return {node}
    if isinstance(node, dict):
        out: set[str] = set()
        for value in node.values():
            out.update(_walk_asset_references(value))
        return out
    if isinstance(node, list):
        out: set[str] = set()
        for value in node:
            out.update(_walk_asset_references(value))
        return out
    return set()


def _scene_asset_inventory(data: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    """Return concrete registry names, render-setting names, and asset modules."""

    from manaenv.pipeline.generators.scene_layout_solver.schema_vnext import (
        VNextSceneSpec,
    )

    asset_names: set[str] = set()
    render_settings: set[str] = set()
    asset_modules: set[str] = set()
    for order in data.get("scene_orders") or []:
        if not isinstance(order, dict):
            continue
        scene = VNextSceneSpec.from_dict(order.get("scene_generator") or order)
        selectors = [scene.background.asset, *(obj.asset for obj in scene.objects.values())]
        for selector in selectors:
            if selector.asset_id:
                asset_names.add(selector.asset_id)
            for candidate in selector.candidates:
                asset_names.update(candidate.assets)
        render_asset = str(scene.render_settings.get("asset") or scene.render_settings.get("tags") or "").strip()
        if render_asset:
            render_settings.add(render_asset)
        if scene.background.asset_module:
            asset_modules.add(scene.background.asset_module)
    return asset_names, render_settings, asset_modules


def _module_local_assets(module_name: str) -> dict[str, str]:
    """Extract registered names and repository asset paths without importing Kit."""

    module_path = _REPO_ROOT / Path(*module_name.split(".")).with_suffix(".py")
    if not module_path.is_file():
        raise FileNotFoundError(f"asset module not found: {module_name} ({module_path})")
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    aliases: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        name = ""
        reference = ""
        for statement in node.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id == "name" and isinstance(statement.value, ast.Constant):
                name = str(statement.value.value or "").strip()
            if target.id != "usd_path" or not isinstance(statement.value, ast.Call):
                continue
            function = statement.value.func
            function_name = function.id if isinstance(function, ast.Name) else getattr(function, "attr", "")
            if function_name != "resolve_runtime_asset_reference" or not statement.value.args:
                continue
            argument = statement.value.args[0]
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                reference = argument.value.strip()
        if name and reference:
            aliases[name] = reference
    return aliases


def _is_builtin_usd_asset(reference: str) -> bool:
    normalized = reference.strip().replace("\\", "/")
    if not normalized:
        return True
    return normalized in _BUILTIN_USD_ASSETS


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute normalized path without dereferencing symlinks."""

    return Path(os.path.abspath(path))


def _inside_asset_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _local_dependency_path(reference: str, *, parent: Path, root: Path) -> Path:
    normalized = reference.replace("\\", "/")
    if urlparse(normalized).scheme:
        raise ValueError(f"contains non-local dependency {reference!r}")
    dependency = _lexical_absolute(parent / normalized)
    if not _inside_asset_root(dependency, root):
        raise ValueError(f"contains dependency outside asset root {reference!r}")
    return dependency


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _record_asset_path(reference: str, *, output: list[str]) -> str:
    output.append(reference)
    return reference


def _check_local_usd_closure(root_paths: set[Path], *, root: Path | None = None) -> list[str]:
    """Check every external USD/texture dependency without starting Kit."""

    try:
        from pxr import Sdf, Tf, UsdUtils
    except ImportError as exc:
        return [f"USD dependency validation requires pxr: {exc}"]

    errors: list[str] = []
    bundle_root = _lexical_absolute(root or asset_root())
    pending: list[Path] = []
    for candidate in sorted(root_paths):
        path = _lexical_absolute(candidate)
        if path.suffix.lower() not in _USD_SUFFIXES:
            continue
        if not _inside_asset_root(path, bundle_root):
            errors.append(f"local USD root is outside asset root: {_display_path(path)}")
            continue
        pending.append(path)
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        try:
            layer = Sdf.Layer.FindOrOpen(str(path))
        except Tf.ErrorException as exc:
            errors.append(f"cannot open local USD layer {_display_path(path)}: {exc}")
            continue
        if layer is None:
            errors.append(f"cannot open local USD layer: {_display_path(path)}")
            continue
        references: list[str] = []
        try:
            UsdUtils.ModifyAssetPaths(
                layer,
                partial(_record_asset_path, output=references),
            )
        except Tf.ErrorException as exc:
            errors.append(f"cannot inspect local USD layer {_display_path(path)}: {exc}")
            continue
        for raw_reference in references:
            reference = str(raw_reference or "").strip()
            if _is_builtin_usd_asset(reference):
                continue
            try:
                dependency = _local_dependency_path(reference, parent=path.parent, root=bundle_root)
            except ValueError as exc:
                errors.append(f"local USD layer {exc}: {_display_path(path)}")
                continue
            if not dependency.is_file():
                errors.append(f"missing USD dependency {reference!r}: {_display_path(path)}")
                continue
            if dependency.suffix.lower() in _USD_SUFFIXES:
                pending.append(dependency)
    return errors


def _check_local_urdf_closure(root_paths: set[Path], *, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    bundle_root = _lexical_absolute(root or asset_root())
    urdf_paths: list[Path] = []
    for candidate in sorted(root_paths):
        path = _lexical_absolute(candidate)
        if path.suffix.lower() != ".urdf":
            continue
        if not _inside_asset_root(path, bundle_root):
            errors.append(f"local URDF root is outside asset root: {_display_path(path)}")
            continue
        urdf_paths.append(path)
    for path in urdf_paths:
        try:
            xml_root = ET.parse(path).getroot()
        except (OSError, ET.ParseError) as exc:
            errors.append(f"cannot parse local URDF {_display_path(path)}: {exc}")
            continue
        for mesh in xml_root.iter("mesh"):
            reference = str(mesh.get("filename") or "").strip()
            if not reference:
                continue
            try:
                dependency = _local_dependency_path(reference, parent=path.parent, root=bundle_root)
            except ValueError as exc:
                errors.append(f"local URDF {exc}: {_display_path(path)}")
                continue
            if not dependency.is_file():
                errors.append(f"missing URDF mesh {reference!r}: {_display_path(path)}")
    return errors


def check_asset_references(path: Path | None = None) -> list[str]:
    """Return missing local assets and dependency-closure failures.

    A grasp manifest is an optional index: the runtime can resolve per-asset
    ``.npy`` labels directly from ``assets/grasp_labels``. Scene pools are
    expanded to their concrete asset names, generic package ``asset.usd`` files
    resolve by parent directory, and task-module paths are inspected statically.
    """

    cfg = path or _X2REAL_CONFIG
    data = load_x2real_document(cfg)
    errors: list[str] = []
    root_paths: set[Path] = set()
    for reference in sorted(_walk_asset_references(data)):
        if reference in {"assets/grasp_labels/manifest.yaml", "asset://grasp_labels/manifest.yaml"}:
            continue
        try:
            root_paths.add(resolve_asset_path(reference))
        except (AssetResolutionError, ValueError) as exc:
            errors.append(str(exc))

    try:
        asset_names, render_settings, asset_modules = _scene_asset_inventory(data)
    except (OSError, TypeError, ValueError) as exc:
        return [*errors, f"failed to expand scene asset inventory: {exc}"]

    aliases: dict[str, str] = {}
    for module_name in sorted(asset_modules):
        try:
            module_aliases = _module_local_assets(module_name)
        except (OSError, SyntaxError) as exc:
            errors.append(str(exc))
            continue
        aliases.update(module_aliases)

    from manaenv.assets.local_discovery import resolve_usd_path

    for asset_name in sorted(asset_names - set(aliases)):
        try:
            root_paths.add(Path(resolve_usd_path(asset_name)))
        except KeyError as exc:
            errors.append(str(exc))
    for asset_name, reference in sorted(aliases.items()):
        if asset_name not in asset_names:
            continue
        try:
            root_paths.add(resolve_asset_path(reference))
        except (AssetResolutionError, ValueError) as exc:
            errors.append(str(exc))
    for render_asset in sorted(render_settings):
        try:
            root_paths.add(Path(resolve_usd_path(f"{render_asset}.settings")))
        except KeyError as exc:
            errors.append(str(exc))
    for reference in _PUBLIC_ROBOT_ASSET_FILES:
        try:
            root_paths.add(resolve_asset_path(reference))
        except (AssetResolutionError, ValueError) as exc:
            errors.append(str(exc))

    errors.extend(_check_local_usd_closure(root_paths))
    errors.extend(_check_local_urdf_closure(root_paths))
    return errors


def _check_public_runtime(data: dict[str, Any]) -> list[str]:
    """Enforce the PhysX-only runtime contract without importing Isaac Lab."""

    pipeline = data.get("pipeline")
    runtime = pipeline.get("pipeline_runtime") if isinstance(pipeline, dict) else None
    if not isinstance(runtime, dict):
        return ["public suite must declare pipeline.pipeline_runtime"]

    errors: list[str] = []
    if runtime.get("solver") != "physx":
        errors.append("public suite pipeline_runtime.solver must be exactly 'physx'")
    removed_fields = sorted({"brick_kappa", "num_substeps"}.intersection(runtime))
    if removed_fields:
        errors.append(f"public suite contains removed runtime field(s): {', '.join(removed_fields)}")
    return errors


def _check_rollout_schema(data: dict[str, Any], cfg: Path) -> list[str]:
    """Reflect public rollout orders before Kit startup catches schema drift."""

    try:
        bootstrap_discovery()
        reflect_section(
            "rollout",
            data.get("rollout") or [],
            base_dir=cfg.resolve().parent,
            context="public rollout schema",
        )
    except (
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        return [f"public rollout schema validation failed: {exc}"]
    return []


def _check_distribution_boundary() -> list[str]:
    """Reject explicit OOD definitions and held-out pool partitions."""

    errors: list[str] = []
    for root in _PUBLIC_DEFINITION_DIRS:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_file() and _OOD_PATH_TOKEN.search(relative):
                errors.append(f"public definition tree contains OOD file: {path.relative_to(_REPO_ROOT)}")
    if _PUBLIC_ASSET_POOL_ROOT.is_dir():
        for path in sorted(_PUBLIC_ASSET_POOL_ROOT.rglob("*")):
            relative = path.relative_to(_PUBLIC_ASSET_POOL_ROOT).as_posix()
            if path.is_file() and _OOD_PATH_TOKEN.search(relative):
                errors.append(f"public asset pool contains OOD file: {path.relative_to(_REPO_ROOT)}")
        for path in sorted(_PUBLIC_ASSET_POOL_ROOT.rglob("*.json")):
            try:
                pool = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                errors.append(f"invalid asset pool {path.relative_to(_REPO_ROOT)}: {exc}")
                continue
            if isinstance(pool, dict) and ("ood_window" in pool or "heldout" in pool or "held_out" in pool):
                errors.append(f"public asset pool declares a held-out partition: {path.relative_to(_REPO_ROOT)}")
    return errors


def check_suite(path: Path | None = None) -> list[str]:
    """Return public benchmark suite consistency errors (empty means OK)."""
    cfg = path or _X2REAL_CONFIG
    data = load_x2real_document(cfg)
    robots = _robot_orders_index(data)
    errors = (
        _check_public_runtime(data)
        + _check_rollout_schema(data, cfg)
        + _check_distribution_boundary()
    )
    rollouts_by_env: dict[str, list[dict[str, Any]]] = {}
    for ro in data.get("rollout") or []:
        if not isinstance(ro, dict):
            continue
        env = str(ro.get("env_ref") or "").strip()
        if not env:
            continue
        rollouts_by_env.setdefault(env, []).append(ro)
    assembly_envs: set[str] = set()

    try:
        cases = load_cases(cfg)
    except (OSError, ValueError) as exc:
        return [str(exc)]
    if len(cases) != _EXPECTED_CASE_COUNT:
        errors.append(f"public suite must contain exactly {_EXPECTED_CASE_COUNT} assemblies, got {len(cases)}")
    if any(case.split != "id" for case in cases):
        errors.append("public suite must contain only ID assemblies")
    if any("gendata" in value for value in _walk_strings(data)):
        errors.append("public suite must not contain gendata rollouts")
    if any("teleop" in value.lower() for value in _walk_strings(data)):
        errors.append("public suite must not contain teleoperation configuration")
    if any("omniverse://" in value for value in _walk_strings(data) if isinstance(value, str)):
        errors.append("public suite contains a remote omniverse:// reference")
    if len(data.get("output") or []) != 1:
        errors.append("public suite must declare exactly one output order")
    else:
        output = data["output"][0]
        if not isinstance(output, dict) or output.get("type") != "benchmark":
            errors.append("public output must be a benchmark formatter")

    for case in cases:
        assembly_envs.add(case.env)
        expected = _ROBOT_GENERATOR.get((case.robot, case.gripper))
        if expected is None:
            errors.append(f"{case.env}: unknown robot/gripper pair " f"({case.robot!r}, {case.gripper!r})")
        elif case.robot_generator_name and case.robot_generator_name != expected:
            errors.append(f"{case.env}: robot_generator_name={case.robot_generator_name!r} " f"expected {expected!r}")
        if case.robot_generator_name and case.robot_generator_name not in robots:
            errors.append(
                f"{case.env}: robot_generator_name={case.robot_generator_name!r} " "missing from robot_orders"
            )
        elif case.robot_generator_name:
            rg = robots[case.robot_generator_name]
            variant = rg.get("gripper_variant")
            if case.gripper in ("H", "G") and variant != case.gripper:
                errors.append(
                    f"{case.env}: robot {case.robot_generator_name!r} "
                    f"gripper_variant={variant!r} expected {case.gripper!r}"
                )
        if case.split == "id" and case.scene_generator_name:
            scene = case.scene_generator_name
            if scene.endswith("_ood") or "_ood_" in scene:
                errors.append(f"{case.env}: ID assembly uses OOD scene {scene!r}")
        _check_mobile_identity(case, errors)
        orders = rollouts_by_env.get(case.env) or []
        if not orders:
            errors.append(f"{case.env}: missing matching rollout.env_ref")
            continue
        if len(orders) != 1:
            errors.append(f"{case.env}: expected exactly one benchmark rollout, got {len(orders)}")
        for ro in orders:
            if "num_demos" in ro:
                errors.append(
                    f"{case.env}: public benchmark rollout must not declare num_demos"
                )
            if "modes" in ro:
                errors.append(f"{case.env}: public rollout must not declare legacy modes")
            planner = ro.get("planner") or {}
            if not (isinstance(planner, dict) and "PolicyPlanner" in planner):
                errors.append(f"{case.env}: benchmark rollout must use PolicyPlanner")
            policy_cfg = (planner.get("PolicyPlanner") or {}).get("policy_cfg") or {}
            implementation = policy_cfg.get("implementation") or {}
            expected_implementation = _POLICY_IMPLEMENTATION.get(case.robot)
            if expected_implementation is None:
                errors.append(f"{case.env}: no policy implementation contract for robot {case.robot!r}")
            elif not isinstance(implementation, dict) or set(implementation) != {expected_implementation}:
                errors.append(
                    f"{case.env}: policy implementation must be exactly "
                    f"{expected_implementation!r}, got {sorted(implementation) if isinstance(implementation, dict) else implementation!r}"
                )

            remote = policy_cfg.get("remote")
            if isinstance(remote, dict) and remote:
                remote_value = next(iter(remote.values()), {})
                if (
                    isinstance(remote_value, dict)
                    and str(remote_value.get("server_address") or "") not in _LOOPBACK_HOSTS
                ):
                    errors.append(f"{case.env}: policy server must default to a loopback address")
            if isinstance(implementation, dict) and "policy_space" in implementation:
                policy_space = implementation.get("policy_space") or {}
                endpoint = str(policy_space.get("endpoint") or "") if isinstance(policy_space, dict) else ""
                if urlparse(endpoint).hostname not in _LOOPBACK_HOSTS:
                    errors.append(f"{case.env}: PolicySpace endpoint must default to a loopback address")

    for env in sorted(set(rollouts_by_env) - assembly_envs):
        if env:
            errors.append(f"rollout env_ref {env!r} has no assembly_orders entry")
    return errors


def validate_runtime_passthrough(argv: list[str]) -> None:
    """Reject removed backend/viewer overrides before a dry-run or exec."""

    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, inline_value = token.partition("=")
        if option not in {"--presets", "--visualizer", "--viz"}:
            index += 1
            continue

        if separator:
            value = inline_value
        else:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError(f"{option} requires a value")
            index += 1
            value = argv[index]

        if option == "--presets" and value.strip().lower() != "physx":
            raise ValueError("the public runtime supports only '--presets physx'")
        if option in {"--visualizer", "--viz"}:
            visualizers = {item.strip().lower() for item in value.split(",") if item.strip()}
            if "newton" in visualizers:
                raise ValueError("the Newton visualizer is not available in the public PhysX runtime")
        index += 1
