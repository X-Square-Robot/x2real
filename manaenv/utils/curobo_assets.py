"""Helpers for resolving cuRobo config asset paths.

cuRobo resolves relative ``kinematics.collision_spheres`` / ``world_config_file``
paths against its own package config directory, which breaks repo-local YAMLs.
These helpers normalise such paths against a base directory before the robot
config is handed to cuRobo.

This module is intentionally Isaac-Sim-light so it can be loaded outside Isaac
Sim for unit tests / static checks.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "curobo_robot_configs_path",
    "resolve_curobo_path",
    "resolve_world_config_file",
]


def curobo_robot_configs_path() -> Path | None:
    """Return the installed cuRobo ``content/configs/robot`` directory.

    Resolves the bundled robot-config root from whatever cuRobo package is
    importable (pip / editable install), so YAMLs can reference cuRobo-shipped
    assets (e.g. ``spheres/franka_mesh.yml``) by their package-relative path
    instead of a machine-specific absolute path. Returns ``None`` when cuRobo
    is not importable (e.g. unit tests / static checks outside Isaac Sim).
    """
    try:
        from curobo.util_file import get_robot_configs_path
    except Exception:  # noqa: BLE001 - cuRobo optional outside Isaac Sim
        return None
    try:
        return Path(get_robot_configs_path())
    except Exception:  # noqa: BLE001 - defensive: never block path resolution
        return None


def resolve_curobo_path(path_value: str | None, *, base_dir: Path | None = None) -> str | None:
    """Resolve cuRobo config asset paths to absolute filesystem paths.

    cuRobo resolves relative ``kinematics.collision_spheres`` against its own
    package config directory, which breaks repo-local YAMLs.  We normalise such
    paths against ``base_dir`` (or CWD) before writing the temp robot config.
    """
    if not path_value or not isinstance(path_value, str):
        return path_value

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    search_roots: list[Path] = []
    if base_dir is not None:
        search_roots.append(base_dir)
    search_roots.append(Path.cwd())
    # Fall back to the cuRobo package's bundled robot-config dir so
    # package-relative assets (e.g. ``spheres/franka_mesh.yml``) resolve to the
    # installed cuRobo location instead of requiring an absolute path.
    curobo_root = curobo_robot_configs_path()
    if curobo_root is not None:
        search_roots.append(curobo_root)

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    root = search_roots[0] if search_roots else Path.cwd()
    return str((root / candidate).resolve())


def resolve_world_config_file(path_value: str | None, *, base_dir: Path | None = None) -> str | None:
    """Resolve ``world_config_file`` while preserving cuRobo-bundled filenames.

    Bare filenames like ``collision_table.yml`` (which live under cuRobo's
    content directory) are passed through unchanged so cuRobo can resolve them
    against ``get_world_configs_path()`` itself.
    """
    if not path_value or not isinstance(path_value, str):
        return path_value

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve()) if candidate.exists() else path_value

    search_roots: list[Path] = []
    if base_dir is not None:
        search_roots.append(base_dir)
    search_roots.append(Path.cwd())

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    return path_value
