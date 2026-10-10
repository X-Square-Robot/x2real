"""Isaac Sim env bootstrap for the env preflight gate.

Source-build Isaac Sim 6 layouts ship ``setup_python_env.sh`` instead of the
older ``setup_conda_env.sh``. A stale conda ``activate.d/setenv.sh`` hook that
only probes the legacy script leaves ``CARB_APP_PATH`` / IsaacSim ``PYTHONPATH``
unset after ``conda activate``, which fails the full-profile preflight gate and
blocks commits. This module re-derives those vars in-process so the gate can
pass without forcing users to regenerate the conda hook (``isaaclab.sh -c``).

``reexec_with_isaac_runtime_env`` extends the same healing to Kit entrypoints:
an installer or shell hook can leave the Isaac path variables unset, so the
next entrypoint cannot import ``isaacsim``. Repairing ``os.environ`` alone is
not enough — the dynamic loader reads ``LD_LIBRARY_PATH`` only at process
start, so ``isaacsim.simulation_app`` would still fail to bind ``libcarb.so``
and leave ``SimulationApp`` as ``None`` — hence the re-exec.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

# Set on the child so a still-broken environment fails with the real ImportError
# instead of re-execing forever.
_REEXEC_GUARD_ENV = "MANAENV_ISAAC_RUNTIME_REEXEC"

# Kit's bundled stdlib root, e.g. ``.../_isaac_sim/kit/python/lib/python3.12``.
_KIT_STDLIB_MARKER = re.compile(r"/kit/python/lib/python3\.\d+")


def _repo_isaaclab_path() -> Path:
    return Path(__file__).resolve().parents[2] / "3rd" / "isaaclabarena" / "submodules" / "IsaacLab"


def _isdir(path: str) -> bool:
    """Indirection seam so tests can model a host without an Isaac install."""
    return os.path.isdir(path)


def _find_isaacsim_spec():
    """Locate ``isaacsim`` without importing it (import needs ``libcarb.so``)."""
    try:
        return find_spec("isaacsim")
    except (ImportError, ValueError):
        return None


def _replace_process(argv: list[str]) -> None:
    """Indirection seam around ``os.execv`` so tests can observe the restart."""
    os.execv(argv[0], argv)


def isaacsim_importable() -> bool:
    """Return whether ``isaacsim`` can be resolved from the current interpreter."""
    return _find_isaacsim_spec() is not None


def resolve_isaac_path() -> str:
    """Return the best-effort Isaac Sim root directory, or ``""`` when none exists."""
    configured = os.environ.get("ISAAC_PATH", "")
    if configured and _isdir(configured):
        return configured

    isaaclab_paths = (
        os.environ.get("ISAACLAB_PATH", ""),
        str(_repo_isaaclab_path()),
    )
    for isaaclab_path in dict.fromkeys(path for path in isaaclab_paths if path):
        candidate = os.path.join(isaaclab_path, "_isaac_sim")
        if _isdir(candidate):
            return os.path.realpath(candidate)

    if _isdir("/isaac-sim"):
        return "/isaac-sim"

    return ""


def _source_isaac_setup_script(isaac_path: str) -> dict[str, str]:
    """Source the Isaac Sim env script and capture the path vars it exports."""
    setup_conda = os.path.join(isaac_path, "setup_conda_env.sh")
    setup_python = os.path.join(isaac_path, "setup_python_env.sh")
    if os.path.isfile(setup_conda):
        cmd = f'source "{setup_conda}" >/dev/null 2>&1 && env'
    elif os.path.isfile(setup_python):
        cmd = f'export ISAAC_PATH="{isaac_path}"; source "$ISAAC_PATH/setup_python_env.sh" >/dev/null 2>&1 && env'
    else:
        return {}

    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return {}

    keys = ("PYTHONPATH", "LD_LIBRARY_PATH", "CARB_APP_PATH", "EXP_PATH", "ISAAC_PATH")
    exported: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key in keys:
            exported[key] = value
    if "PYTHONPATH" in exported:
        exported["PYTHONPATH"] = strip_kit_stdlib_entries(exported["PYTHONPATH"])
    return exported


def strip_kit_stdlib_entries(pythonpath: str) -> str:
    """Drop Kit's bundled cp3xx stdlib from a ``PYTHONPATH`` value.

    ``setup_python_env.sh`` prepends ``kit/python/lib/python3.x`` (and its
    ``site-packages``), which shadows the interpreter's own ``platform.py``.
    Kit's vendored copy cannot parse a conda-forge version string, so
    ``omni.kit.test`` fails to import ("failed to parse CPython sys.version")
    and Kit later crashes. The conda ``activate.d`` hook strips the same
    entries; mirror it so the in-process heal matches a healthy activation.
    """
    return os.pathsep.join(
        entry for entry in pythonpath.split(os.pathsep) if entry and not _KIT_STDLIB_MARKER.search(entry)
    )


def ensure_isaacsim_runtime_env() -> list[str]:
    """Populate missing Isaac Sim launcher/path vars in ``os.environ``.

    Returns notes describing any healing applied. A no-op when the vars are
    already present or no Isaac Sim install can be located, so it is safe to
    call unconditionally before the env preflight check.
    """
    notes: list[str] = []
    isaac_path = resolve_isaac_path()
    if not isaac_path:
        return notes

    os.environ.setdefault("ISAAC_PATH", isaac_path)
    os.environ.setdefault("CARB_APP_PATH", os.path.join(isaac_path, "kit"))
    os.environ.setdefault("EXP_PATH", os.path.join(isaac_path, "apps"))

    pythonpath = os.environ.get("PYTHONPATH", "")
    if "isaacsim" not in pythonpath.lower() and "_isaac_sim" not in pythonpath:
        sourced = _source_isaac_setup_script(isaac_path)
        # The setup script re-exports the inherited value plus the Isaac entries
        # (``export PYTHONPATH=$PYTHONPATH:...``), so overwrite rather than
        # setdefault — otherwise a pre-existing non-Isaac PYTHONPATH blocks the
        # merge and the gate stays red.
        for key in ("PYTHONPATH", "LD_LIBRARY_PATH"):
            if key in sourced:
                os.environ[key] = sourced[key]
        if sourced:
            script = (
                "setup_conda_env.sh"
                if os.path.isfile(os.path.join(isaac_path, "setup_conda_env.sh"))
                else "setup_python_env.sh"
            )
            notes.append(f"Applied Isaac Sim setup script fallback ({script}).")

    if notes:
        notes.insert(0, f"Resolved Isaac Sim root at {isaac_path}.")
    return notes


def reexec_with_isaac_runtime_env() -> list[str]:
    """Heal the Isaac Sim env vars and restart this process when needed.

    A no-op when ``isaacsim`` already resolves, when no Isaac Sim install can be
    located, or when the current process is already a re-exec attempt. Otherwise
    this call does not return: the interpreter is replaced with an identical
    invocation that inherits the repaired ``PYTHONPATH`` / ``LD_LIBRARY_PATH``.

    Call it before importing ``isaaclab.app``: ``AppLauncher`` suppresses the
    ``isaacsim`` ImportError at module import and then dereferences the missing
    global (``NameError: name 'isaacsim' is not defined``).
    """
    notes: list[str] = []
    if isaacsim_importable():
        return notes

    if os.environ.get(_REEXEC_GUARD_ENV):
        notes.append("Isaac Sim runtime env already retried; 'isaacsim' is still not importable.")
        return notes

    notes.extend(ensure_isaacsim_runtime_env())
    if not notes:
        # No Isaac Sim install to heal from — let the caller surface the real
        # ImportError rather than restarting into the same failure.
        return notes

    notes.append("Re-executing to apply the repaired Isaac Sim runtime environment.")
    os.environ[_REEXEC_GUARD_ENV] = "1"
    _replace_process([sys.executable, *sys.argv])
    return notes


def source_checkout_pythonpath(
    repo_root: str | Path,
    *,
    dependency_python: str | Path | None = None,
    inherited: str = "",
) -> str:
    """Build a PYTHONPATH for Kit's bundled Python using this checkout's sources.

    Source-build Isaac Sim ships its own CPython, so invoking it through ``python.sh``
    does not process editable-install ``.pth`` files from the ManaEnv conda env. Add
    the checkout package roots directly. Callers expose the conda dependency site
    separately through ``MANAENV_DEPENDENCY_SITE_PACKAGES``; putting it on
    ``PYTHONPATH`` would place backport modules ahead of Kit's standard library
    (for example an old ``argparse.py`` can hide ``BooleanOptionalAction``).
    """

    root = Path(repo_root).resolve()
    arena_root = root / "3rd" / "isaaclabarena"
    source_root = arena_root / "submodules" / "IsaacLab" / "source"
    entries = [root, arena_root]
    if source_root.is_dir():
        entries.extend(path for path in sorted(source_root.iterdir()) if path.is_dir())

    # Retained in the public signature for existing callers. The dependency
    # interpreter is intentionally not translated into a PYTHONPATH entry; each
    # Kit entrypoint calls ``site.addsitedir`` only after importing stdlib modules.
    _ = dependency_python

    existing = [Path(entry) for entry in inherited.split(os.pathsep) if entry]
    entries.extend(existing)
    unique: list[str] = []
    for entry in entries:
        value = str(entry)
        if value not in unique:
            unique.append(value)
    return os.pathsep.join(unique)


def python_site_packages(executable: str | Path | None) -> str:
    """Return an interpreter's primary site-packages directory, if available."""

    if not executable:
        return ""
    candidate = Path(executable).expanduser()
    if not candidate.is_file():
        return ""
    try:
        result = subprocess.run(
            [str(candidate), "-c", "import site; print(site.getsitepackages()[0])"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip().splitlines()[-1] if result.returncode == 0 and result.stdout.strip() else ""
