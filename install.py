"""Install ManaEnv and its public-release source dependencies."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
X2REAL_BRANCH = "x2real"
POLICY_SPACE_BRANCH = "main"
# Isaac Sim runtime installation remains NVIDIA pip; the gitlink is provenance only.
ISAACSIM_VERSION = "6.0.1.0"
ISAACSIM_INSTALL_METHOD = "pip"
NEWTON_PIP_SPEC = "newton[importers] @ git+https://github.com/X-Square-Robot/Newton.git@x2real"
NEWTON_SCHEMAS_PIP_SPEC = "newton-usd-schemas @ git+https://github.com/X-Square-Robot/Newton.git@x2real#subdirectory=extern/newton-usd-schemas"
DEFAULT_ISAACLAB_URL = "https://github.com/X-Square-Robot/IsaacLab.git"
DEFAULT_ISAACSIM_URL = "https://github.com/X-Square-Robot/IsaacSim.git"
DEFAULT_ARENA_URL = "https://github.com/X-Square-Robot/IsaacLabArena.git"
DEFAULT_POLICY_SPACE_URL = "https://github.com/X-Square-Robot/policy-space.git"


class Installer:
    def __init__(self, *, dry_run: bool) -> None:
        self.dry_run = dry_run

    @staticmethod
    def _display(args: tuple[object, ...]) -> str:
        return shlex.join(str(arg) for arg in args)

    def run(self, *args: object, cwd: Path | None = None) -> None:
        print(f"+ {self._display(args)}")
        if not self.dry_run:
            subprocess.run([str(arg) for arg in args], cwd=cwd, check=True)

    @staticmethod
    def output(*args: object, cwd: Path | None = None) -> str:
        return subprocess.check_output(
            [str(arg) for arg in args], cwd=cwd, text=True
        ).strip()


def _is_git_checkout(path: Path) -> bool:
    if not path.is_dir():
        return False
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    try:
        return Path(result.stdout.strip()).resolve() == path.resolve()
    except OSError:
        return False


def _git_output(installer: Installer, path: Path, *args: str) -> str:
    return installer.output("git", "-C", path, *args)


def _source_url(
    *,
    explicit: str | None,
    env_name: str,
    default: str,
) -> str:
    return explicit or os.environ.get(env_name) or default


def _policy_space_source_url(
    installer: Installer,
    *,
    explicit: str | None,
    path: Path,
) -> str:
    configured = explicit or os.environ.get("MANAENV_POLICY_SPACE_URL")
    if configured:
        return configured
    if _is_git_checkout(path):
        return _git_output(installer, path, "remote", "get-url", "origin")
    return DEFAULT_POLICY_SPACE_URL


def _require_clean_checkout(
    installer: Installer,
    path: Path,
    label: str,
    *,
    ignored_prefixes: tuple[str, ...] = (),
) -> None:
    dirty = _git_output(
        installer, path, "status", "--porcelain", "--untracked-files=all"
    )
    if ignored_prefixes:
        dirty = "\n".join(
            line
            for line in dirty.splitlines()
            if not any(line[3:].startswith(prefix) for prefix in ignored_prefixes)
        )
    if dirty:
        raise RuntimeError(
            f"{label} checkout is dirty; preserve or clean it before installation: {path}"
        )


def _ensure_checkout(
    installer: Installer,
    *,
    label: str,
    url: str,
    path: Path,
    branch: str,
    update: bool,
) -> None:
    if path.is_dir() and not any(path.iterdir()):
        if not installer.dry_run:
            path.rmdir()
    if not path.exists():
        if not installer.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
        installer.run(
            "git",
            "clone",
            "--branch",
            branch,
            "--single-branch",
            url,
            path,
        )
        return
    if not _is_git_checkout(path):
        raise RuntimeError(f"{label} target exists but is not a Git checkout: {path}")

    actual_url = _git_output(installer, path, "remote", "get-url", "origin")
    if actual_url != url:
        raise RuntimeError(f"{label} origin mismatch: expected {url}, got {actual_url}")
    try:
        actual_branch = _git_output(
            installer, path, "symbolic-ref", "--quiet", "--short", "HEAD"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{label} must be checked out on branch {branch}: {path}"
        ) from exc
    if actual_branch != branch:
        raise RuntimeError(
            f"{label} must be on {branch}, got {actual_branch}: {path}"
        )
    _require_clean_checkout(installer, path, label)

    if update:
        installer.run("git", "fetch", "origin", branch, cwd=path)
        installer.run(
            "git", "merge", "--ff-only", f"origin/{branch}", cwd=path
        )
        if not installer.dry_run:
            head = _git_output(installer, path, "rev-parse", "HEAD")
            remote = _git_output(
                installer, path, "rev-parse", f"origin/{branch}"
            )
            if head != remote:
                raise RuntimeError(
                    f"{label} did not reach origin/{branch}: {path}"
                )


def _ensure_isaaclab_submodule(
    installer: Installer,
    *,
    arena_dir: Path,
    isaaclab_dir: Path,
    isaaclab_url: str,
    update: bool,
) -> None:
    if not _is_git_checkout(arena_dir):
        if installer.dry_run:
            installer.run(
                "git",
                "-C",
                arena_dir,
                "-c",
                f"submodule.submodules/IsaacLab.url={isaaclab_url}",
                "submodule",
                "update",
                "--init",
                "--checkout",
                "--depth",
                "1",
                "--",
                "submodules/IsaacLab",
            )
            return
        raise RuntimeError(f"Arena checkout is unavailable: {arena_dir}")

    configured_branch = _git_output(
        installer,
        arena_dir,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.submodules/IsaacLab.branch",
    )
    if configured_branch != X2REAL_BRANCH:
        raise RuntimeError(
            "Arena's IsaacLab submodule must track x2real; "
            f".gitmodules contains {configured_branch!r}."
        )
    if _is_git_checkout(isaaclab_dir):
        _require_clean_checkout(
            installer,
            isaaclab_dir,
            "IsaacLab",
            ignored_prefixes=("newton/",),
        )

    installer.run(
        "git", "submodule", "sync", "--", "submodules/IsaacLab", cwd=arena_dir
    )
    installer.run(
        "git",
        "config",
        "submodule.submodules/IsaacLab.url",
        isaaclab_url,
        cwd=arena_dir,
    )
    installer.run(
        "git",
        "-c",
        f"submodule.submodules/IsaacLab.url={isaaclab_url}",
        "submodule",
        "update",
        "--init",
        "--checkout",
        "--depth",
        "1",
        "--",
        "submodules/IsaacLab",
        cwd=arena_dir,
    )
    if installer.dry_run:
        return

    installer.run("git", "remote", "set-url", "origin", isaaclab_url, cwd=isaaclab_dir)
    gitlink = _git_output(installer, arena_dir, "rev-parse", "HEAD:submodules/IsaacLab")
    head = _git_output(installer, isaaclab_dir, "rev-parse", "HEAD")
    if head != gitlink:
        raise RuntimeError(
            f"IsaacLab checkout {head} does not match Arena gitlink {gitlink}"
        )
    if update:
        installer.run("git", "fetch", "origin", X2REAL_BRANCH, cwd=isaaclab_dir)
        contained = subprocess.run(
            [
                "git",
                "-C",
                str(isaaclab_dir),
                "merge-base",
                "--is-ancestor",
                head,
                "FETCH_HEAD",
            ],
            check=False,
        )
        if contained.returncode != 0:
            raise RuntimeError(
                f"Arena's IsaacLab gitlink is not from origin/{X2REAL_BRANCH}: {head}"
            )


def _validate_isaacsim_gitlink(
    installer: Installer,
    *,
    isaaclab_dir: Path,
    isaacsim_url: str,
) -> None:
    if installer.dry_run:
        return
    if not _is_git_checkout(isaaclab_dir):
        raise RuntimeError(f"IsaacLab checkout is unavailable: {isaaclab_dir}")

    configured_url = _git_output(
        installer,
        isaaclab_dir,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.IsaacSim.url",
    )
    if configured_url != isaacsim_url:
        raise RuntimeError(
            "IsaacLab's IsaacSim gitlink URL mismatch: "
            f"expected {isaacsim_url}, got {configured_url}"
        )

    configured_branch = _git_output(
        installer,
        isaaclab_dir,
        "config",
        "-f",
        ".gitmodules",
        "--get",
        "submodule.IsaacSim.branch",
    )
    if configured_branch != X2REAL_BRANCH:
        raise RuntimeError(
            "IsaacLab's IsaacSim gitlink must track x2real; "
            f".gitmodules contains {configured_branch!r}."
        )


def _verify_python(python: Path) -> None:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"Python interpreter is not executable: {python}")
    version = subprocess.check_output(
        [
            str(python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        text=True,
    ).strip()
    if version != "3.12":
        raise RuntimeError(f"ManaEnv requires Python 3.12; {python} reports {version}")


def _verify_prerequisites(python: Path, *, install_sources: bool) -> None:
    modules = (
        ("isaacsim",)
        if install_sources
        else (
            "isaacsim",
            "newton",
            "newton_usd_schemas",
            "isaaclab",
            "isaaclab_arena",
            "isaaclab_visualizers.kit",
            "policy_space_protocol",
        )
    )
    code = """
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, version

expected_isaacsim = sys.argv[1]
try:
    installed_isaacsim = version("isaacsim")
except PackageNotFoundError:
    raise SystemExit("[ERROR] Missing simulation prerequisite: isaacsim")
if installed_isaacsim != expected_isaacsim:
    raise SystemExit(
        f"[ERROR] Isaac Sim {expected_isaacsim} is required; found {installed_isaacsim}"
    )

missing = [name for name in sys.argv[2:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("[ERROR] Missing simulation prerequisites: " + ", ".join(missing))
"""
    subprocess.run([str(python), "-c", code, ISAACSIM_VERSION, *modules], check=True)


def _isaaclab_targets(isaaclab_dir: Path) -> list[Path]:
    source_root = isaaclab_dir / "source"
    if not source_root.is_dir():
        raise RuntimeError(f"IsaacLab source directory is missing: {source_root}")
    targets = sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir()
        and ((path / "pyproject.toml").is_file() or (path / "setup.py").is_file())
    )
    if not targets:
        raise RuntimeError(
            f"No installable IsaacLab packages found under {source_root}"
        )
    return targets


def _install_editables(
    installer: Installer,
    *,
    python: Path,
    arena_dir: Path,
    isaaclab_dir: Path,
    policy_space_dir: Path,
    install_sources: bool,
) -> None:
    if install_sources:
        installer.run(
            python,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            NEWTON_PIP_SPEC,
            NEWTON_SCHEMAS_PIP_SPEC,
        )
        if installer.dry_run and not isaaclab_dir.is_dir():
            print(
                "+ "
                f"{shlex.quote(str(python))} -m pip install --no-deps --editable "
                "<IsaacLab source packages> --editable <Arena> "
                "--editable <Policy Space protocol>"
            )
        else:
            targets = _isaaclab_targets(isaaclab_dir)
            targets.append(arena_dir)
            targets.append(policy_space_dir / "packages" / "protocol")
            command: list[object] = [python, "-m", "pip", "install", "--no-deps"]
            for target in targets:
                command.extend(("--editable", target))
            installer.run(*command)
    installer.run(python, "-m", "pip", "install", "--editable", ROOT)
    installer.run(python, "-P", ROOT / "manaenv/scripts/x2real_runtime_wheels.py", "--install")


def _verify_install(
    python: Path,
    *,
    arena_dir: Path,
    isaaclab_dir: Path,
    policy_space_dir: Path,
    install_sources: bool,
) -> None:
    code = """
import importlib
import sys
from importlib.metadata import version
from pathlib import Path

installed_isaacsim = version("isaacsim")
if installed_isaacsim != sys.argv[1]:
    raise SystemExit(
        f"[ERROR] Isaac Sim {sys.argv[1]} is required; found {installed_isaacsim}"
    )
modules = (
    "isaacsim",
    "newton",
    "newton_usd_schemas",
    "isaaclab",
    "isaaclab_arena",
    "isaaclab_visualizers.kit",
    "policy_space_protocol",
    "manaenv",
)
loaded = {name: importlib.import_module(name) for name in modules}
if sys.argv[2] == "true":
    expected = {
        "isaaclab": Path(sys.argv[3]).resolve(),
        "isaaclab_arena": Path(sys.argv[4]).resolve(),
        "policy_space_protocol": Path(sys.argv[5]).resolve(),
        "manaenv": Path(sys.argv[6]).resolve(),
    }
    for name, root in expected.items():
        location = Path(loaded[name].__file__).resolve()
        if not location.is_relative_to(root):
            raise SystemExit(f"[ERROR] {name} resolved outside {root}: {location}")
print("ManaEnv publication source install: OK")
"""
    subprocess.run(
        [
            str(python),
            "-c",
            code,
            ISAACSIM_VERSION,
            str(install_sources).lower(),
            str(isaaclab_dir),
            str(arena_dir),
            str(policy_space_dir),
            str(ROOT),
        ],
        check=True,
    )
    subprocess.run([str(python), "-m", "pip", "check"], check=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="target Python 3.12 interpreter",
    )
    parser.add_argument(
        "--arena-url", help="override the public IsaacLab Arena x2real URL"
    )
    parser.add_argument(
        "--isaaclab-url", help="override the public IsaacLab x2real URL"
    )
    parser.add_argument(
        "--isaacsim-url", help="override the public Isaac Sim x2real URL"
    )
    parser.add_argument(
        "--policy-space-url",
        help="override the GitHub Policy Space main-branch URL",
    )
    parser.add_argument(
        "--skip-source-update",
        action="store_true",
        help="do not fetch existing source checkouts",
    )
    parser.add_argument(
        "--skip-sources",
        action="store_true",
        help="use already-installed simulation packages",
    )
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="prepare dependency source checkouts without pip installs",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip prerequisite and final import checks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print mutating commands without running them",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.skip_sources and args.sources_only:
        raise RuntimeError("--skip-sources and --sources-only cannot be used together")

    python = args.python.expanduser().resolve()
    if not args.sources_only:
        _verify_python(python)
    installer = Installer(dry_run=args.dry_run)
    deps_root = (
        Path(os.environ.get("MANAENV_DEPS_ROOT", ROOT / "3rd")).expanduser().resolve()
    )
    arena_dir = deps_root / "isaaclabarena"
    isaaclab_dir = arena_dir / "submodules" / "IsaacLab"
    policy_space_dir = deps_root / "policy_space"

    if not args.skip_sources:
        isaacsim_url = _source_url(
            explicit=args.isaacsim_url,
            env_name="MANAENV_ISAACSIM_URL",
            default=DEFAULT_ISAACSIM_URL,
        )
        arena_url = _source_url(
            explicit=args.arena_url,
            env_name="MANAENV_ARENA_URL",
            default=DEFAULT_ARENA_URL,
        )
        isaaclab_url = _source_url(
            explicit=args.isaaclab_url,
            env_name="MANAENV_ISAACLAB_URL",
            default=DEFAULT_ISAACLAB_URL,
        )
        policy_space_url = _policy_space_source_url(
            installer,
            explicit=args.policy_space_url,
            path=policy_space_dir,
        )
        update = not args.skip_source_update
        _ensure_checkout(
            installer,
            label="Arena",
            url=arena_url,
            path=arena_dir,
            branch=X2REAL_BRANCH,
            update=update,
        )
        _ensure_checkout(
            installer,
            label="Policy Space",
            url=policy_space_url,
            path=policy_space_dir,
            branch=POLICY_SPACE_BRANCH,
            update=update,
        )
        _ensure_isaaclab_submodule(
            installer,
            arena_dir=arena_dir,
            isaaclab_dir=isaaclab_dir,
            isaaclab_url=isaaclab_url,
            update=update,
        )
        _validate_isaacsim_gitlink(
            installer,
            isaaclab_dir=isaaclab_dir,
            isaacsim_url=isaacsim_url,
        )

    if args.sources_only:
        print("Dependency source preparation complete.")
        return 0
    if not args.skip_verify and not args.dry_run:
        _verify_prerequisites(python, install_sources=not args.skip_sources)
    _install_editables(
        installer,
        python=python,
        arena_dir=arena_dir,
        isaaclab_dir=isaaclab_dir,
        policy_space_dir=policy_space_dir,
        install_sources=not args.skip_sources,
    )
    if not args.skip_verify and not args.dry_run:
        _verify_install(
            python,
            arena_dir=arena_dir,
            isaaclab_dir=isaaclab_dir,
            policy_space_dir=policy_space_dir,
            install_sources=not args.skip_sources,
        )
    print("ManaEnv publication install complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
