"""Build traceable local Isaac Sim wheels for the tested X2Real runtime profile.

Only distribution metadata changes. Runtime files are copied and checked against
the original NVIDIA wheels before pip installs the explicitly versioned builds.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import json
import os
import shlex
import subprocess
import sys
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path
from zipfile import ZipFile

BASE_VERSION = "6.0.1.0"
LOCAL_VERSION = BASE_VERSION + "+x2real.1"
PROVENANCE = "MANAENV_COMPAT.json"
# NVIDIA's CPython 3.12 / manylinux_2_35_x86_64 originals used for this profile.
SOURCE_SHA256 = {
    "isaacsim-core": "aadb3c5dffd433d5caec5cf0c4c7a314d30b222997bb8496385baae67dd3367c",
    "isaacsim-kernel": "618fd6fb09d7d67ca5102251b999bf1a851a8918dadbeea0789e7d4f2a24bbb1",
}
# Exact versions exercised by this profile. A new profile needs a local version
# bump and runtime validation; the upstream NVIDIA release remains identifiable.
OVERRIDES = {
    "isaacsim-core": {
        "mujoco==3.8.0": "mujoco==3.11.0",
        "mujoco-warp==3.8.0.3": "mujoco-warp==3.11.0",
        "newton[sim]==1.2.1": "newton[sim]==1.6.0.dev0",
        "newton-usd-schemas==0.2.0": "newton-usd-schemas==0.5.0",
    },
    "isaacsim-kernel": {
        "coverage==7.4.4": "coverage==7.6.1",
        "typing_extensions==4.12.2": "typing_extensions==4.16.0",
        "websockets==12.0": "websockets==16.1.1",
    },
}


def _digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _file_hash(path: Path) -> bytes:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").digest()


def _renamed(path: str) -> str:
    head, separator, tail = path.partition("/")
    if head.endswith((".dist-info", ".data")):
        head = head.replace("-" + BASE_VERSION + ".", "-" + LOCAL_VERSION + ".", 1)
    return head + separator + tail


def _patch_metadata(data: bytes, distribution: str) -> bytes:
    message = BytesParser().parsebytes(data)
    if message["Version"] != BASE_VERSION or message["Name"].replace("_", "-") != distribution:
        raise ValueError("Only the unmodified Isaac Sim 6.0.1.0 metadata is supported")
    requirements = message.get_all("Requires-Dist", [])
    replacements = OVERRIDES[distribution]
    if any(requirements.count(requirement) != 1 for requirement in replacements):
        raise ValueError(f"{distribution}: upstream requirements changed; review the compatibility profile")
    message.replace_header("Version", LOCAL_VERSION)
    del message["Requires-Dist"]
    for requirement in requirements:
        message["Requires-Dist"] = replacements.get(requirement, requirement)
    return message.as_bytes()


def build_wheel(source: Path, destination: Path, distribution: str) -> Path:
    """Create a local-version wheel, preserving every non-METADATA source file."""
    normalized = distribution.replace("-", "_")
    prefix = f"{normalized}-{BASE_VERSION}-"
    if distribution not in OVERRIDES or not source.name.startswith(prefix):
        raise ValueError(f"Unexpected source wheel: {source.name}")
    filename = source.name.replace("-" + BASE_VERSION + "-", "-" + LOCAL_VERSION + "-", 1)
    output = destination / filename
    dist_info = f"{normalized}-{LOCAL_VERSION}.dist-info"
    source_hash = _file_hash(source).hex()
    provenance = json.dumps({
        "profile": LOCAL_VERSION,
        "upstream_wheel": source.name,
        "upstream_sha256": source_hash,
        "requires_dist_changes": OVERRIDES[distribution],
        "runtime_payload": "byte-identical to upstream wheel",
    }, sort_keys=True, indent=2).encode()
    destination.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    try:
        with ZipFile(source) as original, ZipFile(temporary, "w") as patched:
            members = original.namelist()
            if len(set(members)) != len(members):
                raise ValueError("Duplicate source wheel members")
            # Detached signatures cannot apply to a downstream build.
            if any(name.endswith(("/RECORD.jws", "/RECORD.p7s")) for name in members):
                raise ValueError("Signed source wheel needs a reviewed downstream signing process")
            metadata_path = f"{normalized}-{BASE_VERSION}.dist-info/METADATA"
            updated_metadata = _patch_metadata(original.read(metadata_path), distribution)
            records = []
            for info in original.infolist():
                if info.is_dir() or info.filename.endswith(".dist-info/RECORD"):
                    continue
                new_info = copy.copy(info)
                new_info.filename = _renamed(info.filename)
                data = updated_metadata if info.filename == metadata_path else original.read(info.filename)
                patched.writestr(new_info, data)
                records.append((new_info.filename, "sha256=" + _digest(data), len(data)))
            provenance_path = dist_info + "/" + PROVENANCE
            patched.writestr(provenance_path, provenance)
            records.append((provenance_path, "sha256=" + _digest(provenance), len(provenance)))
            record_path = dist_info + "/RECORD"
            records.append((record_path, "", ""))
            record = io.StringIO(newline="")
            csv.writer(record).writerows(records)
            patched.writestr(record_path, record.getvalue())
        verify_payload(source, temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def verify_payload(source: Path, patched: Path) -> int:
    """Independently re-read both archives and compare preserved file contents."""
    with ZipFile(source) as original, ZipFile(patched) as rebuilt:
        count = 0
        for info in original.infolist():
            if info.is_dir() or info.filename.endswith((".dist-info/METADATA", ".dist-info/RECORD")):
                continue
            if original.read(info.filename) != rebuilt.read(_renamed(info.filename)):
                raise ValueError(f"Wheel payload changed: {info.filename}")
            count += 1
        return count


def verify_installed_payload(source: Path, distribution: str) -> None:
    """Preserve local runtime edits rather than overwriting them during repair."""
    installed = metadata.distribution(distribution)
    with ZipFile(source) as original:
        for info in original.infolist():
            if info.is_dir() or ".dist-info/" in info.filename or ".data/" in info.filename:
                continue
            path = Path(installed.locate_file(info.filename))
            if not path.is_file() or _file_hash(path) != hashlib.sha256(original.read(info.filename)).digest():
                raise ValueError(f"Installed runtime differs from NVIDIA wheel; preserve and review: {path}")


def _run(command: list[str]) -> None:
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Build, verify, and install into this interpreter")
    parser.add_argument("--dry-run", action="store_true", help="Show the profile without downloads or installation")
    cache = os.environ.get("MANAENV_RUNTIME_WHEEL_CACHE")
    default_cache = Path(cache) if cache else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "manaenv/isaacsim-x2real"
    parser.add_argument("--wheel-dir", type=Path, default=default_cache)
    args = parser.parse_args(argv)
    print(json.dumps({"interpreter": sys.executable, "local_version": LOCAL_VERSION, "overrides": OVERRIDES}, indent=2), flush=True)
    if args.dry_run:
        print("Dry-run: download NVIDIA originals, verify runtime payloads, build local wheels, install and run pip check")
        return 0
    for name, expected in (("newton", "1.6.0.dev0"), ("newton-usd-schemas", "0.5.0")):
        if metadata.version(name) != expected:
            raise ValueError(f"{name}: this profile requires the repository's {expected} X2Real build")
    pending = []
    for name in OVERRIDES:
        installed = metadata.distribution(name)
        if installed.version == LOCAL_VERSION:
            provenance = json.loads(installed.read_text(PROVENANCE) or "{}")
            if (
                provenance.get("requires_dist_changes") != OVERRIDES[name]
                or provenance.get("upstream_sha256") != SOURCE_SHA256[name]
            ):
                raise ValueError(f"{name}: installed compatibility provenance does not match the profile")
        elif installed.version == BASE_VERSION:
            pending.append(name)
        else:
            raise ValueError(f"{name}: unsupported installed version {installed.version}")
    wheels = []
    originals = args.wheel_dir / "original"
    for name in pending:
        originals.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
              "--index-url", "https://pypi.nvidia.com", "--dest", str(originals), name + "==" + BASE_VERSION])
        sources = list(originals.glob(name.replace("-", "_") + "-" + BASE_VERSION + "-*.whl"))
        if len(sources) != 1:
            raise ValueError(f"{name}: expected exactly one source wheel for this interpreter/platform")
        if _file_hash(sources[0]).hex() != SOURCE_SHA256[name]:
            raise ValueError(f"{name}: NVIDIA source wheel hash changed; review and validate a new profile")
        verify_installed_payload(sources[0], name)
        wheel = build_wheel(sources[0], args.wheel_dir / "patched", name)
        print(f"Verified unchanged payload: {wheel}", flush=True)
        wheels.append(str(wheel))
    if args.install:
        # These auxiliary versions are also the dependencies encoded in the
        # local kernel wheel; coverage restores Numba's actual import contract.
        _run([sys.executable, "-m", "pip", "install", "--no-deps",
              "coverage==7.6.1", "typing_extensions==4.16.0", "websockets==16.1.1", *wheels])
        _run([sys.executable, "-m", "pip", "check"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, metadata.PackageNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
