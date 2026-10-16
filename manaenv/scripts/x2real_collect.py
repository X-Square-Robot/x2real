#!/usr/bin/env python3
"""Collect manual demonstrations on public X2Real ID scenes.

Keyboard, Vuer/WebXR controllers, and WebSocket master arms share the same
HDF5 recorder. Selection, validation, and dry-runs do not start Isaac Sim.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from manaenv.pipeline.registry.bootstrap import bootstrap_discovery
from manaenv.pipeline.translator.reflective_translator import reflect_section
from manaenv.scripts._x2real_suite import check_asset_references
from manaenv.scripts.x2real_eval import (
    EvalBoard,
    EvalCase,
    build_run_suite_document,
    check_eval_catalog,
    filter_eval_cases,
    get_board,
    load_all_eval_cases,
    load_boards,
    resolve_seed,
)

_METHODS = {
    "keyboard": ("keyboard_teleop.yaml", "KeyboardTeleopPlanner"),
    "vuer": ("vuer_controller_teleop.yaml", "VRTeleopPlanner"),
    "masterslave": ("masterslave_teleop.yaml", "MasterSlaveTeleopPlanner"),
}
_ROBOTS = {"artixon_arm6a", "quanta_x1"}


def build_collection_suite(
    case: EvalCase,
    *,
    board: EvalBoard,
    teleop: str,
    save_path: str,
    num_demos: int = 10,
    planner_overrides: dict[str, Any] | None = None,
    seed: int | None = None,
    episode_length_s: int | None = None,
) -> dict[str, Any]:
    """Derive one manual session from an existing public ID assembly."""
    if case.robot not in _ROBOTS:
        raise ValueError(f"{case.robot} has no public bimanual teleop device contract")
    if isinstance(num_demos, bool) or not isinstance(num_demos, int) or num_demos <= 0:
        raise ValueError("num_demos must be a positive integer")
    if teleop == "vr":
        teleop = "vuer"
    if teleop not in _METHODS:
        raise ValueError(f"Unknown manual collection method: {teleop}")
    if not str(save_path).strip():
        raise ValueError("save_path must be a local directory")
    if "://" in str(save_path):
        raise ValueError("Manual collection requires a local output directory")
    document = build_run_suite_document([case], board=board, num_parallel_envs=1)
    filename, symbol = _METHODS[teleop]
    planner = yaml.safe_load((_ROOT / "manaenv/defination/datacollection" / filename).read_text())
    planner["planner"][symbol].update(planner_overrides or {})
    order = document["rollout"][0]
    order.pop("num_episodes", None)
    order.pop("resume", None)
    order.update(planner)
    order.update(num_demos=num_demos, num_parallel_envs=1, enable_hotkeys=False)
    if episode_length_s is not None:
        if episode_length_s <= 0:
            raise ValueError("episode_length_s must be positive")
        order["episode_length_s"] = int(episode_length_s)
    if seed is not None:
        order.setdefault("metadata", {})["collection_seed"] = int(seed)
    destination = Path(os.path.expandvars(save_path)).expanduser().resolve()
    document["output"] = [{"type": "datagen", "enabled": True, "format": "hdf5", "save_path": str(destination)}]
    bootstrap_discovery()
    reflect_section("rollout", document["rollout"], base_dir=board.suite.parent)
    reflect_section("output", document["output"], base_dir=board.suite.parent)
    return document


def check_collection() -> list[str]:
    errors = check_eval_catalog()
    boards = load_boards()
    cases = [case for case in load_all_eval_cases(boards) if case.robot in _ROBOTS]
    for teleop in _METHODS:
        for robot in sorted(_ROBOTS):
            case = next(case for case in cases if case.robot == robot)
            try:
                build_collection_suite(
                    case,
                    board=boards[case.board],
                    teleop=teleop,
                    save_path="./outputs/x2real_collect/check",
                    num_demos=1,
                )
            except (OSError, TypeError, ValueError, KeyError, ImportError) as exc:
                errors.append(f"{teleop}/{robot}: {exc}")
    return errors


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("gendata",), default="gendata", help="Manual demonstration collection (default)"
    )
    parser.add_argument("--board", help="Public ID board; inferred when --env selects one case")
    parser.add_argument("--env", action="append", help="Exact public environment name")
    parser.add_argument("--task", action="append", help="Filter by public task name")
    parser.add_argument("--gripper", action="append", help="Filter by public gripper")
    parser.add_argument("--teleop", choices=(*_METHODS, "vr"), default="keyboard", help="vr is an alias for Vuer/WebXR")
    parser.add_argument("--num-demos", type=_positive, default=10, help="Successful demonstrations to save")
    parser.add_argument(
        "--episode-length-s", type=_positive, default=None, help="Override the ID case timeout for human operation"
    )
    parser.add_argument("--save-path", default="./outputs/x2real_collect", help="Local HDF5 and report directory")
    parser.add_argument("--seed", default=None, help="Integer or random")
    parser.add_argument("--master-address", default="127.0.0.1", help="WebSocket master-arm server host")
    parser.add_argument("--master-port", type=_positive, default=5555)
    parser.add_argument("--vuer-host", default="127.0.0.1", help="Bind address; set explicitly for a remote headset")
    parser.add_argument("--vuer-port", type=_positive, default=8012)
    parser.add_argument("--viz", choices=("kit", "none"), default="kit")
    parser.add_argument("--list", action="store_true", help="List supported manual-collection ID cases")
    parser.add_argument("--check", action="store_true", help="Validate public catalog and manual configurations")
    parser.add_argument("--asset-check", action="store_true", help="Validate the local public asset bundle")
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve a session without starting Isaac Sim or devices"
    )
    parser.add_argument("--inspect", type=Path, help="Inspect a saved HDF5 dataset without simulation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.inspect:
        import json

        from manaenv.pipeline.datacollection.dataset import inspect_dataset

        print(json.dumps(inspect_dataset(args.inspect), ensure_ascii=False, indent=2))
        return 0
    try:
        if args.check or args.asset_check:
            errors = check_collection() if args.check else check_asset_references()
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            if not errors:
                print("Manual collection configuration check passed" if args.check else "Asset check passed")
            return 1 if errors else 0
        boards = load_boards()
        board = get_board(args.board, boards) if args.board else None
        cases = [case for case in load_all_eval_cases(boards) if case.robot in _ROBOTS]
        if board:
            cases = [case for case in cases if case.board == board.board_id]
        cases = filter_eval_cases(cases, envs=args.env, tasks=args.task, grippers=args.gripper)
        if args.list:
            for case in cases:
                print(f"{case.board}\t{case.env}")
            print(f"{len(cases)} manual-collection ID cases")
            return 0
        if len(cases) != 1:
            parser.error(
                f"select exactly one supported ID case with --env or --board/--task/--gripper; matched {len(cases)}"
            )
        case = cases[0]
        board = boards[case.board]
        method = "vuer" if args.teleop == "vr" else args.teleop
        if method == "keyboard" and args.viz != "kit":
            parser.error("keyboard teleoperation requires --viz kit")
        overrides: dict[str, Any] = {}
        if method == "masterslave":
            overrides = {"remote_ip": args.master_address, "remote_port": args.master_port}
        elif method == "vuer":
            overrides = {"vuer_host": args.vuer_host, "vuer_port": args.vuer_port}
        seed = resolve_seed(args.seed)
        document = build_collection_suite(
            case,
            board=board,
            teleop=method,
            save_path=args.save_path,
            num_demos=args.num_demos,
            planner_overrides=overrides,
            seed=seed,
            episode_length_s=args.episode_length_s,
        )
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    with tempfile.NamedTemporaryFile(
        prefix="_x2real_collect_", suffix=".yaml", dir=board.suite.parent, delete=False
    ) as handle:
        config = Path(handle.name)
    try:
        config.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
        command = [
            sys.executable,
            "-P",
            str(_ROOT / "manaenv/scripts/main.py"),
            "--mode",
            args.mode,
            "--config",
            str(config),
            "--env-name",
            case.env,
            "--seed",
            str(seed),
            "--enable_cameras",
        ]
        command.extend(["--visualizer", args.viz])
        print(f"Manual collection: {case.env}, {method}, {args.num_demos} demonstrations, seed={seed}", flush=True)
        print(f"Output: {document['output'][0]['save_path']}", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            print("Dry-run only; no simulator or input device started")
            return 0
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        return subprocess.run(command, cwd=_ROOT, env=env, check=False).returncode
    finally:
        config.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
