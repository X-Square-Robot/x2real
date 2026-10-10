#!/usr/bin/env python3
"""Run the five formal public X2Real ID evaluation boards.

The board registry and its ID-only leaderboard CSVs define the public cases.
The checked-in ``x2real.yaml`` catalog supplies their scene, task, robot, and
policy wiring. Each run materializes a temporary suite containing only the
selected board cases and applies the board's standardized rollout budgets.

Filters accept comma-separated values or repeated options::

    --board artixon_arm6a_tabletop|artixon_arm6a_challenge|quanta_x1_mobile|arx_r5_tabletop|franka_panda_tabletop
    --subcategory <leaderboard category>
    --robot artixon_arm6a|quanta_x1|arx_r5|franka_panda
    --gripper H|G|default|panda
    --task <task slug>
    --env <exact environment name>

Examples::

    python manaenv/scripts/x2real_eval.py --list --board artixon_arm6a_tabletop
    python manaenv/scripts/x2real_eval.py --board arx_r5_tabletop --dry-run --task usb_plug \\
      --num-episodes 2 --timeout-s 30 --num-parallel-envs 1
    python manaenv/scripts/x2real_eval.py --board artixon_arm6a_tabletop \\
      --env artixon_arm6a_H_plug_in_charger_id_env \\
      --seed 42 --server-address 127.0.0.1 --server-port 8001 --viz kit

ID benchmark is the only public mode. Held-out-axis composition,
teleoperation, preview, and data-generation modes are not part of this entry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Keep ``python manaenv/scripts/x2real_eval.py`` bound to this checkout even
# when the active environment contains an editable private ManaEnv install.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from manaenv.pipeline.evaluation.report import load_evaluation_episode_jsonl
from manaenv.scripts._x2real_suite import (
    canonical_env_name,
    canonical_robot_name,
    check_asset_references,
    check_suite,
    load_cases,
    load_x2real_document,
    validate_runtime_passthrough,
)

_X2REAL_CONFIG = _REPO_ROOT / "manaenv" / "defination" / "x2real.yaml"
_DEFINITION_ROOT = _X2REAL_CONFIG.parent
_BOARD_REGISTRY = _DEFINITION_ROOT / "x2real" / "task" / "boards.yaml"
_DEFAULT_BOARD = "artixon_arm6a_tabletop"
_DEFAULT_EVAL_SEED = 42
_STEPS_PER_SECOND = 30
_SUITE_SWEEP_MAX_AGE_S = 24 * 60 * 60
_LEADERBOARD_FIELDS = (
    "subcategory",
    "task",
    "x2real_package",
    "n_id",
    "id_envs",
    "id_timeout_s",
)
_EXPECTED_BOARD_COUNTS = {
    "artixon_arm6a_tabletop": 38,
    "artixon_arm6a_challenge": 10,
    "quanta_x1_mobile": 3,
    "arx_r5_tabletop": 12,
    "franka_panda_tabletop": 10,
}
_BOARD_ALIASES = {
    "ex001_tabletop": "artixon_arm6a_tabletop",
    "ex001_challenge": "artixon_arm6a_challenge",
    "ex001_mobile": "quanta_x1_mobile",
    "arx": "arx_r5_tabletop",
    "franka_panda": "franka_panda_tabletop",
}
_BOARD_COLUMN_WIDTH = max(len("board"), *(len(name) for name in _EXPECTED_BOARD_COUNTS))


@dataclass(frozen=True)
class EvalBoard:
    """One independent formal ID evaluation board."""

    board_id: str
    robot: str
    grippers: tuple[str, ...]
    leaderboard_csv: Path
    suite: Path
    case_count: int
    num_episodes: int
    num_parallel_envs: int
    episode_length_s: int | None
    max_steps: int | None
    save_path: str


@dataclass(frozen=True)
class EvalCase:
    """One formal public ID case and its board rollout budget."""

    board: str
    subcategory: str
    leaderboard_task: str
    package: str
    robot: str
    gripper: str
    env_task: str
    split: str
    env: str
    timeout_s: int
    max_steps: int
    num_episodes: int
    num_parallel_envs: int


def _positive_int(value: Any, label: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a positive integer; got {value!r}") from None
    if resolved < 1:
        raise ValueError(f"{label} must be a positive integer; got {resolved}")
    return resolved


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _definition_path(value: Any, label: str) -> Path:
    raw = Path(str(value or "").strip())
    if not str(raw) or raw.is_absolute():
        raise ValueError(f"{label} must be relative to {_DEFINITION_ROOT}")
    resolved = (_DEFINITION_ROOT / raw).resolve()
    try:
        resolved.relative_to(_DEFINITION_ROOT.resolve())
    except ValueError:
        raise ValueError(f"{label} escapes {_DEFINITION_ROOT}: {raw}") from None
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a non-empty list")
    items = tuple(str(item).strip() for item in value if str(item).strip())
    if not items:
        raise ValueError(f"{label} must be a non-empty list")
    return items


def load_boards(path: Path | None = None) -> dict[str, EvalBoard]:
    """Load the sanitized public board registry."""

    registry = (path or _BOARD_REGISTRY).expanduser().resolve()
    try:
        document = yaml.safe_load(registry.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid board registry {registry}: {exc}") from exc
    raw_boards = document.get("boards") if isinstance(document, dict) else None
    if not isinstance(raw_boards, dict) or not raw_boards:
        raise ValueError(f"board registry has no boards mapping: {registry}")

    boards: dict[str, EvalBoard] = {}
    for raw_id, raw_body in raw_boards.items():
        board_id = str(raw_id).strip()
        if not board_id or not isinstance(raw_body, dict):
            raise ValueError(f"invalid board entry {raw_id!r}")
        if raw_body.get("scoring") != "id_only":
            raise ValueError(f"{board_id}: scoring must be 'id_only'")
        if "num_demos" in raw_body:
            raise ValueError(
                f"{board_id}: num_demos is a data-collection field and is not "
                "available on a public benchmark board"
            )
        if _string_tuple(raw_body.get("capabilities"), f"{board_id}.capabilities") != (
            "benchmark",
        ):
            raise ValueError(f"{board_id}: capabilities must be exactly ['benchmark']")
        episode_length_s = _optional_positive_int(
            raw_body.get("episode_length_s"), f"{board_id}.episode_length_s"
        )
        max_steps = _optional_positive_int(
            raw_body.get("max_steps"), f"{board_id}.max_steps"
        )
        if (episode_length_s is None) != (max_steps is None):
            raise ValueError(
                f"{board_id}: episode_length_s and max_steps must be set together"
            )
        boards[board_id] = EvalBoard(
            board_id=board_id,
            robot=str(raw_body.get("robot") or "").strip(),
            grippers=_string_tuple(
                raw_body.get("grippers"), f"{board_id}.grippers"
            ),
            leaderboard_csv=_definition_path(
                raw_body.get("leaderboard_csv"), f"{board_id}.leaderboard_csv"
            ),
            suite=_definition_path(raw_body.get("suite"), f"{board_id}.suite"),
            case_count=_positive_int(
                raw_body.get("case_count"), f"{board_id}.case_count"
            ),
            num_episodes=_positive_int(
                raw_body.get("num_episodes"), f"{board_id}.num_episodes"
            ),
            num_parallel_envs=_positive_int(
                raw_body.get("num_parallel_envs"),
                f"{board_id}.num_parallel_envs",
            ),
            episode_length_s=episode_length_s,
            max_steps=max_steps,
            save_path=str(raw_body.get("save_path") or "").strip(),
        )
        if not boards[board_id].robot or not boards[board_id].save_path:
            raise ValueError(f"{board_id}: robot and save_path must be non-empty")
    return boards


def get_board(board_id: str, boards: dict[str, EvalBoard] | None = None) -> EvalBoard:
    resolved = boards or load_boards()
    requested = str(board_id or "").strip().lower().replace("-", "_")
    canonical = _BOARD_ALIASES.get(requested, requested)
    try:
        return resolved[canonical]
    except KeyError:
        raise ValueError(
            f"unknown board {board_id!r}; choose from {', '.join(resolved)}"
        ) from None


def _split_column(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _rollouts_by_env(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rollouts: dict[str, dict[str, Any]] = {}
    for raw in document.get("rollout") or []:
        if not isinstance(raw, dict):
            continue
        env = str(raw.get("env_ref") or "").strip()
        if not env:
            continue
        if env in rollouts:
            raise ValueError(f"public suite contains duplicate rollout.env_ref {env!r}")
        rollouts[env] = raw
    return rollouts


def load_eval_cases(
    board_id: str = _DEFAULT_BOARD,
    boards: dict[str, EvalBoard] | None = None,
) -> list[EvalCase]:
    """Load one board's cases from its ID-only leaderboard CSV."""

    resolved_boards = boards or load_boards()
    board = get_board(board_id, resolved_boards)
    document = load_x2real_document(board.suite)
    rollouts = _rollouts_by_env(document)
    suite_cases = {case.env: case for case in load_cases(board.suite)}
    cases: list[EvalCase] = []
    seen: set[str] = set()
    with board.leaderboard_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _LEADERBOARD_FIELDS:
            raise ValueError(
                f"{board_id}: leaderboard columns must be "
                + ",".join(_LEADERBOARD_FIELDS)
            )
        for row_number, row in enumerate(reader, start=2):
            subcategory = str(row.get("subcategory") or "").strip()
            leaderboard_task = str(row.get("task") or "").strip()
            package = str(row.get("x2real_package") or "").strip()
            envs = _split_column(row.get("id_envs"))
            timeouts = [
                _positive_int(item, f"{board_id}:row {row_number}.id_timeout_s")
                for item in _split_column(row.get("id_timeout_s"))
            ]
            declared_count = _positive_int(
                row.get("n_id"), f"{board_id}:row {row_number}.n_id"
            )
            if not subcategory or not leaderboard_task or not package:
                raise ValueError(f"{board_id}: row {row_number} has an empty label")
            if len(envs) != declared_count or len(timeouts) != len(envs):
                raise ValueError(
                    f"{board_id}: row {row_number} count mismatch "
                    f"(n_id={declared_count}, envs={len(envs)}, timeouts={len(timeouts)})"
                )
            for env, csv_timeout in zip(envs, timeouts, strict=True):
                if env in seen:
                    raise ValueError(f"{board_id}: duplicate environment {env!r}")
                seen.add(env)
                suite_case = suite_cases.get(env)
                if suite_case is None:
                    raise ValueError(f"{board_id}: missing suite assembly {env!r}")
                if suite_case.split != "id":
                    raise ValueError(f"{board_id}: non-ID assembly {env!r}")
                if suite_case.robot != board.robot or suite_case.gripper not in board.grippers:
                    raise ValueError(
                        f"{board_id}: {env} has robot/gripper "
                        f"{suite_case.robot}/{suite_case.gripper} outside board contract"
                    )
                rollout = rollouts.get(env)
                if rollout is None:
                    raise ValueError(f"{board_id}: missing benchmark rollout for {env!r}")
                timeout = board.episode_length_s or csv_timeout
                max_steps = board.max_steps or timeout * _STEPS_PER_SECOND
                expected = {
                    "num_episodes": board.num_episodes,
                    "episode_length_s": timeout,
                    "max_steps": max_steps,
                    "num_parallel_envs": board.num_parallel_envs,
                }
                for key, expected_value in expected.items():
                    actual = _positive_int(rollout.get(key), f"{env}.{key}")
                    if actual != expected_value:
                        raise ValueError(
                            f"{board_id}: {env}.{key}={actual}, expected {expected_value}"
                        )
                cases.append(
                    EvalCase(
                        board=board_id,
                        subcategory=subcategory,
                        leaderboard_task=leaderboard_task,
                        package=package,
                        robot=suite_case.robot,
                        gripper=suite_case.gripper,
                        env_task=suite_case.task,
                        split=suite_case.split,
                        env=env,
                        timeout_s=timeout,
                        max_steps=max_steps,
                        num_episodes=board.num_episodes,
                        num_parallel_envs=board.num_parallel_envs,
                    )
                )

    if len(cases) != board.case_count:
        raise ValueError(
            f"{board_id}: expected {board.case_count} cases, got {len(cases)}"
        )
    return cases


def load_all_eval_cases(
    boards: dict[str, EvalBoard] | None = None,
) -> list[EvalCase]:
    resolved = boards or load_boards()
    cases = [case for board_id in resolved for case in load_eval_cases(board_id, resolved)]
    envs = [case.env for case in cases]
    if len(envs) != len(set(envs)):
        raise ValueError("formal boards contain duplicate environment names")
    return cases


def _name_set(values: Iterable[str] | None) -> set[str] | None:
    if not values:
        return None
    names = {name.strip() for value in values for name in str(value).split(",")}
    names.discard("")
    return names or None


def _task_matches(case: EvalCase, requested: set[str]) -> bool:
    candidates = {
        case.leaderboard_task.lower(),
        case.env_task.lower(),
        case.package.rsplit("/", 1)[-1].lower(),
    }
    for raw_name in requested:
        name = raw_name.lower()
        if name in candidates:
            return True
        prefix = f"{name}_"
        if any(
            candidate.startswith(prefix) and candidate[len(prefix) :].isdigit()
            for candidate in candidates
        ):
            return True
    return False


def filter_eval_cases(
    cases: Iterable[EvalCase],
    *,
    envs: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    subcategories: Iterable[str] | None = None,
    robots: Iterable[str] | None = None,
    grippers: Iterable[str] | None = None,
) -> list[EvalCase]:
    """Apply public ID filters while preserving suite order."""

    raw_env_names = _name_set(envs)
    env_names = (
        {canonical_env_name(name) for name in raw_env_names}
        if raw_env_names
        else None
    )
    task_names = _name_set(tasks)
    subcategory_names = _name_set(subcategories)
    raw_robot_names = _name_set(robots)
    robot_names = (
        {canonical_robot_name(name) for name in raw_robot_names}
        if raw_robot_names
        else None
    )
    gripper_names = _name_set(grippers)
    selected: list[EvalCase] = []
    for case in cases:
        if case.split != "id":
            continue
        if env_names and case.env not in env_names:
            continue
        if task_names and not _task_matches(case, task_names):
            continue
        if subcategory_names and case.subcategory not in subcategory_names:
            continue
        if robot_names and case.robot not in robot_names:
            continue
        if gripper_names and case.gripper not in gripper_names:
            continue
        selected.append(case)
    return selected


def catalog_env_names(board_id: str = _DEFAULT_BOARD) -> set[str]:
    return {case.env for case in load_eval_cases(board_id)}


def resolve_seed(raw: str | None) -> int:
    """Resolve the CLI seed to a concrete, replayable non-negative integer."""

    if raw is None:
        return _DEFAULT_EVAL_SEED
    token = raw.strip().lower()
    if token in {"random", "rand", "entropy"}:
        return secrets.randbelow(2**31)
    try:
        seed = int(token, 0)
    except ValueError:
        raise ValueError(
            f"--seed must be an integer or 'random'; got {raw!r}"
        ) from None
    if seed < 0:
        raise ValueError(f"--seed must be non-negative; got {seed}")
    return seed


def _raw_document(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid x2real YAML {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise TypeError(f"x2real config is not a mapping: {path}")
    return document


def _assembly_env(order: Any) -> str:
    if not isinstance(order, dict):
        return ""
    generator = order.get("assembly_generator") or order
    if not isinstance(generator, dict):
        return ""
    return str(generator.get("name") or "").strip()


def _rollout_env(order: Any) -> str:
    return str(order.get("env_ref") or "").strip() if isinstance(order, dict) else ""


def _load_resume_episode_counts(
    cases: list[EvalCase],
    *,
    save_path: str,
) -> dict[str, int]:
    root = Path(os.path.expandvars(save_path)).expanduser()
    if not root.is_absolute():
        root = _REPO_ROOT / root
    counts: dict[str, int] = {}
    for case in cases:
        report_path = root / case.env / "evaluation_report.jsonl"
        counts[case.env] = len(load_evaluation_episode_jsonl(report_path, env_name=case.env))
    return counts


def _derive_resume_attempt_seed(base_seed: int, counts: dict[str, int]) -> int:
    completed = tuple(sorted((env_name, count) for env_name, count in counts.items() if count > 0))
    if not completed:
        return int(base_seed)
    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "completed": completed,
            "version": "public_jsonl_resume.v1",
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    derived = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) & 0x7FFFFFFF
    if derived == int(base_seed):
        derived = (derived + 1) & 0x7FFFFFFF
    return derived


def build_run_suite_document(
    cases: list[EvalCase],
    *,
    board: EvalBoard,
    num_episodes: int | None = None,
    num_parallel_envs: int | None = None,
    timeout_s: int | None = None,
    save_path: str | None = None,
    resume: bool = False,
    resume_episode_counts: dict[str, int] | None = None,
    base_seed: int | None = None,
    attempt_seed: int | None = None,
) -> dict[str, Any]:
    """Build a temporary-suite document containing only *cases*."""

    if not cases:
        raise ValueError("no public ID cases selected")
    if num_episodes is not None:
        num_episodes = _positive_int(num_episodes, "--num-episodes")
    if num_parallel_envs is not None:
        num_parallel_envs = _positive_int(num_parallel_envs, "--num-parallel-envs")
    if timeout_s is not None:
        timeout_s = _positive_int(timeout_s, "--timeout-s")

    selected = [case.env for case in cases]
    selected_set = set(selected)
    if len(selected) != len(selected_set):
        raise ValueError(
            "selected evaluation cases contain duplicate environment names"
        )
    if any(case.split != "id" for case in cases):
        raise ValueError("temporary public evaluation suites may contain only ID cases")
    if {case.board for case in cases} != {board.board_id}:
        raise ValueError("a run suite may contain cases from exactly one board")

    source = board.suite
    document = _raw_document(source)
    if board.board_id == "arx_r5_tabletop":
        pipeline = document.get("pipeline")
        runtime = pipeline.get("pipeline_runtime") if isinstance(pipeline, dict) else None
        if not isinstance(runtime, dict):
            raise ValueError("arx_r5_tabletop requires pipeline.pipeline_runtime")
        runtime["include"] = "runtime/solver/physx_cpu.yaml"
        runtime["kit_render_mode"] = "sync"
    cases_by_env = {case.env: case for case in cases}
    assembly_orders = [
        order
        for order in document.get("assembly_orders") or []
        if _assembly_env(order) in selected_set
    ]
    rollout_orders = [
        order
        for order in document.get("rollout") or []
        if _rollout_env(order) in selected_set
    ]
    assembly_envs = {_assembly_env(order) for order in assembly_orders}
    rollout_envs = {_rollout_env(order) for order in rollout_orders}
    missing_assemblies = sorted(selected_set - assembly_envs)
    missing_rollouts = sorted(selected_set - rollout_envs)
    if missing_assemblies:
        raise ValueError(
            "selected assembly not found in public suite: "
            + ", ".join(missing_assemblies)
        )
    if missing_rollouts:
        raise ValueError(
            "selected rollout not found in public suite: " + ", ".join(missing_rollouts)
        )

    for rollout in rollout_orders:
        case = cases_by_env[_rollout_env(rollout)]
        episodes, timeout, max_steps, parallel = _effective_budget(
            case,
            num_episodes=num_episodes,
            num_parallel_envs=num_parallel_envs,
            timeout_s=timeout_s,
        )
        # The public runtime is benchmark-only.  Strip a legacy dev-suite
        # collection budget defensively instead of widening RolloutOrderCFG.
        rollout.pop("num_demos", None)
        rollout["num_episodes"] = episodes
        rollout["episode_length_s"] = timeout
        rollout["max_steps"] = max_steps
        rollout["num_parallel_envs"] = parallel
        if resume:
            existing_episodes = (resume_episode_counts or {}).get(case.env, 0)
            if isinstance(existing_episodes, bool) or not isinstance(existing_episodes, int):
                raise TypeError(f"{case.env}: resume episode count must be an integer")
            if existing_episodes < 0 or existing_episodes > episodes:
                raise ValueError(f"{case.env}: resume report has {existing_episodes} episodes; target is {episodes}")
            metadata = rollout.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise TypeError(f"{case.env}: rollout metadata must be a mapping")
            metadata["public_jsonl_resume"] = {
                "version": 1,
                "existing_episodes": existing_episodes,
                "target_episodes": episodes,
                "base_seed": base_seed,
                "attempt_seed": attempt_seed,
            }
            rollout["metadata"] = metadata
            rollout["resume"] = True

    document["assembly_orders"] = assembly_orders
    document["rollout"] = rollout_orders
    outputs = document.get("output") or []
    if len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise ValueError("public suite must declare exactly one output order")
    outputs[0]["save_path"] = save_path or board.save_path
    return document


def sweep_stale_suites(
    directory: Path | None = None,
    max_age_s: int = _SUITE_SWEEP_MAX_AGE_S,
) -> list[Path]:
    """Remove generated run suites older than the concurrency safety window."""

    root = directory or _X2REAL_CONFIG.parent
    cutoff = time.time() - max_age_s
    removed: list[Path] = []
    for path in root.glob("_x2real_eval_*.yaml"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed.append(path)
        except OSError:
            continue
    return removed


def materialize_run_suite(
    cases: list[EvalCase],
    *,
    board: EvalBoard,
    num_episodes: int | None = None,
    num_parallel_envs: int | None = None,
    timeout_s: int | None = None,
    save_path: str | None = None,
    copy_to_save_path: bool = False,
    resume: bool = False,
    resume_episode_counts: dict[str, int] | None = None,
    base_seed: int | None = None,
    attempt_seed: int | None = None,
) -> Path:
    """Write a selected ID suite next to its source so relative includes resolve."""

    source = board.suite
    swept = sweep_stale_suites(source.parent)
    if swept:
        print(
            f"INFO: removed {len(swept)} stale temporary suite(s) from {source.parent}"
        )
    document = build_run_suite_document(
        cases,
        board=board,
        num_episodes=num_episodes,
        num_parallel_envs=num_parallel_envs,
        timeout_s=timeout_s,
        save_path=save_path,
        resume=resume,
        resume_episode_counts=resume_episode_counts,
        base_seed=base_seed,
        attempt_seed=attempt_seed,
    )
    with tempfile.NamedTemporaryFile(
        prefix="_x2real_eval_",
        suffix=".yaml",
        dir=source.parent,
        delete=False,
    ) as handle:
        path = Path(handle.name)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    if copy_to_save_path:
        destination_root = Path(os.path.expandvars(save_path or board.save_path)).expanduser()
        if not destination_root.is_absolute():
            destination_root = _REPO_ROOT / destination_root
        destination = destination_root / "eval_suite.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return path


def _build_main_argv(
    config: Path,
    env_names: list[str],
    passthrough: list[str],
    *,
    seed: int,
) -> list[str]:
    main_py = _REPO_ROOT / "manaenv" / "scripts" / "main.py"
    try:
        config_arg = str(config.relative_to(_REPO_ROOT))
    except ValueError:
        config_arg = str(config)
    return [
        str(main_py),
        "--config",
        config_arg,
        # The public pipeline CLI is evaluation-only and intentionally has no
        # runtime ``--mode`` switch.  ID is enforced by this entry point and
        # benchmark output is declared by the generated suite itself.
        "--env-name",
        ",".join(env_names),
        "--seed",
        str(seed),
        *passthrough,
    ]


def _validate_passthrough(argv: list[str]) -> None:
    validate_runtime_passthrough(argv)
    disallowed = {
        "--config",
        "--gendata",
        "--ood",
        "--ood-axis",
        "--preview",
        "--teleop",
    }
    for token in argv:
        option = token.partition("=")[0]
        if option in disallowed:
            raise ValueError(
                f"{option} is not available in the public ID evaluation entry"
            )
        if option == "--seed":
            raise ValueError("pass --seed to x2real_eval.py itself")


def check_eval_catalog() -> list[str]:
    """Return formal-board/catalog consistency errors (empty means valid)."""

    errors: list[str] = []
    try:
        boards = load_boards()
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return [str(exc)]

    if set(boards) != set(_EXPECTED_BOARD_COUNTS):
        errors.append(
            "public boards must be exactly: "
            + ", ".join(_EXPECTED_BOARD_COUNTS)
        )
    all_cases: list[EvalCase] = []
    for board_id, expected_count in _EXPECTED_BOARD_COUNTS.items():
        board = boards.get(board_id)
        if board is None:
            continue
        if board.suite != _X2REAL_CONFIG.resolve():
            errors.append(f"{board_id}: suite must be {_X2REAL_CONFIG}")
        if board.case_count != expected_count:
            errors.append(
                f"{board_id}: case_count={board.case_count}, expected {expected_count}"
            )
        try:
            all_cases.extend(load_eval_cases(board_id, boards))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            errors.append(str(exc))

    envs = [case.env for case in all_cases]
    if len(envs) != 73 or len(set(envs)) != 73:
        errors.append(
            f"formal boards must contain 73 unique ID cases, got "
            f"{len(envs)} total / {len(set(envs))} unique"
        )
    try:
        suite_envs = {case.env for case in load_cases(_X2REAL_CONFIG)}
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(str(exc))
    else:
        catalog_envs = set(envs)
        missing = sorted(catalog_envs - suite_envs)
        extra = sorted(suite_envs - catalog_envs)
        if missing:
            errors.append("catalog env(s) missing from suite: " + ", ".join(missing))
        if extra:
            errors.append("suite env(s) outside formal boards: " + ", ".join(extra))
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one formal public X2Real ID evaluation board.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--board",
        default=_DEFAULT_BOARD,
        help="Formal board id. Default: artixon_arm6a_tabletop.",
    )
    parser.add_argument(
        "--subcategory",
        action="append",
        default=None,
        help="Leaderboard subcategory; comma list or repeated.",
    )
    parser.add_argument(
        "--robot",
        action="append",
        default=None,
        help=(
            "Paper robot key; comma list or repeated. Legacy EX001/X2Robot "
            "selectors are accepted as compatibility aliases."
        ),
    )
    parser.add_argument(
        "--gripper",
        action="append",
        default=None,
        help="Gripper key; comma list or repeated.",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Leaderboard task, package slug, or env task; comma list or repeated.",
    )
    parser.add_argument(
        "--env",
        "--env-name",
        dest="envs",
        action="append",
        default=None,
        help="Exact public environment name; comma list or repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=("id", "ID"),
        default="id",
        help="Public evaluation mode (ID only).",
    )
    parser.add_argument(
        "--split", default=None, help="Deprecated alias for --mode (ID only)."
    )
    parser.add_argument(
        "--seed",
        default=None,
        help=f"Master seed; integer or 'random'. Default {_DEFAULT_EVAL_SEED}.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Override episodes per selected case. Default: the board value.",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=None,
        help="Override episode_length_s and its 30 Hz max_steps budget.",
    )
    parser.add_argument(
        "--num-parallel-envs",
        type=int,
        default=None,
        help="Override parallel environments per selected case. Default: the board value.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Benchmark output directory; also receives a copy named eval_suite.yaml.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=("Continue each selected case from durable episode rows in <save-path>/<env>/evaluation_report.jsonl."),
    )
    parser.add_argument(
        "--list", action="store_true", help="Print matching cases and exit."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate all five boards, the 73-case suite, and rollout budgets.",
    )
    parser.add_argument(
        "--asset-check",
        action="store_true",
        help="Validate local public asset references, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the temporary suite and print launch argv.",
    )
    return parser


def _effective_budget(
    case: EvalCase,
    *,
    num_episodes: int | None,
    num_parallel_envs: int | None,
    timeout_s: int | None,
) -> tuple[int, int, int, int]:
    timeout = timeout_s if timeout_s is not None else case.timeout_s
    max_steps = timeout * _STEPS_PER_SECOND if timeout_s is not None else case.max_steps
    episodes = num_episodes if num_episodes is not None else case.num_episodes
    parallel = (
        num_parallel_envs if num_parallel_envs is not None else case.num_parallel_envs
    )
    return episodes, timeout, max_steps, parallel


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args(argv)
    try:
        _validate_passthrough(passthrough)
        mode = str(args.split if args.split is not None else args.mode).strip().lower()
        if mode != "id":
            raise ValueError(f"unsupported public evaluation mode {mode!r}; choose id")
        if args.num_episodes is not None:
            _positive_int(args.num_episodes, "--num-episodes")
        if args.timeout_s is not None:
            _positive_int(args.timeout_s, "--timeout-s")
        if args.num_parallel_envs is not None:
            _positive_int(args.num_parallel_envs, "--num-parallel-envs")
        if args.resume and str(args.seed or "").strip().lower() in {
            "random",
            "rand",
            "entropy",
        }:
            raise ValueError("--resume requires a deterministic integer --seed")
        seed = resolve_seed(args.seed)
        boards = load_boards()
        board = get_board(args.board, boards)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    if args.check:
        try:
            errors = check_suite(_X2REAL_CONFIG) + check_eval_catalog()
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            errors = [str(exc)]
        if errors:
            print("x2real eval check FAILED:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("x2real eval check OK (5 boards; 73 unique ID cases)")
        return 0

    if args.asset_check:
        try:
            errors = check_asset_references(_X2REAL_CONFIG)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            errors = [str(exc)]
        if errors:
            print("x2real eval asset check FAILED:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("x2real eval asset check OK (73 formal ID cases)")
        return 0

    try:
        board_cases = load_eval_cases(board.board_id, boards)
        requested_envs = {
            canonical_env_name(name) for name in (_name_set(args.envs) or set())
        }
        unknown_envs = sorted(
            requested_envs - {case.env for case in board_cases}
        )
        if unknown_envs:
            raise ValueError(
                f"unknown environment(s) for board {board.board_id}: "
                + ", ".join(unknown_envs)
            )
        matched = filter_eval_cases(
            board_cases,
            envs=args.envs,
            tasks=args.task,
            subcategories=args.subcategory,
            robots=args.robot,
            grippers=args.gripper,
        )
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.list:
        print(
            f"{'board':<{_BOARD_COLUMN_WIDTH}} {'subcategory':<24} {'gripper':<8} {'episodes':>8} "
            f"{'timeout':>7} {'steps':>7} {'parallel':>8} {'task':<28} env"
        )
        for case in matched:
            episodes, timeout, max_steps, parallel = _effective_budget(
                case,
                num_episodes=args.num_episodes,
                num_parallel_envs=args.num_parallel_envs,
                timeout_s=args.timeout_s,
            )
            print(
                f"{case.board:<{_BOARD_COLUMN_WIDTH}} {case.subcategory:<24} {case.gripper:<8} "
                f"{episodes:>8} {timeout:>7} {max_steps:>7} {parallel:>8} "
                f"{case.leaderboard_task:<28} {case.env}"
            )
        print(f"{len(matched)} case(s); board={board.board_id}; seed={seed}")
        return 0

    if not matched:
        print(
            f"ERROR: no ID evaluation cases matched board {board.board_id}",
            file=sys.stderr,
        )
        return 2

    effective_save_path = args.save_path or board.save_path
    resume_episode_counts: dict[str, int] = {}
    attempt_seed = seed
    try:
        if args.resume:
            resume_episode_counts = _load_resume_episode_counts(
                matched,
                save_path=effective_save_path,
            )
            for case in matched:
                episodes, _, _, _ = _effective_budget(
                    case,
                    num_episodes=args.num_episodes,
                    num_parallel_envs=args.num_parallel_envs,
                    timeout_s=args.timeout_s,
                )
                completed = resume_episode_counts[case.env]
                if completed > episodes:
                    raise ValueError(
                        f"{case.env}: evaluation report has {completed} episodes; requested target is {episodes}"
                    )
            attempt_seed = _derive_resume_attempt_seed(seed, resume_episode_counts)
        generated = materialize_run_suite(
            matched,
            board=board,
            num_episodes=args.num_episodes,
            num_parallel_envs=args.num_parallel_envs,
            timeout_s=args.timeout_s,
            save_path=effective_save_path,
            copy_to_save_path=args.save_path is not None,
            resume=args.resume,
            resume_episode_counts=resume_episode_counts,
            base_seed=seed,
            attempt_seed=attempt_seed,
        )
    except (OSError, TypeError, ValueError, RuntimeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    env_names = [case.env for case in matched]
    main_argv = _build_main_argv(
        generated,
        env_names,
        passthrough,
        seed=attempt_seed,
    )
    if args.dry_run:
        print(f"board: {board.board_id}")
        print(f"config: {generated}")
        print(f"cases: {len(matched)}")
        print(f"envs: {', '.join(env_names)}")
        print(
            f"episodes: {args.num_episodes if args.num_episodes is not None else board.num_episodes}"
        )
        timeout_label = args.timeout_s if args.timeout_s is not None else "board/csv"
        print(f"timeout_s: {timeout_label}")
        print(
            "parallel_envs: "
            f"{args.num_parallel_envs if args.num_parallel_envs is not None else board.num_parallel_envs}"
        )
        print(f"save_path: {effective_save_path}")
        print(f"resume: {args.resume}")
        if args.resume:
            print(f"resumed_episodes: {sum(resume_episode_counts.values())}")
            print(f"base_seed: {seed}")
        print(f"seed: {attempt_seed}")
        print(f"argv: {' '.join(main_argv)}")
        return 0

    print(
        f"INFO: public X2Real ID evaluation board={board.board_id} seed={attempt_seed} "
        f"resume={args.resume} resumed_episodes={sum(resume_episode_counts.values())} "
        f"cases={len(matched)} suite={generated}"
    )
    os.chdir(_REPO_ROOT)
    os.execv(sys.executable, [sys.executable, *main_argv])
    return 0  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
