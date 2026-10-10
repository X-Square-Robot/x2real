"""Helpers for persisting evaluation records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _episode_payload(
    record: dict[str, Any],
    *,
    env_name: str,
    prompt: str,
) -> dict[str, Any]:
    payload = {
        "record_type": "episode",
        "env_name": env_name,
        "prompt": prompt,
    }
    payload.update({key: value for key, value in record.items() if key not in payload})
    return payload


def load_evaluation_episode_jsonl(
    input_path: str | Path,
    *,
    env_name: str | None = None,
) -> list[dict[str, Any]]:
    """Load durable episode rows, ignoring only a torn final JSONL row."""

    path = Path(input_path)
    if not path.exists():
        return []
    if not path.is_file():
        raise ValueError(f"evaluation JSONL path is not a regular file: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    last_nonempty = nonempty[-1] if nonempty else -1
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == last_nonempty:
                break
            raise ValueError(f"invalid evaluation JSONL at {path}:{index + 1}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"evaluation JSONL row must be an object at {path}:{index + 1}")

        record_type = str(payload.get("record_type") or "")
        row_env_name = str(payload.get("env_name") or "")
        if env_name and row_env_name != env_name:
            raise ValueError(
                f"evaluation JSONL environment mismatch at {path}:{index + 1}: "
                f"expected {env_name!r}, got {row_env_name!r}"
            )
        if record_type == "summary":
            continue
        if record_type != "episode":
            raise ValueError(f"unsupported evaluation JSONL record_type {record_type!r} at {path}:{index + 1}")

        episode_index = payload.get("episode_index")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int):
            raise TypeError(f"evaluation episode_index must be an integer at {path}:{index + 1}")
        if episode_index != len(records):
            raise ValueError(
                f"evaluation episode_index is not contiguous at {path}:{index + 1}: "
                f"expected {len(records)}, got {episode_index}"
            )
        record = dict(payload)
        for key in ("record_type", "env_name", "prompt"):
            record.pop(key, None)
        records.append(record)
    return records


def _write_jsonl_atomic(path: Path, payloads: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return str(path)


def write_evaluation_episode_jsonl(
    output_path: str | Path,
    *,
    records: list[dict[str, Any]],
    env_name: str,
    prompt: str = "",
) -> str:
    """Atomically rewrite only durable episode rows before a resumed append."""

    path = Path(output_path)
    payloads = [_episode_payload(record, env_name=env_name, prompt=prompt) for record in records]
    return _write_jsonl_atomic(path, payloads)


def append_evaluation_episode_jsonl(
    output_path: str | Path,
    *,
    record: dict[str, Any],
    env_name: str,
    prompt: str = "",
) -> str:
    """Atomically add one finalized episode record."""
    path = Path(output_path)
    records = load_evaluation_episode_jsonl(path, env_name=env_name)
    episode_index = record.get("episode_index")
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError(f"cannot append evaluation episode with non-integer index {episode_index!r}")
    if episode_index != len(records):
        raise ValueError(f"cannot append evaluation episode {episode_index!r} to {path}: expected index {len(records)}")
    records.append(dict(record))
    write_evaluation_episode_jsonl(
        path,
        records=records,
        env_name=env_name,
        prompt=prompt,
    )
    return str(path)


def write_evaluation_report_jsonl(
    output_path: str | Path,
    *,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    env_name: str,
    prompt: str = "",
) -> str:
    """Rewrite the full report (episodes + trailing summary).

    Mid-session streaming uses :func:`append_evaluation_episode_jsonl`; session-end
    collection rewrites via this helper so the file ends with a fresh summary.
    """
    path = Path(output_path)
    payloads = [_episode_payload(record, env_name=env_name, prompt=prompt) for record in records]
    summary_payload = {
        "record_type": "summary",
        "env_name": env_name,
        "prompt": prompt,
    }
    summary_payload.update({key: value for key, value in summary.items() if key not in summary_payload})
    payloads.append(summary_payload)
    return _write_jsonl_atomic(path, payloads)
