"""Local HDF5 metadata and artifact inspection, without simulation imports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py


def write_collection_metadata(path: str | Path, metadata: dict[str, Any], *, episode_name: str | None = None) -> None:
    """Attach the language instruction and action contract to saved demonstrations."""
    with h5py.File(path, "r+") as stream:
        data = stream["data"]
        data.attrs["manaenv_collection"] = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        for name, episode in data.items():
            if not name.startswith("demo_"):
                continue
            if episode_name is not None and name != episode_name:
                continue
            if episode_name is None and "language_instruction" in episode.attrs:
                continue
            episode.attrs["language_instruction"] = str(metadata.get("instruction", ""))
            episode.attrs["mana_action_metadata"] = json.dumps(metadata.get("action", {}), sort_keys=True)
        stream.flush()


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    """Read persisted episode counts and lengths instead of trusting loop counters."""
    with h5py.File(path, "r") as stream:
        data = stream.get("data")
        episodes = []
        if data is not None:
            for name, group in data.items():
                if name.startswith("demo_") and "actions" in group:
                    episodes.append(
                        {
                            "name": name,
                            "num_frames": int(group["actions"].shape[0]),
                            "success": bool(group.attrs.get("success", False)),
                        }
                    )
        return {
            "path": str(Path(path).resolve()),
            "num_episodes": len(episodes),
            "num_frames": sum(item["num_frames"] for item in episodes),
            "episodes": episodes,
        }
