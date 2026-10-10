from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class BaseArtifact:
    """Base class for all pipeline artifacts."""

    schema_version: str = "1.0"
    artifact_id: str = ""
    run_id: str = ""
    iteration: int = 0
    artifact_type: str = ""
    created_at: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseArtifact:
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @staticmethod
    def generate_id(prefix: str = "art") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()


class ArtifactLedger:
    """Append-only JSONL ledger that tracks all artifacts for a run."""

    _TYPE_REGISTRY: dict[str, type] = {}

    @classmethod
    def register_type(cls, artifact_type: str, artifact_cls: type) -> None:
        cls._TYPE_REGISTRY[artifact_type] = artifact_cls

    @classmethod
    def get_registered_type(cls, artifact_type: str) -> type | None:
        """Return the registered artifact class for ``artifact_type``, if any."""
        return cls._TYPE_REGISTRY.get(artifact_type)

    @classmethod
    def registered_types(cls) -> dict[str, type]:
        """Return a copy of the artifact type registry."""
        return dict(cls._TYPE_REGISTRY)

    def __init__(self, run_id: str, ledger_dir: str):
        self.run_id = run_id
        self.ledger_path = Path(ledger_dir) / "artifact_ledger.jsonl"
        self.artifacts: list[BaseArtifact] = []
        self._load()

    def register(self, artifact: BaseArtifact) -> None:
        self.artifacts.append(artifact)
        self._append(artifact)

    def query(
        self,
        iteration: int | None = None,
        artifact_type: str | None = None,
    ) -> list[BaseArtifact]:
        results = self.artifacts
        if iteration is not None:
            results = [a for a in results if a.iteration == iteration]
        if artifact_type is not None:
            results = [a for a in results if a.artifact_type == artifact_type]
        return results

    def manifest(self) -> dict[str, Any]:
        """Return a compact, machine-readable summary of ledger contents."""
        by_type: dict[str, int] = {}
        by_iteration: dict[str, int] = {}
        artifact_ids: list[str] = []
        for artifact in self.artifacts:
            by_type[artifact.artifact_type] = by_type.get(artifact.artifact_type, 0) + 1
            iteration_key = str(artifact.iteration)
            by_iteration[iteration_key] = by_iteration.get(iteration_key, 0) + 1
            if artifact.artifact_id:
                artifact_ids.append(artifact.artifact_id)
        return {
            "run_id": self.run_id,
            "ledger_path": str(self.ledger_path),
            "count": len(self.artifacts),
            "by_type": by_type,
            "by_iteration": by_iteration,
            "artifact_ids": artifact_ids,
        }

    def validate_graph(self) -> dict[str, Any]:
        """Validate minimal artifact graph integrity for agent control tools.

        The graph is intentionally schema-light: it checks identifiers,
        duplicate IDs, and common reference fields without requiring every
        artifact type to declare a custom dependency schema.
        """
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        seen: dict[str, BaseArtifact] = {}

        for artifact in self.artifacts:
            if not artifact.artifact_id:
                errors.append({"artifact_id": "", "error": "missing_artifact_id"})
                continue
            if artifact.artifact_id in seen:
                errors.append({"artifact_id": artifact.artifact_id, "error": "duplicate_artifact_id"})
            seen[artifact.artifact_id] = artifact
            if not artifact.artifact_type:
                errors.append({"artifact_id": artifact.artifact_id, "error": "missing_artifact_type"})

        known_ids = set(seen)
        for artifact in self.artifacts:
            if not artifact.artifact_id:
                continue
            data = artifact.to_dict()
            for field_name in ("evidence_refs", "kept_artifacts", "discarded_artifacts", "rollback_targets"):
                refs = data.get(field_name) or []
                if not isinstance(refs, list):
                    warnings.append(
                        {
                            "artifact_id": artifact.artifact_id,
                            "field": field_name,
                            "warning": "reference_field_not_list",
                        }
                    )
                    continue
                for ref in refs:
                    if ref and ref not in known_ids:
                        warnings.append(
                            {
                                "artifact_id": artifact.artifact_id,
                                "field": field_name,
                                "ref": ref,
                                "warning": "missing_reference",
                            }
                        )
            for field_name in ("source_hypothesis_id", "source_eval_id", "next_hypothesis_id", "idea_ref"):
                ref = data.get(field_name)
                if ref and ref not in known_ids:
                    warnings.append(
                        {
                            "artifact_id": artifact.artifact_id,
                            "field": field_name,
                            "ref": ref,
                            "warning": "missing_reference",
                        }
                    )

        return {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "warnings": warnings,
            "manifest": self.manifest(),
        }

    def _append(self, artifact: BaseArtifact) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(artifact.to_json() + "\n")

    def _load(self) -> None:
        if not self.ledger_path.exists():
            return
        with open(self.ledger_path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        raise TypeError(f"expected dict, got {type(data).__name__}")
                    art_type = data.get("artifact_type", "")
                    cls = self._TYPE_REGISTRY.get(art_type, BaseArtifact)
                    self.artifacts.append(cls.from_dict(data))
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
                    import warnings

                    warnings.warn(f"ArtifactLedger: skipping corrupt line {i}: {exc}", stacklevel=2)
