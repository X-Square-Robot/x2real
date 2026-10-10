from __future__ import annotations

from dataclasses import dataclass

NodeId = str
GroupId = str


@dataclass(frozen=True)
class DomainId:
    kind: str
    id: str

    @classmethod
    def node(cls, node_id: str) -> DomainId:
        return cls("node", str(node_id))

    @classmethod
    def group(cls, group_id: str) -> DomainId:
        return cls("group", str(group_id))

    def to_json(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}
