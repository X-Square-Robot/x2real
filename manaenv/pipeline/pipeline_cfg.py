"""Pipeline orchestration CFGs."""

from __future__ import annotations

from dataclasses import dataclass, field

from manaenv.pipeline.runtime_cfg import PipelineRuntimeCFG


@dataclass
class PipelineSelectionCFG:
    """Selection metadata for multi-task pipeline YAMLs."""

    active_task_idx: int = 0


@dataclass
class PipelineDagNodeCFG:
    """One execution node in the pipeline DAG.

    YAML spells each node explicitly as ``pipeline_dag_node:`` so the key
    reflects to this CFG; ``cfg_key`` references a top-level YAML section
    reflected by the translator, and the node carries only DAG metadata.
    """

    cfg_key: str = ""
    id: str = ""
    deps: dict[str, str] = field(default_factory=dict)
    orders: list[str] = field(default_factory=list)
    after: list[str] = field(default_factory=list)
    foreach: str | None = None
    runtime: str | None = None

    def __post_init__(self) -> None:
        if not self.cfg_key:
            raise ValueError(
                "pipeline.pipeline_dag.nodes: each node requires a cfg_key referencing a top-level CFG section"
            )
        self.id = self.id or self.cfg_key


@dataclass
class PipelineDagSolution:
    """Solved DAG views consumed by the evaluation pipeline."""

    order: list[str]
    levels: list[list[str]]
    split_nodes: list[str]


@dataclass
class PipelineDagCFG:
    """Pipeline execution DAG declaration."""

    nodes: list[PipelineDagNodeCFG] = field(default_factory=list)

    def solve(self) -> PipelineDagSolution:
        """Validate and solve the DAG into topological order and levels."""
        nodes_by_id: dict[str, PipelineDagNodeCFG] = {}
        for node in self.nodes:
            if node.id in nodes_by_id:
                raise ValueError(f"pipeline.pipeline_dag.nodes: duplicate node id {node.id!r}")
            nodes_by_id[node.id] = node

        incoming: dict[str, set[str]] = {node_id: set(node.after) for node_id, node in nodes_by_id.items()}
        for node_id, deps in incoming.items():
            unknown = sorted(dep for dep in deps if dep not in nodes_by_id)
            if unknown:
                raise ValueError(f"pipeline.pipeline_dag.nodes.{node_id}: unknown after node(s): {unknown}")

        ready = [node_id for node_id in nodes_by_id if not incoming[node_id]]
        order: list[str] = []
        levels: list[list[str]] = []

        while ready:
            level = list(ready)
            levels.append(level)
            order.extend(level)
            ready = []
            for solved_id in level:
                for node_id in nodes_by_id:
                    if solved_id not in incoming[node_id]:
                        continue
                    incoming[node_id].remove(solved_id)
                    if not incoming[node_id] and node_id not in order and node_id not in ready:
                        ready.append(node_id)

        if len(order) != len(nodes_by_id):
            cycle_nodes = sorted(node_id for node_id, deps in incoming.items() if deps)
            raise ValueError(f"pipeline.pipeline_dag.nodes contains a cycle involving: {cycle_nodes}")

        split_nodes = [node.id for node in self.nodes if node.foreach]
        return PipelineDagSolution(order=order, levels=levels, split_nodes=split_nodes)


@dataclass
class ReproducibilityCFG:
    """Single entropy budget for a pipeline run.

    ``master_seed`` set → all subsystems derive deterministically.
    ``master_seed`` omitted / null → OS entropy everywhere (never silent ``0``).
    CLI ``--seed`` overrides this field when publishing process state.

    Pipeline setup publishes this seed before reflecting ``scene_orders`` so
    schema-vNext materialization (background / asset pools) inherits it.
    """

    master_seed: int | None = None


@dataclass
class PipelineCFG:
    """Top-level pipeline orchestration configuration."""

    pipeline_runtime: PipelineRuntimeCFG | None = None
    pipeline_dag: PipelineDagCFG = field(default_factory=PipelineDagCFG)
    reproducibility: ReproducibilityCFG | None = None
