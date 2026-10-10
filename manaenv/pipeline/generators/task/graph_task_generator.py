"""Graph task generator.

Single-node graphs delegate to their child generator. Multi-node graphs build
child tasks, finalize them independently, and forge a composite ManaTask that
owns the solved dependency graph.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from manaenv.pipeline.generators.task.graph_spec import parse_graph_spec
from manaenv.pipeline.generators.task.graph_task_forge import forge_graph_task_type
from manaenv.pipeline.generators.task.recursive_composite import RecursiveCompositeTaskMixin
from manaenv.pipeline.generators.task.ref_text import collect_text_refs
from manaenv.pipeline.generators.task_generator import TaskGenerationCompilation, TaskGenerator, TaskGeneratorCFG
from manaenv.pipeline.graph.mana_graph import ManaGraph

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.task_base import ManaTask


@dataclass
class GraphTaskGenerationCompilation(TaskGenerationCompilation):
    """Graph-task compilation: carries the parsed dependency graph."""

    graph: ManaGraph | None = None


class GraphTaskGenerator(TaskGenerator):
    """Compose N atomic-task nodes (+ dependency edges) into one ManaTask + graph."""

    def __init__(self, cfg: GraphTaskGeneratorCFG, **kwargs: object) -> None:
        super().__init__(cfg)
        self._ext_model_api = kwargs.get("ext_model_api")
        # Children in graph-solve order — for every graph, one node or many.
        self._multi_nodes: list[tuple[str, TaskGenerator, Any]] = []
        self._multi_graph: ManaGraph | None = None

    def compile(self, definition: dict[str, Any] | None = None, **kwargs: object) -> GraphTaskGenerationCompilation:
        """Parse nodes/edges into a ManaGraph (pure; no scene/sim)."""
        del kwargs
        graph = parse_graph_spec(getattr(self.cfg, "nodes", None), getattr(self.cfg, "edges", None))
        if len(graph) == 0:
            raise ValueError("GraphTaskGenerator requires at least one node")
        roles = dict(getattr(self.cfg, "roles", None) or {})
        self.roles = dict(roles)
        return GraphTaskGenerationCompilation(
            definition=dict(definition or self.cfg.definition or {}),
            roles=roles,
            graph=graph,
        )

    def _build_sub_generator(self, sub_cfg: TaskGeneratorCFG) -> TaskGenerator:
        """Instantiate a node's sub-generator from its reflected CFG."""
        impl = getattr(sub_cfg, "class_type", None)
        if impl is None:
            raise TypeError(f"graph node sub-generator CFG {type(sub_cfg).__name__} has no class_type")
        return impl(sub_cfg, ext_model_api=self._ext_model_api)

    def instancing(
        self,
        compiled_cfg: GraphTaskGenerationCompilation,
        task: ManaTask | None = None,
    ) -> ManaTask:
        """Build the per-node task(s).

        Every graph — one node or many — forges a composite ManaTask holding live child
        tasks. A one-node graph used to short-circuit and hand back the bare child, which
        gave it an EMPTY graph: a single-node graph with no nodes. It got no stage machine,
        no ``dag_*`` extras and no per-stage evaluation, while a two-node graph got all
        three. N=1 is not a special case; it is the smallest graph.
        """
        del task
        graph = compiled_cfg.graph
        if graph is None or len(graph) == 0:
            raise ValueError("GraphTaskGenerator.instancing requires a non-empty graph")
        return self._instance_multi_node(graph, compiled_cfg.definition, compiled_cfg.roles)

    def _mana_task_base(self):  # overridable for sim-free tests
        from isaaclab_arena.tasks.task_base import ManaTask

        return ManaTask

    def _task_cfg_cls(self):  # overridable for sim-free tests
        from isaaclab_arena.tasks.task_base import TaskCFG

        return TaskCFG

    def _instance_multi_node(
        self,
        graph: ManaGraph,
        definition: dict[str, Any],
        graph_roles: dict[str, Any],
    ) -> "ManaTask":
        # Build un-finalized child tasks; scene/robot binding happens in finalize().
        order = graph.solve().order
        nodes: list[tuple[str, TaskGenerator, Any]] = []
        for node_id in order:
            child_cfg = copy.deepcopy(graph.get_node(node_id).payload)
            child_roles = dict(graph_roles)
            child_roles.update(dict(getattr(child_cfg, "roles", None) or {}))
            child_cfg.roles = child_roles
            sub_gen = self._build_sub_generator(child_cfg)
            nodes.append((node_id, sub_gen, sub_gen.generate()))
        self._multi_nodes = nodes
        self._multi_graph = graph

        forged_cls = forge_graph_task_type(
            graph_name=str(definition.get("task_type") or "graph"),
            base_cls=self._mana_task_base(),
            node_types=tuple(nid for nid, _, _ in nodes),
            mixins=(RecursiveCompositeTaskMixin,),
        )
        # Compile a provisional graph-level instruction; @refs are rendered after
        # child roles are scene-resolved in finalize().
        from manaenv.pipeline.generators.task.components.instruction_component import build_instruction_from_definition

        default_prompt, instruction_log_specs = build_instruction_from_definition(definition, roles={})
        composite_runtime_cfgs: dict[str, Any] = {}
        if instruction_log_specs:
            composite_runtime_cfgs["log_producers"] = instruction_log_specs
        # Graph lives on the ManaTask instance, not TaskCFG (upstream TaskCFG
        # has no ``graph`` field). Same contract as the single-node path.
        task_cfg = self._task_cfg_cls()(
            episode_length_s=definition.get("episode_length_s"),
            prompt=definition.get("prompt") or default_prompt,
            runtime_cfgs=composite_runtime_cfgs,
            roles=dict(graph_roles),
            definition=dict(definition or {}),
        )
        composite = forged_cls(cfg=task_cfg)
        composite.graph = graph
        composite._composite_children = [(nid, child) for nid, _, child in nodes]
        composite._composite_edges = [(edge.src, edge.dst) for edge in graph.edges]
        # Node ``score`` attrs weight the composite evaluation leaves.
        composite._composite_node_scores = {
            nid: float(graph.get_node(nid).attrs.get("score", 1.0)) for nid, _, _ in nodes
        }
        self._results.append(composite)
        return composite

    def finalize(
        self,
        task: ManaTask,
        scene: Scene,
        robot: EmbodimentBase,
        **kwargs: object,
    ) -> ManaTask:
        """Finalize each node against scene/robot.

        No shape to dispatch on any more: every graph product is a composite holding its
        nodes as live children. (The old ``hasattr(task, "_composite_children")`` test could
        not survive the convergence anyway — every task carries that attribute now, so it
        answered True for everything and silently routed one-node graphs down the wrong path.)
        """
        del kwargs
        return self._finalize_multi_node(task, scene, robot)

    @staticmethod
    def _remap_deferred_for_clone(sub_gen, original_child, cloned_child) -> None:
        """Copy a child generator's deferred sections onto the cloned child id."""
        deferred = getattr(sub_gen, "_deferred_component_sections", None)
        if isinstance(deferred, dict) and id(original_child) in deferred:
            deferred[id(cloned_child)] = copy.deepcopy(deferred[id(original_child)])

    def _finalize_multi_node(self, task, scene, robot):
        # Finalize cloned children by node id and remap each child's deferred sections.
        by_id = {nid: (sub_gen, orig) for nid, sub_gen, orig in self._multi_nodes}
        finalized_children: list[tuple[str, Any]] = []
        for nid, cloned_child in list(getattr(task, "_composite_children", [])):
            sub_gen, original_child = by_id[nid]
            self._remap_deferred_for_clone(sub_gen, original_child, cloned_child)
            finalized_children.append((nid, sub_gen.finalize(cloned_child, scene=scene, robot=robot)))
        task._composite_children = finalized_children
        # Keep graph payloads queryable as finalized child ManaTasks.
        graph = getattr(task, "graph", None)
        if graph is not None:
            for nid, child in finalized_children:
                try:
                    graph.get_node(nid).payload = child
                except (KeyError, AttributeError):
                    pass
        # Render graph-level @refs once child roles are scene-resolved.
        self._render_composite_runtime_sections(task, scene, finalized_children)
        # Ensure graph-level ee_contact_against reaches embodiment EE sensors
        # (children without scene_cfg used to skip this bind).
        from manaenv.terms.scene import bind_ee_contact_sensor_object_filters

        graph_roles: dict[str, Any] = {}
        for _nid, child in finalized_children:
            child_roles = getattr(child, "roles", None)
            if isinstance(child_roles, dict):
                graph_roles.update(child_roles)
        definition_roles = getattr(getattr(task, "cfg", None), "roles", None)
        if isinstance(definition_roles, dict):
            graph_roles = {**definition_roles, **graph_roles}
        # Prefer live task.roles when present (already scene-resolved).
        if isinstance(getattr(task, "roles", None), dict) and task.roles:
            graph_roles = {**graph_roles, **task.roles}
        bind_ee_contact_sensor_object_filters({}, graph_roles, robot, scene)
        # Multi-node composites need their graph-level runtime_sample bridge here.
        from manaenv.pipeline.generators.scene_layout_solver.manaenv_bridge.task_runtime import (
            compile_layout_runtime_events_for_task,
        )

        compile_layout_runtime_events_for_task(task, scene)
        return task

    def _render_composite_runtime_sections(self, task, scene, finalized_children) -> None:
        """Re-render graph-level runtime sections after child roles resolve to assets."""
        from manaenv.pipeline.generators.task.components.event_component import EventComponent, EventComponentCFG
        from manaenv.pipeline.generators.task.components.instruction_component import build_instruction_from_definition
        from manaenv.pipeline.generators.task.components.log_component import LogComponent, LogComponentCFG
        from manaenv.pipeline.generators.task.components.overlay_component import OverlayComponent, OverlayComponentCFG

        definition = getattr(task.cfg, "definition", None) or {}
        sections = (definition.get("component_sections") or {}) if isinstance(definition, dict) else {}
        if not sections:
            return

        roles: dict[str, Any] = {ref: ref for ref in collect_text_refs(sections)}
        for _nid, child in finalized_children:
            child_roles = getattr(child, "roles", None)
            if isinstance(child_roles, dict):
                roles.update(child_roles)  # already-resolved asset objects win over tags
        roles = self._resolve_roles_from_scene(roles, scene)

        # Update cfg.runtime_cfgs in place so the task mirror sees the final spec.
        runtime_cfgs = getattr(task.cfg, "runtime_cfgs", None)
        if not isinstance(runtime_cfgs, dict):
            runtime_cfgs = {}
            task.cfg.runtime_cfgs = runtime_cfgs

        log_specs: list[dict[str, Any]] = []
        instruction_section = sections.get("instruction_component") or sections.get("instruction")
        if instruction_section is not None:
            default_prompt, instruction_log_specs = build_instruction_from_definition(definition, roles=roles)
            task.cfg.prompt = definition.get("prompt") or default_prompt
            log_specs.extend(instruction_log_specs)

        log_section = sections.get("log_component") or sections.get("log")
        if isinstance(log_section, LogComponentCFG):
            log_cfg = log_section
        elif isinstance(log_section, dict):
            log_cfg = LogComponentCFG(entries=dict(log_section.get("entries") or {}))
        else:
            log_cfg = None
        if log_cfg is not None:
            result = LogComponent().generate(log_cfg, roles=roles)
            log_specs.extend(list(result.get("log_producers") or []))

        event_section = sections.get("event_component") or sections.get("event")
        if isinstance(event_section, EventComponentCFG):
            event_cfg = event_section
        elif isinstance(event_section, dict):
            event_cfg = EventComponentCFG(terms=dict(event_section.get("terms") or {}))
        else:
            event_cfg = None
        if event_cfg is not None and event_cfg.terms:
            result = EventComponent().generate(event_cfg, roles=roles)
            incoming_events = result.get("event_terms")
            existing_events = runtime_cfgs.get("event_terms")
            if incoming_events is not None and existing_events is not None:
                from isaaclab_arena.utils.configclass import combine_configclass_instances

                runtime_cfgs["event_terms"] = combine_configclass_instances(
                    "GraphRuntimeEventsCfg", existing_events, incoming_events
                )
            elif incoming_events is not None:
                runtime_cfgs["event_terms"] = incoming_events

        overlay_section = sections.get("overlay_component") or sections.get("overlay")
        if isinstance(overlay_section, OverlayComponentCFG):
            overlay_cfg = overlay_section
        elif isinstance(overlay_section, dict):
            overlay_cfg = OverlayComponentCFG(
                evaluation=overlay_section.get("evaluation", True),
                entries=list(overlay_section.get("entries") or []),
            )
        else:
            overlay_cfg = None
        if overlay_cfg is not None:
            overlay_result = OverlayComponent().generate(overlay_cfg, roles=roles)
            runtime_cfgs["overlay_entries"] = overlay_result.get("overlay_entries") or []
            runtime_cfgs["overlay_cfg"] = overlay_result.get("overlay_cfg") or {}
        else:
            runtime_cfgs.pop("overlay_entries", None)
            runtime_cfgs.pop("overlay_cfg", None)

        if log_specs:
            runtime_cfgs["log_producers"] = log_specs
        else:
            runtime_cfgs.pop("log_producers", None)
        if isinstance(getattr(task, "runtime_cfgs", None), dict) and task.runtime_cfgs is not runtime_cfgs:
            task.runtime_cfgs = runtime_cfgs


@dataclass
class GraphTaskGeneratorCFG(TaskGeneratorCFG):
    """CFG for the unified graph task workflow.

    ``nodes`` / ``edges`` stay raw (list of dicts) through reflection so
    ``parse_graph_spec`` can reflect each node's sub-generator block on demand.
    """

    class_type: type[TaskGenerator] | None = GraphTaskGenerator
    nodes: list[Any] = field(default_factory=list)
    edges: list[Any] = field(default_factory=list)
