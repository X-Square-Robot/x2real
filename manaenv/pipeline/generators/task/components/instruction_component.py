from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG
from manaenv.pipeline.generators.task.ref_text import render_ref_text, render_refs_in

_DEFAULT_VARIANT_LOG_KEY = "instruction_variant"


class InstructionComponent(TaskComponent):
    """Builds instruction / prompt configuration, with optional reset-randomized variants.

    Single-instruction form (byte-equivalent to before)::

        instruction_component:
          instruction: "Pick up @target.name and place it in @bin.name."
          topic: "pick_and_place"
          subtasks:
            - "grasp @target.name"
            - "move to @bin.name"

    Multi-variant form — several wordings of the SAME goal, one chosen per-env
    per-reset and the choice recorded as section log (``task_yaml_optimization.md``
    section 3)::

        instruction_component:
          instructions:
            - text: "Pick up @target.name."
              log: { instruction_variant: direct }
            - text: "Kindly relocate @target.name."
              log: { instruction_variant: polite }

    Inline ``@role.<attr>`` references in every text field are rendered against
    the scene-bound roles at finalize (see
    :mod:`manaenv.pipeline.generators.task.ref_text`). When more than one variant
    is declared, a ``kind="section"`` log-producer spec is emitted onto
    ``log_producers`` so the runtime picks one variant per env at reset, writes
    the rendered text into the per-env active-instruction store, and records
    ``{instruction_variant: <chosen>}`` under ``frame_idx == -1``. A single
    variant emits no section producer, so existing tasks are unchanged.
    """

    def generate(self, cfg: InstructionComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        roles = roles or {}
        variants = self._normalize_variants(cfg)
        rendered = [render_ref_text(variant["text"], roles) for variant in variants]

        result: dict[str, Any] = {
            "instruction": rendered[0] if rendered else "",
            "topic": render_ref_text(cfg.topic, roles),
            "subtasks": render_refs_in(list(cfg.subtasks), roles),
        }
        if len(variants) > 1:
            key = self._resolve_variant_log_key(variants)
            options = [{"value": variants[i]["log_value"], "rendered": rendered[i]} for i in range(len(variants))]
            spec: dict[str, Any] = {"kind": "section", "key": key, "options": options}
            self._attach_tensor_log(spec, variants)
            if cfg.seed is not None:
                spec["seed"] = cfg.seed
            result["log_producers"] = [spec]
        return result

    @staticmethod
    def _normalize_variants(cfg: InstructionComponentCFG) -> list[dict[str, Any]]:
        """Normalize the declared instruction(s) into ``[{text, log_key, log_value}]``.

        Accepts the legacy single ``instruction: str`` (one variant, no log) and
        the multi-variant ``instructions`` list whose items are either a plain
        text string or ``{text, log: {<key>: <value>}}``.
        """
        raw_variants = cfg.instructions or []
        if not raw_variants:
            text = cfg.instruction or ""
            return [{"text": text, "log_key": None, "log_value": None}] if text else []

        variants: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_variants):
            if isinstance(raw, str):
                variants.append({"text": raw, "log_key": None, "log_value": str(index)})
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"instruction variant #{index} must be a str or dict, got {type(raw).__name__}")
            text = raw.get("text", "")
            log = raw.get("log") or {}
            if not isinstance(log, dict):
                raise ValueError(f"instruction variant #{index} 'log' must be a dict, got {type(log).__name__}")
            if len(log) > 1:
                raise ValueError(
                    f"instruction variant #{index} 'log' must declare a single {{key: value}}; got keys {sorted(log)}"
                )
            log_key, log_value = next(iter(log.items()), (None, str(index)))
            tensor_log = raw.get("tensor_log") or {}
            if not isinstance(tensor_log, dict):
                raise ValueError(
                    f"instruction variant #{index} 'tensor_log' must be a dict, got {type(tensor_log).__name__}"
                )
            if len(tensor_log) > 1:
                raise ValueError(
                    f"instruction variant #{index} 'tensor_log' must declare a single {{key: value}}; "
                    f"got keys {sorted(tensor_log)}"
                )
            tensor_key, tensor_value = next(iter(tensor_log.items()), (None, None))
            variants.append(
                {
                    "text": text,
                    "log_key": log_key,
                    "log_value": log_value,
                    "tensor_key": tensor_key,
                    "tensor_value": tensor_value,
                }
            )
        return variants

    @staticmethod
    def _resolve_variant_log_key(variants: list[dict[str, Any]]) -> str:
        """All variants of one instruction record under one section-log key."""
        declared = {v["log_key"] for v in variants if v["log_key"] is not None}
        if len(declared) > 1:
            raise ValueError(
                f"instruction variants must share one log key (got {sorted(declared)}); "
                "all wordings of one instruction record the same section-log key"
            )
        return next(iter(declared), _DEFAULT_VARIANT_LOG_KEY)

    @staticmethod
    def _attach_tensor_log(spec: dict[str, Any], variants: list[dict[str, Any]]) -> None:
        """Attach an explicit fixed-width tensor log payload to section options."""
        declared = {variant["tensor_key"] for variant in variants if variant.get("tensor_key") is not None}
        if not declared:
            return
        if len(declared) > 1:
            raise ValueError(f"instruction tensor_log variants must share one key, got {sorted(declared)}")
        tensor_key = next(iter(declared))
        values = []
        for variant in variants:
            if variant.get("tensor_key") != tensor_key:
                raise ValueError("instruction tensor_log must be present on every variant when declared")
            value = variant.get("tensor_value")
            if not isinstance(value, list) or not value:
                raise ValueError(f"instruction tensor_log {tensor_key!r} must be a non-empty list of ints")
            values.append([int(item) for item in value])
        lengths = {len(item) for item in values}
        if len(lengths) != 1:
            raise ValueError(f"instruction tensor_log {tensor_key!r} values must all have the same length")
        for option, tensor_value in zip(spec["options"], values):
            option["tensor_value"] = tensor_value
        spec["tensor_key"] = tensor_key


@dataclass
class InstructionComponentCFG(TaskComponentCFG):

    class_type: type = InstructionComponent
    renders_text_refs = True  # instruction prose carries @role.<attr> interpolation
    instruction: str = ""
    instructions: list[Any] = field(default_factory=list)
    topic: str = ""
    subtasks: list[str] = field(default_factory=list)
    seed: int | None = None


def build_instruction_from_definition(
    definition: dict[str, Any] | None, roles: dict[str, Any] | None = None
) -> tuple[str, list[dict[str, Any]]]:
    """Run the instruction component declared on a composite/graph definition.

    Composite (graph) tasks do not flow ``component_sections`` through the normal
    per-node component path, so a graph-level instruction is applied here instead.
    Returns ``(default_prompt, log_producer_specs)``: the default prompt (first
    rendered wording, used by ``get_prompt`` until per-reset feeding is wired) and
    any ``kind="section"`` producer specs (empty for a single instruction). Pure
    (no sim import) so graph generators call it and stability tests exercise it.

    ``@ref`` in the instruction text is rendered against *roles* via the same
    ``render_ref_text`` path the per-node instruction uses; plain prose needs no
    roles. ``instruction_component`` may arrive either as a raw mapping
    (hand-built / ``yaml.safe_load``) or as an already-reflected
    ``InstructionComponentCFG`` (the reflective translator deep-reflects every
    registered CFG key, including those nested under ``definition``), so both
    forms are accepted here.
    """
    sections = (definition or {}).get("component_sections") or {}
    raw = sections.get("instruction_component") or sections.get("instruction")
    if isinstance(raw, InstructionComponentCFG):
        cfg = raw
    elif isinstance(raw, dict):
        cfg = InstructionComponentCFG(
            instruction=raw.get("instruction", ""),
            instructions=list(raw.get("instructions") or []),
            topic=raw.get("topic", ""),
            subtasks=list(raw.get("subtasks") or []),
            seed=raw.get("seed"),
        )
    else:
        return "", []
    result = InstructionComponent().generate(cfg, roles=roles or {})
    return result.get("instruction") or "", list(result.get("log_producers") or [])
