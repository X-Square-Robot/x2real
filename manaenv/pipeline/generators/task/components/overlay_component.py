from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from manaenv.pipeline.generators.task.component import TaskComponent, TaskComponentCFG


class OverlayComponent(TaskComponent):
    """Declares which existing per-episode log keys are mirrored to the live overlay.

    The overlay does not own any fact: each entry only references a ``log_key``
    already produced by ``log_component`` / ``instruction_component`` and adds a
    display hint. YAML example (under ``component_sections``)::

        overlay_component:
          evaluation: true
          entries:
            - label: Target word
              log_key: target_word
              font_size: 18
              font_color: 0xFFFFDD44
            - log_key: instruction_variant

    ``evaluation`` gates whether the overlay still shows the built-in evaluation
    section (max score / term hits / stage progress / score). ``label`` defaults
    to ``log_key``; ``font_size`` / ``font_color`` are optional UI hints that fall
    back to the overlay default style. Labels are plain text (no ``@role``
    interpolation), so this component is not a text-ref section.

    Produces two plain-dict runtime cfgs that survive the assembly deepcopy and
    are consumed by :class:`manaenv.pipeline.evaluation.runtime.EvaluationRuntime`:

    * ``runtime_cfgs["overlay_entries"]`` -- ordered ``[{log_key, label, font_size, font_color}]``.
    * ``runtime_cfgs["overlay_cfg"]`` -- ``{evaluation: bool}``.
    """

    def generate(self, cfg: OverlayComponentCFG, roles: dict[str, Any] | None = None) -> dict[str, Any]:
        del roles
        compiled: list[dict[str, Any]] = []
        for index, raw in enumerate(cfg.entries):
            compiled.append(self._compile_entry(index, raw))
        return {
            "overlay_entries": compiled,
            "overlay_cfg": {"evaluation": bool(cfg.evaluation)},
        }

    @staticmethod
    def _compile_entry(index: int, raw: Any) -> dict[str, Any]:
        if isinstance(raw, str):
            return {"log_key": raw, "label": raw, "font_size": None, "font_color": None}
        if not isinstance(raw, dict):
            raise ValueError(f"overlay entry #{index} must be a str or dict, got {type(raw).__name__}")
        log_key = raw.get("log_key")
        if not log_key:
            raise ValueError(f"overlay entry #{index} requires a non-empty 'log_key'")
        # Explicit empty label means "value only" (no "Label: " prefix).
        if "label" in raw:
            label = "" if raw["label"] is None else str(raw["label"])
        else:
            label = str(log_key)
        return {
            "log_key": str(log_key),
            "label": label,
            "font_size": raw.get("font_size"),
            "font_color": raw.get("font_color"),
        }


@dataclass
class OverlayComponentCFG(TaskComponentCFG):
    """Overlay display configuration.

    ``entries`` lists the log keys to surface (str shorthand = ``log_key`` with a
    default label, or a ``{log_key, label?, font_size?, font_color?}`` dict).
    ``evaluation`` controls whether the built-in evaluation section still renders.
    """

    class_type: type = OverlayComponent
    evaluation: bool = True
    entries: list[Any] = field(default_factory=list)
