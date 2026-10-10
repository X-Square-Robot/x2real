from __future__ import annotations

from collections.abc import Callable

from manaenv.pipeline.registry.resolver import resolve_func

TASK_TERM_MODULES: dict[str, str] = {
    "command": "manaenv.terms.commands",
    "curriculum": "manaenv.terms.curriculum",
    "domain_randomization": "manaenv.terms.domain_randomization",
    "event": "manaenv.terms.events",
    "metric": "manaenv.terms.metrics",
    "observation": "manaenv.terms.observations",
    "reward": "manaenv.terms.rewards",
    "scene_constraint": "manaenv.terms.scene",
    "termination": "manaenv.terms.terminations",
}


def resolve_task_term(component: str, name: str) -> Callable:
    """Resolve a task term function through the shared lazy-compatible resolver."""
    if component == "event" and name == "mdp.reset_scene_to_default":
        return resolve_func("reset_scene_to_default", TASK_TERM_MODULES[component])

    if name.startswith("mdp."):
        return resolve_func(name[4:], "isaaclab.envs.mdp")

    try:
        default_module = TASK_TERM_MODULES[component]
    except KeyError as exc:
        raise KeyError(f"Unknown task term component: {component!r}") from exc
    return resolve_func(name, default_module)
