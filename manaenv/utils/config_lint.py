"""Static linter for manaenv ``defination`` YAML reference integrity.

Purpose
-------
The pipeline is full of *fallback* logic that masks broken config instead of
failing loudly:

* ``_resolve_value`` raises only at runtime, per-env, after the image is built
  (the ``89cd1bf5`` ``@table_contact_sensor`` regression).
* ``_find_product_by_generator_name`` **silently uses the first available
  product** when an assembly's ``generator_name`` matches nothing
  (assembly_factory.py) — a wrong-task-runs-quietly bug.
* ``_resolve_policy_ref`` **drops** an unknown ``policy_ref`` with a warning
  (translator.py).
* a scene whose ``definition.scene_type`` is missing is **skipped**.

This linter resolves the cross-file/cross-section reference graph from the YAML
alone (no Isaac Sim, no runtime) and surfaces dangling references and orphaned
definitions at commit time, before any fallback can hide them.

Reference graph
---------------
* **Task definition file** (top-level ``roles:`` / ``scene_constraint:`` /
  ``termination:`` / ``sensor_component`` / ``direct_task_generator``):
    - defined roles = ``roles:`` keys ∪ contact-sensor role names from
      ``sensor_component.sensors[]`` map keys (preferred) ∪ legacy
      ``scene_constraint.sensors[].output_role`` (deprecated)
    - referenced roles = every ``"@<name>"`` string
    - dangling ``@role`` → ERROR
* **Pipeline / entry file** (declares ``pipeline:`` or any ``*_factory:`` key):
    - section keys carry the ``*_orders`` suffix in production entries
      (``scene_orders`` / ``robot_orders`` / ``task_orders`` / ``assembly_orders``;
      the translator strips ``_orders`` — see reflective_translator.py). The bare
      names (``scene`` / ``robot`` / ``task`` / ``assembly``) are also accepted.
    - identities are collected *permissively* — the union of every form the
      reflective translator may assign — so a valid, resolvable file can never
      raise a false dangling-reference ERROR:
        * scene: every ``scene_type`` value ∪ ``"{name}_scene"`` ∪ ``name``
          (fragments without an explicit ``scene_generator.scene_type`` get
          ``scene_type = "{name}_scene"`` derived by the translator)
        * task:  every ``task_type`` value ∪ ``name``
        * robot: ``robot_generator.name`` ∪ every ``robot_type`` value ∪ ``name``
        * assembly: ``assembly_generator.name`` (or inline ``name``)
        * policy: include stem / inline ``name``
      (all resolved through ``include:`` exactly like Translator._resolve_include)
    - assembly ``{scene,robot,task}_generator_name`` must resolve to a known
      identity (numeric index selection is always valid) → ERROR
    - any ``env_ref`` (e.g. ``rollout[].env_ref``) must be a known assembly → ERROR
    - any ``policy_ref`` must be a known policy identity → ERROR
    - ``include:`` path must exist on disk → ERROR
    - a scene/task defined but referenced by no assembly → WARNING (orphan)

Pure stdlib + PyYAML — safe under the stability-test admission guard.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REFERENCE_PREFIX = "@"

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"

@dataclass(frozen=True)
class LintIssue:
    """A single config integrity problem."""

    severity: str
    category: str
    path: Path
    message: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.category}: {self.message}"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _walk_strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_strings(item)


def _collect_values_for_key(node: Any, key: str) -> list[Any]:
    """Collect every value mapped to *key* anywhere in a nested structure."""
    out: list[Any] = []
    if isinstance(node, Mapping):
        for k, v in node.items():
            if k == key:
                out.append(v)
            out.extend(_collect_values_for_key(v, key))
    elif isinstance(node, (list, tuple)):
        for item in node:
            out.extend(_collect_values_for_key(item, key))
    return out


# ---------------------------------------------------------------------------
# Role-closure (intra task document)
# ---------------------------------------------------------------------------


def collect_referenced_roles(node: Any) -> set[str]:
    """Collect role names referenced via ``"@<name>"`` strings."""
    return {s[1:] for s in _walk_strings(node) if s.startswith(REFERENCE_PREFIX) and len(s) > 1}


def _sensor_roles_from_sensor_component(block: Mapping[str, Any] | None) -> set[str]:
    """Collect role names published by a ``sensor_component`` sensors list."""
    defined: set[str] = set()
    if not isinstance(block, Mapping):
        return defined
    sensors = block.get("sensors")
    if not isinstance(sensors, list):
        return defined
    for entry in sensors:
        if isinstance(entry, Mapping):
            for name in entry.keys():
                if isinstance(name, str) and name:
                    defined.add(name)
    return defined


def _sensor_roles_from_legacy_scene_constraint(block: Mapping[str, Any] | None) -> set[str]:
    """Collect deprecated ``output_role`` names from a scene_constraint block."""
    defined: set[str] = set()
    if not isinstance(block, Mapping):
        return defined
    sensors = block.get("sensors")
    if not isinstance(sensors, list):
        return defined
    for sensor in sensors:
        if isinstance(sensor, Mapping):
            output_role = sensor.get("output_role")
            if isinstance(output_role, str) and output_role:
                defined.add(output_role)
    return defined


def collect_defined_roles(doc: Mapping[str, Any]) -> set[str]:
    """Collect roles a task doc declares (``roles:`` keys + sensor roles)."""
    defined: set[str] = set()

    roles = doc.get("roles")
    if isinstance(roles, Mapping):
        defined |= {str(k) for k in roles.keys()}

    defined |= _sensor_roles_from_legacy_scene_constraint(doc.get("scene_constraint"))
    defined |= _sensor_roles_from_sensor_component(doc.get("sensor_component"))

    dtg = doc.get("direct_task_generator")
    if isinstance(dtg, Mapping):
        if isinstance(dtg.get("roles"), Mapping):
            defined |= {str(k) for k in dtg["roles"].keys()}
        sections = dtg.get("component_sections")
        if isinstance(sections, Mapping):
            defined |= _sensor_roles_from_legacy_scene_constraint(sections.get("scene_constraint_component"))
            defined |= _sensor_roles_from_sensor_component(sections.get("sensor_component"))

    return defined


def find_dangling_role_references(doc: Mapping[str, Any]) -> list[str]:
    """Return sorted role names referenced but never defined in *doc*."""
    return sorted(collect_referenced_roles(doc) - collect_defined_roles(doc))


def _is_task_definition_doc(doc: Any) -> bool:
    if not isinstance(doc, Mapping):
        return False
    return any(k in doc for k in ("roles", "scene_constraint", "sensor_component", "termination"))


def _is_pipeline_doc(doc: Any) -> bool:
    """A mana entry file: declares the orchestrator ``pipeline`` DAG or any
    ``*_factory`` contract (``task_factory`` / ``scene_factory`` / ...).

    Production entry files use per-domain ``*_factory`` keys plus a ``pipeline``
    node graph; they have no bare ``factory`` key. Gating on these entry-level
    contracts (rather than on ``assembly``) keeps *included fragments* — which
    carry ``*_orders`` payload but no factory/pipeline contract and only resolve
    in their parent's include context — out of standalone graph-linting. A bare
    ``factory`` key (older/hand-written form) is still accepted.
    """
    if not isinstance(doc, Mapping):
        return False
    return "pipeline" in doc or "factory" in doc or any(isinstance(k, str) and k.endswith("_factory") for k in doc)


# ---------------------------------------------------------------------------
# Identity resolution (mirrors Translator._resolve_include)
# ---------------------------------------------------------------------------


def _unwrap_generator(entry: Any, section: str | None) -> Any:
    """Unwrap a ``{section}_generator`` envelope to its inner spec mapping.

    Production orders nest the payload one level down
    (``robot_orders: [{robot_generator: {include: ...}}]``); older/hand-written
    fragments and the linter's fixtures put ``include``/identity keys directly on
    the entry. Only unwrap when the inner value is a mapping so a bare entry is
    always returned unchanged.
    """
    if section is None or not isinstance(entry, Mapping):
        return entry
    inner = entry.get(f"{section}_generator")
    return inner if isinstance(inner, Mapping) else entry


def _deep_merge_override(base: dict[str, Any], override: Mapping[str, Any]) -> None:
    """Merge *override* into *base* in place, recursing into nested mappings.

    A shallow update would drop sibling keys the fragment supplied: an order that
    overrides only ``definition.metadata`` must not erase the fragment's
    ``definition.scene_type``, which is where the scene identity lives.
    """
    for key, value in override.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _deep_merge_override(existing, value)
        else:
            base[key] = value


def _resolve_entry(
    entry: Any, base_dir: Path, section: str | None = None
) -> tuple[Mapping[str, Any] | None, Path | None, Path | None]:
    """Resolve a section entry to (doc, include_path, missing_include_path).

    Returns the loaded mapping (inline or included), the include path if any,
    and a non-None ``missing`` path when an ``include`` does not exist. When
    *section* is given, a ``{section}_generator`` envelope is unwrapped first so
    nested production orders resolve their include the same as bare entries.
    """
    if not isinstance(entry, Mapping):
        return None, None, None
    entry = _unwrap_generator(entry, section)
    if not isinstance(entry, Mapping):
        return None, None, None
    inc = entry.get("include")
    include_paths: list[str] = []
    if isinstance(inc, str) and inc:
        include_paths = [inc]
    elif isinstance(inc, list):
        include_paths = [item for item in inc if isinstance(item, str) and item]
        if inc and not include_paths:
            return None, None, None
    if include_paths:
        merged: dict[str, Any] = {}
        last_resolved: Path | None = None
        for rel in include_paths:
            resolved = base_dir / rel
            if not resolved.is_file():
                return None, resolved, resolved
            try:
                doc = _load_yaml(resolved)
            except yaml.YAMLError:
                return None, resolved, None
            if not isinstance(doc, Mapping):
                return None, resolved, None
            _deep_merge_override(merged, doc)
            last_resolved = resolved
        assert last_resolved is not None
        # The entry's own keys override the included fragment's. A variant order
        # reuses a shared fragment and renames itself on top of it
        # (``{include: robot/ex001_6r...yaml, name: EX001-6R-G}``); keeping only the
        # fragment's identity would leave every assembly referencing the variant
        # name unresolvable.
        _deep_merge_override(merged, {k: v for k, v in entry.items() if k != "include"})
        merged.setdefault("name", last_resolved.stem)
        return merged, last_resolved, None
    return entry, None, None


def _section_entries(doc: Mapping[str, Any], base: str) -> list[Any]:
    """Return a section's list payload, honouring the ``*_orders`` suffix.

    Production entries name sections ``scene_orders`` / ``task_orders`` / ...;
    the reflective translator strips ``_orders``. Older/hand-written fragments
    (and the linter's own fixtures) use the bare name. Accept both.
    """
    for key in (f"{base}_orders", base):
        value = doc.get(key)
        if isinstance(value, list):
            return value
    return []


def _string_values_for_key(doc: Mapping[str, Any], key: str) -> set[str]:
    return {v for v in _collect_values_for_key(doc, key) if isinstance(v, str) and v}


def _scene_identities(doc: Mapping[str, Any]) -> set[str]:
    """Every identity a resolved scene fragment may answer to.

    ``scene_generator.scene_type`` when authored explicitly, otherwise the
    translator derives ``"{name}_scene"``; the bare ``name`` is kept too so the
    valid set is a superset and cannot false-flag a resolvable reference.
    """
    ids = _string_values_for_key(doc, "scene_type")
    name = doc.get("name")
    if isinstance(name, str) and name:
        ids.add(name)
        ids.add(f"{name}_scene")
    return ids


def _task_identities(doc: Mapping[str, Any]) -> set[str]:
    ids = _string_values_for_key(doc, "task_type")
    name = doc.get("name")
    if isinstance(name, str) and name:
        ids.add(name)
    return ids


def _robot_identities(doc: Mapping[str, Any]) -> set[str]:
    ids = _string_values_for_key(doc, "robot_type")
    # Inline robot orders carry the identity as ``robot_generator.name``.
    ids |= _string_values_for_key(doc, "name")
    return ids


def _assembly_spec(entry: Any) -> Mapping[str, Any] | None:
    """Unwrap an assembly order to its spec mapping.

    Production form nests under ``assembly_generator``; a bare mapping (inline
    fixture form) is used as-is.
    """
    if not isinstance(entry, Mapping):
        return None
    inner = entry.get("assembly_generator")
    if isinstance(inner, Mapping):
        return inner
    return entry


# ---------------------------------------------------------------------------
# Pipeline graph lint
# ---------------------------------------------------------------------------


_MAX_DOC_INCLUDE_DEPTH = 8


def _expand_doc_includes(doc: Mapping[str, Any], base_dir: Path, _depth: int = 0) -> Mapping[str, Any]:
    """Merge a document-level ``include:`` parent under *doc*'s own keys.

    A pipeline variant may carry only its overrides and inherit the rest
    (for example, a task-profile variant can include a base profile for its
    ``task_orders``). Without merging, the variant's assemblies reference
    identities the linter never sees. Child keys win over the parent's, matching
    the translator's override semantics. Depth-capped so a cyclic include chain
    cannot loop.
    """
    inc = doc.get("include")
    if not isinstance(inc, str) or not inc or _depth >= _MAX_DOC_INCLUDE_DEPTH:
        return doc
    parent_path = base_dir / inc
    if not parent_path.is_file():
        return doc
    try:
        parent = _load_yaml(parent_path)
    except yaml.YAMLError:
        return doc
    if not isinstance(parent, Mapping):
        return doc
    merged: dict[str, Any] = dict(_expand_doc_includes(parent, parent_path.resolve().parent, _depth + 1))
    merged.update(doc)
    return merged


def _lint_pipeline(doc: Mapping[str, Any], path: Path, base_dir: Path) -> list[LintIssue]:  # noqa: C901
    issues: list[LintIssue] = []
    own_sections = {base for base in ("scene", "task") if _section_entries(doc, base)}
    doc = _expand_doc_includes(doc, base_dir)

    # Union of every identity form per domain (permissive → no false ERRORs).
    scene_valid: set[str] = set()
    robot_valid: set[str] = set()
    task_valid: set[str] = set()
    policy_names: set[str] = set()
    # Per-definition (ids, origin) so orphan detection keys on the *entry*,
    # not on individual permissive aliases (else the bare ``name`` alias would
    # look orphaned even when its ``{name}_scene`` form is referenced).
    scene_defs: list[tuple[set[str], str]] = []
    task_defs: list[tuple[set[str], str]] = []
    # Raw generator_names actually consumed by assemblies (orphan detection).
    referenced_scene: set[str] = set()
    referenced_task: set[str] = set()

    def _process_section(section: str, id_fn, valid: set[str], defs: list[tuple[set[str], str]] | None) -> None:
        for entry in _section_entries(doc, section):
            resolved, inc_path, missing = _resolve_entry(entry, base_dir, section)
            if missing is not None:
                issues.append(
                    LintIssue(SEVERITY_ERROR, "include_path", path, f"{section} include not found: {missing}")
                )
                continue
            if not isinstance(resolved, Mapping):
                continue
            ids = id_fn(resolved)
            if not ids:
                if inc_path is not None:
                    issues.append(
                        LintIssue(
                            SEVERITY_WARNING,
                            f"{section}_identity_missing",
                            path,
                            f"{section} include {inc_path} declares no identity; "
                            "assemblies cannot reference it by name",
                        )
                    )
                continue
            valid |= ids
            if defs is not None:
                defs.append((ids, str(inc_path) if inc_path else f"{section} (inline)"))

    _process_section("scene", _scene_identities, scene_valid, scene_defs)
    _process_section("robot", _robot_identities, robot_valid, None)
    _process_section("task", _task_identities, task_valid, task_defs)

    # Policy identities (stem or inline name).
    for entry in _section_entries(doc, "policy"):
        resolved, inc_path, missing = _resolve_entry(entry, base_dir, "policy")
        if missing is not None:
            issues.append(LintIssue(SEVERITY_ERROR, "include_path", path, f"policy include not found: {missing}"))
            continue
        if isinstance(resolved, Mapping):
            policy_names |= _string_values_for_key(resolved, "name")

    # Assemblies: names + ``{scene,robot,task}_generator_name`` reference checks.
    assembly_names: set[str] = set()
    for spec in (_assembly_spec(e) for e in _section_entries(doc, "assembly")):
        if spec is None:
            continue
        asm_name = spec.get("name")
        if isinstance(asm_name, str) and asm_name:
            assembly_names.add(asm_name)

        for ref_key, valid, sink_ref, label in (
            ("scene_generator_name", scene_valid, referenced_scene, "scene"),
            ("robot_generator_name", robot_valid, None, "robot"),
            ("task_generator_name", task_valid, referenced_task, "task"),
        ):
            gen = spec.get(ref_key)
            if not isinstance(gen, str) or not gen:
                continue
            if gen.isdigit():  # index-based selection is always valid
                continue
            if sink_ref is not None:
                sink_ref.add(gen)
            if gen not in valid:
                issues.append(
                    LintIssue(
                        SEVERITY_ERROR,
                        f"assembly_{label}_ref",
                        path,
                        f"assembly '{asm_name or '?'}' references {ref_key} '{gen}' but no "
                        f"{label} defines it (known {label} identities: {sorted(valid)})",
                    )
                )

    # env references (e.g. rollout[].env_ref) → must be a known assembly.
    for env in sorted(_string_values_for_key(doc, "env_ref")):
        if env not in assembly_names:
            issues.append(
                LintIssue(
                    SEVERITY_ERROR,
                    "env_ref",
                    path,
                    f"env reference '{env}' is not a defined assembly (assemblies: {sorted(assembly_names)})",
                )
            )

    # policy_ref → must be a known policy identity (only when policies exist;
    # without a policy section we cannot know the identities, so stay silent).
    if policy_names:
        for v in sorted(_string_values_for_key(doc, "policy_ref")):
            if v not in policy_names:
                issues.append(
                    LintIssue(
                        SEVERITY_ERROR,
                        "policy_ref",
                        path,
                        f"policy_ref '{v}' is not a defined policy (policies: {sorted(policy_names)})",
                    )
                )

    # Orphans (弃置): a scene/task defined but consumed by no assembly. Only
    # check sections this document authors itself — a variant that inherits a
    # shared pool via document-level ``include:`` and binds a subset of it is a
    # deliberate pattern, not an orphan.
    if "scene" in own_sections:
        for ids, origin in scene_defs:
            if ids.isdisjoint(referenced_scene):
                issues.append(
                    LintIssue(
                        SEVERITY_WARNING,
                        "orphan_scene",
                        path,
                        f"scene {sorted(ids)} ({origin}) is defined but no assembly references it",
                    )
                )
    if "task" in own_sections:
        for ids, origin in task_defs:
            if ids.isdisjoint(referenced_task):
                issues.append(
                    LintIssue(
                        SEVERITY_WARNING,
                        "orphan_task",
                        path,
                        f"task {sorted(ids)} ({origin}) is defined but no assembly references it",
                    )
                )

    # NB: inline pipeline tasks are *not* role-closure checked here. In the real
    # schema they nest their ``roles:`` under a ``*_task_generator`` wrapper and
    # often compose roles across staged ``include:`` fragments, so a single-entry
    # closure check false-flags valid configs. Standalone task-definition files
    # (top-level ``roles:``) are still covered by the _is_task_definition_doc path.
    return issues


# ---------------------------------------------------------------------------
# Public file/tree entry points
# ---------------------------------------------------------------------------


def lint_file(path: str | Path) -> list[LintIssue]:
    """Lint a single YAML file, auto-detecting its kind."""
    path = Path(path)
    try:
        doc = _load_yaml(path) or {}
    except yaml.YAMLError as exc:
        return [LintIssue(SEVERITY_ERROR, "yaml_parse", path, f"YAML parse error: {exc}")]

    if not isinstance(doc, Mapping):
        return []

    issues: list[LintIssue] = []

    if _is_pipeline_doc(doc):
        issues.extend(_lint_pipeline(doc, path, base_dir=path.resolve().parent))

    # Standalone task definition role-closure (skip pipeline files, whose
    # inline tasks are handled above and whose top level has no roles).
    if _is_task_definition_doc(doc) and not _is_pipeline_doc(doc):
        for role in find_dangling_role_references(doc):
            issues.append(
                LintIssue(
                    SEVERITY_ERROR,
                    "role_reference",
                    path,
                    f"references @{role} but it is not defined "
                    "('roles:', sensor_component sensors keys, or legacy "
                    "scene_constraint.sensors[].output_role)",
                )
            )

    return issues


# --- Backward-compatible helpers (kept for existing imports/tests) ----------


def lint_task_file(path: str | Path) -> list[str]:
    """Lint one file; return role-closure error strings only (legacy shape)."""
    return [str(i) for i in lint_file(path) if i.category == "role_reference"]


def _is_scratch_eval(path: Path) -> bool:
    # gitignored leftover suites from gen_x2real_eval / pytest tmp writes
    return path.name.startswith("_x2real_eval_") and path.suffix == ".yaml"


def iter_yaml_files(root: str | Path) -> list[Path]:
    r = Path(root)
    if not r.is_dir():
        return []
    return sorted(p for p in r.rglob("*.yaml") if not _is_scratch_eval(p))


def iter_task_definition_files(task_root: str | Path) -> list[Path]:
    """All ``*.yaml`` under *task_root* (recursive)."""
    return iter_yaml_files(task_root)


def lint_tree(defination_root: str | Path) -> list[LintIssue]:
    """Lint every YAML under *defination_root* (recursive)."""
    issues: list[LintIssue] = []
    for path in iter_yaml_files(defination_root):
        issues.extend(lint_file(path))
    return issues


def _default_defination_root() -> Path:
    return Path(__file__).resolve().parents[1] / "defination"


def _lint_path(path: Path) -> list[LintIssue]:
    """Lint a single file or every YAML under a directory."""
    if path.is_file():
        return lint_file(path)
    if path.is_dir():
        return lint_tree(path)
    return [LintIssue(SEVERITY_ERROR, "path", path, f"path not found: {path}")]


def main(argv: Iterable[str] | None = None) -> int:
    """CLI: lint defination YAML reference integrity. Exit 1 on any ERROR."""

    parser = argparse.ArgumentParser(
        prog="manaenv-config-lint",
        description=(
            "Static linter for manaenv defination YAML reference integrity "
            "(role closure, assembly graph, include paths). Pure YAML — no Isaac Sim."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="YAML file or directory to lint (default: manaenv/defination)",
    )
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Treat WARNING issues as errors (non-zero exit)",
    )
    parsed = parser.parse_args(list(argv) if argv is not None else None)

    target = Path(parsed.path) if parsed.path else _default_defination_root()
    issues = _lint_path(target)
    errors = [i for i in issues if i.severity == SEVERITY_ERROR]
    warnings = [i for i in issues if i.severity == SEVERITY_WARNING]

    for issue in errors:
        print(issue)
    for issue in warnings:
        print(issue)

    fail = bool(errors) or (parsed.warnings_as_errors and warnings)
    print(
        f"[config_lint] {len(errors)} error(s), {len(warnings)} warning(s) under {target}"
        if (errors or warnings)
        else f"[config_lint] OK — no issues under {target}"
    )
    return 1 if fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
