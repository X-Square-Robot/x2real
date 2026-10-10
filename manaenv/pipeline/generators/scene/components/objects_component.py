"""Objects component for scene objects.

Extends ``AssetBasedComponent`` with **pose randomisation** — when
``position_range_min/max`` (and optionally ``orientation_range_min/max``)
are configured, each ``generate()`` call samples a random pose within
the axis-aligned box / Euler range.

Asset selection is constrained at the **task generation** stage (via
``ai_object_tags`` in the YAML) so only appropriate objects ever reach
the scene factory.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from manaenv.utils.debug_print import manaprint
from manaenv.utils.runtime_type_spec import prepare_runtime_asset_class
from manaenv.utils.semantic_tags import apply_semantic_tags

from .asset_based_component import AssetBasedComponent, AssetBasedComponentCFG, _coerce_scale_triple

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry


def _strip_child_z_rotations_under_prim(root_prim_path: str) -> int:
    """Set ``xformOp:rotateXYZ`` Z to 0 under ``visual/`` and ``collision/`` children."""
    try:
        import omni.usd
        from pxr import Gf, Usd, UsdGeom
    except Exception:
        return 0

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return 0

    cleared = 0
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if "/visual/" not in path and "/collision/" not in path:
            continue
        xformable = UsdGeom.Xformable(prim)
        for op in xformable.GetOrderedXformOps() or ():
            if op.GetOpName() != "xformOp:rotateXYZ":
                continue
            val = op.Get()
            if val is None:
                continue
            x, y, z = float(val[0]), float(val[1]), float(val[2])
            if abs(z) < 1e-6:
                continue
            op.Set(Gf.Vec3f(x, y, 0.0))
            cleared += 1
    return cleared


def _disable_world_fixed_joints_under_prim(root_prim_path: str) -> int:
    """Disable ``PhysicsFixedJoint`` children that attach to the absolute world.

    Some art assets (e.g. ``rack_ea.usd``) weld the rigid root to world via a
    FixedJoint with empty ``body0``. Under an env Xform that joint fights
    env-local pose writes and prevents tabletop collision with dynamic objects.
    Kinematic spawn already pins the body; the world joint must be turned off.
    """
    try:
        import omni.usd
        from pxr import Sdf, Usd, UsdPhysics
    except Exception:
        return 0

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return 0
    root = stage.GetPrimAtPath(root_prim_path)
    if not root or not root.IsValid():
        return 0

    disabled = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "PhysicsFixedJoint":
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body0 = joint.GetBody0Rel().GetTargets() if joint.GetBody0Rel() else []
        if body0:
            continue
        enabled_attr = prim.GetAttribute("physics:jointEnabled")
        if enabled_attr and enabled_attr.IsValid() and enabled_attr.Get() is False:
            continue
        if enabled_attr and enabled_attr.IsValid():
            enabled_attr.Set(False)
        else:
            prim.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        disabled += 1
    return disabled


@dataclass
class ShelfObject:
    """One obj asset choice for a procedurally filled shelf slot.

    An empty ``asset_name`` is a first-class *empty-slot* state: when the
    weighted picker selects it, the slot is simply left bare. ``weight`` sets
    its relative selection probability, so "leave this slot empty 30% of the
    time" is expressed as an empty obj with the matching weight — no
    separate emptiness check anywhere in the pipeline.
    """

    asset_name: str = ""
    semantic_tags: list[Any] = field(default_factory=list)
    weight: float = 1.0


class ObjectsComponent(AssetBasedComponent):
    """Component for scene objects with randomised or procedural placement."""

    SHELF_FILL_GEN_MODE = "shelf_fill_objects_component"

    def __init__(self, cfg: ObjectsComponentCFG | None = None, asset_registry: AssetRegistry = None):
        super().__init__(cfg, asset_registry)

    # ------------------------------------------------------------------
    # Pose randomisation (override base-class fixed-pose resolvers)
    # ------------------------------------------------------------------

    def _resolve_position(self) -> list[float]:
        cfg = self.cfg
        if getattr(cfg, "position_range_min", None) and getattr(cfg, "position_range_max", None):
            return [random.uniform(lo, hi) for lo, hi in zip(cfg.position_range_min, cfg.position_range_max)]
        return list(cfg.initial_position)

    def _resolve_orientation(self) -> list[float]:
        cfg = self.cfg
        rmin = getattr(cfg, "orientation_range_min", None)
        rmax = getattr(cfg, "orientation_range_max", None)
        if rmin and rmax:
            roll = math.radians(random.uniform(rmin[0], rmax[0]))
            pitch = math.radians(random.uniform(rmin[1], rmax[1]))
            yaw = math.radians(random.uniform(rmin[2], rmax[2]))
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
            return [x, y, z, w]
        return list(cfg.initial_orientation)

    # ------------------------------------------------------------------
    # Procedural generation (generator)
    # ------------------------------------------------------------------

    def _generate_by_generator(self) -> list[object]:
        generator = str(getattr(self.cfg, "generator", "") or "")
        if not generator:
            manaprint("WARNING: ObjectsComponent requires explicit generator")
            return []

        if generator == self.SHELF_FILL_GEN_MODE:
            return self._generate_shelf_fill_objects()

        manaprint(f"WARNING: ObjectsComponent unknown generator '{generator}'")
        return []

    def _generate_shelf_fill_objects(self) -> list[object]:
        shelf_slots = self._resolve_shelf_slots()
        if not shelf_slots:
            manaprint(
                "WARNING: ObjectsComponent(gen_method=2, gen_mode=shelf_fill_objects_component): "
                "no shelf positions configured"
            )
            return []

        # 1) Declare the object pool (obj assets, incl. any empty options).
        object_pool = self._normalize_object_pool()
        # 2) Choose which slots are in play, then SOLVE the full layout
        #    (slot -> obj / empty) up front, before anything is created.
        selected_slots = self._select_shelf_slots(shelf_slots)
        layout = self._solve_shelf_layout(shelf_slots, selected_slots, object_pool)
        # 3) Instantiate the solved layout.
        return self._instantiate_shelf_layout(selected_slots, layout)

    def _solve_shelf_layout(
        self,
        shelf_slots: list[dict[str, Any]],
        selected_slots: list[dict[str, Any]],
        object_pool: list[ShelfObject],
    ) -> list[ShelfObject | None]:
        """Resolve every selected slot to a obj (or ``None`` for empty)
        BEFORE any asset is instantiated.

        This is the layout solver: mode-specific pre-computation (row-exclusive
        deal, shuffled deck) happens once here, then each slot is resolved. The
        returned list is aligned with ``selected_slots``; ``None`` marks a slot
        the solver chose to leave bare. Instantiation is a separate pass.
        """
        mode = str(getattr(self.cfg, "slot_candidate_mode", "random") or "random").strip().lower()
        shelf_grid = getattr(self.cfg, "shelf_grid", {}) or {}
        grid_x_count = max(0, int(shelf_grid.get("x_count", 0) or 0)) if isinstance(shelf_grid, dict) else 0
        shuffled_deck = self._build_shuffled_object_deck(object_pool, len(selected_slots), mode)
        # ``row_exclusive``: deal each row one distinct asset (no replacement)
        # once per solve, so every slot in a row shares that row's asset.
        row_assignment = (
            self._build_row_exclusive_assignment(shelf_slots, object_pool) if mode == "row_exclusive" else {}
        )

        layout: list[ShelfObject | None] = []
        for slot_ordinal, slot in enumerate(selected_slots):
            obj = self._select_object_for_slot(
                slot,
                object_pool=object_pool,
                mode=mode,
                grid_x_count=grid_x_count,
                shuffled_deck=shuffled_deck,
                slot_ordinal=slot_ordinal,
                row_assignment=row_assignment,
            )
            # An empty obj (empty asset_name) is the "leave this slot bare"
            # outcome — record it as None in the solved layout.
            layout.append(obj if (obj is not None and obj.asset_name) else None)
        return layout

    def _instantiate_shelf_layout(
        self,
        selected_slots: list[dict[str, Any]],
        layout: list[ShelfObject | None],
    ) -> list[object]:
        """Materialise the solved layout: one pass that turns each non-empty
        slot decision into asset instances. No selection logic lives here."""
        slot_offset = list(self.cfg.slot_offset)
        assets: list[object] = []
        for slot, obj in zip(selected_slots, layout):
            if obj is None:
                continue
            bottle_count = self._resolve_bottles_per_shelf()
            shelf_end = list(slot["position"])
            for bottle_idx in range(bottle_count):
                position = [float(base + bottle_idx * delta) for base, delta in zip(shelf_end, slot_offset)]
                position = self._apply_slot_position_jitter(position)
                semantic_tags = self._resolve_slot_semantic_tags(
                    obj.semantic_tags,
                    z_idx=slot.get("z_idx"),
                    x_idx=slot.get("x_idx"),
                )
                asset = self._instantiate_object_asset(
                    obj=obj,
                    position=position,
                    orientation=self._resolve_orientation(),
                    semantic_tags=semantic_tags,
                )
                if asset is not None:
                    assets.append(asset)
        return assets

    def _apply_slot_position_jitter(self, position: list[float]) -> list[float]:
        """Uniform XY jitter around each shelf slot (Z unchanged)."""
        jitter = float(getattr(self.cfg, "slot_xy_jitter", 0.0) or 0.0)
        if jitter <= 0.0 or len(position) < 2:
            return position
        return [
            float(position[0]) + random.uniform(-jitter, jitter),
            float(position[1]) + random.uniform(-jitter, jitter),
            *position[2:],
        ]

    def _resolve_slot_semantic_tags(
        self,
        base_tags: list[str],
        *,
        z_idx: int | None,
        x_idx: int | None,
    ) -> list[str]:
        tags = list(base_tags or [])
        row_indices = getattr(self.cfg, "grasp_target_row_indices", None) or []
        col_indices = getattr(self.cfg, "grasp_target_x_indices", None) or []
        try:
            normalized_rows = {int(value) for value in row_indices}
        except (TypeError, ValueError):
            normalized_rows = set()
        try:
            normalized_cols = {int(value) for value in col_indices}
        except (TypeError, ValueError):
            normalized_cols = set()
        row_ok = z_idx is not None and z_idx in normalized_rows
        col_ok = not normalized_cols or (x_idx is not None and x_idx in normalized_cols)
        if row_ok and col_ok and "drink_grasp_target" not in tags:
            tags.append("drink_grasp_target")
        return tags

    def _select_shelf_slots(self, shelf_slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        total = len(shelf_slots)
        min_count = max(0, int(getattr(self.cfg, "selected_shelves_min", total)))
        max_count = max(0, int(getattr(self.cfg, "selected_shelves_max", total)))
        min_count = min(min_count, total)
        max_count = min(max(max_count, min_count), total)
        if max_count == 0:
            return []
        count = random.randint(min_count, max_count)
        if count >= total:
            return shelf_slots
        return random.sample(shelf_slots, k=count)

    def _select_shelves(self, shelf_positions: list[list[float]]) -> list[list[float]]:
        total = len(shelf_positions)
        min_count = max(0, int(getattr(self.cfg, "selected_shelves_min", total)))
        max_count = max(0, int(getattr(self.cfg, "selected_shelves_max", total)))
        min_count = min(min_count, total)
        max_count = min(max(max_count, min_count), total)
        if max_count == 0:
            return []
        count = random.randint(min_count, max_count)
        if count >= total:
            return shelf_positions
        return random.sample(shelf_positions, k=count)

    def _resolve_bottles_per_shelf(self) -> int:
        min_count, max_count = self._resolve_bottles_per_shelf_bounds()
        if max_count <= 0:
            return 0
        return random.randint(min_count, max_count)

    def _resolve_bottles_per_shelf_bounds(self) -> tuple[int, int]:
        min_count = getattr(self.cfg, "bottles_per_shelf_min", None)
        max_count = getattr(self.cfg, "bottles_per_shelf_max", None)

        if min_count is None and max_count is None:
            fixed_count = max(0, int(getattr(self.cfg, "bottles_per_shelf", 0)))
            return fixed_count, fixed_count

        if min_count is None:
            min_count = getattr(self.cfg, "bottles_per_shelf", 0)
        if max_count is None:
            max_count = getattr(self.cfg, "bottles_per_shelf", min_count)

        min_count = max(0, int(min_count))
        max_count = max(min_count, int(max_count))
        return min_count, max_count

    def _resolve_shelf_slots(self) -> list[dict[str, Any]]:
        explicit_positions = [list(pos) for pos in getattr(self.cfg, "shelf_end_positions", []) if len(pos) == 3]
        if explicit_positions:
            return [{"position": pos, "z_idx": None, "x_idx": None} for pos in explicit_positions]

        shelf_grid = getattr(self.cfg, "shelf_grid", {}) or {}
        if not isinstance(shelf_grid, dict):
            return []

        origin = shelf_grid.get("origin", [])
        if not isinstance(origin, (list, tuple)) or len(origin) != 3:
            return []

        try:
            origin_x, origin_y, origin_z = (float(origin[0]), float(origin[1]), float(origin[2]))
            x_spacing = float(shelf_grid.get("x_spacing", 0.0))
            z_spacing = float(shelf_grid.get("z_spacing", 0.0))
            x_count = max(0, int(shelf_grid.get("x_count", 0)))
            z_count = max(0, int(shelf_grid.get("z_count", 0)))
        except (TypeError, ValueError):
            return []

        slots: list[dict[str, Any]] = []
        for z_idx in range(z_count):
            for x_idx in range(x_count):
                slots.append(
                    {
                        "position": [
                            origin_x + x_idx * x_spacing,
                            origin_y,
                            origin_z + z_idx * z_spacing,
                        ],
                        "z_idx": z_idx,
                        "x_idx": x_idx,
                    }
                )
        return slots

    def _resolve_shelf_positions(self) -> list[list[float]]:
        return [list(slot["position"]) for slot in self._resolve_shelf_slots()]

    def _select_object(self) -> ShelfObject:
        object_pool = self._normalize_object_pool()
        if object_pool:
            return random.choice(object_pool)
        return self._fallback_shelf_object()

    def _fallback_shelf_object(self) -> ShelfObject:
        fallback_name = str(getattr(self.cfg, "tags", "") or "")
        fallback_semantic_tags = list(getattr(self.cfg, "semantic_tags", []) or [])
        return ShelfObject(asset_name=fallback_name, semantic_tags=fallback_semantic_tags)

    def _build_shuffled_object_deck(
        self,
        object_pool: list[ShelfObject],
        slot_count: int,
        mode: str,
    ) -> list[ShelfObject]:
        if mode != "shuffled_deck" or not object_pool or slot_count <= 0:
            return []
        per_type, remainder = divmod(slot_count, len(object_pool))
        deck: list[ShelfObject] = []
        for idx, obj in enumerate(object_pool):
            copies = per_type + (1 if idx < remainder else 0)
            deck.extend([obj] * copies)
        random.shuffle(deck)
        return deck

    def _build_row_exclusive_assignment(
        self,
        shelf_slots: list[dict[str, Any]],
        object_pool: list[ShelfObject],
    ) -> dict[int, list[ShelfObject]]:
        """Deal each shelf row one distinct asset from ``object_pool`` (no replacement).

        Rows are mutually exclusive: no asset is assigned to two rows. The shared
        empty option (weight = ``empty_weight``) is appended to every row's pool so
        any row's slot can still be left bare without consuming a real asset. When
        there are more rows than assets, surplus rows get only the empty option
        (they stay bare unless ``empty_weight`` is 0, in which case they are empty
        by necessity since the pool is empty).

        Returns ``{z_idx: [asset_object?, empty_object?]}``.
        """
        # Real (non-empty) assets form the exclusive pool; explicit empty
        # object_pool are ignored here — emptiness is governed by ``empty_weight``.
        pool = [c for c in object_pool if c.asset_name]
        row_indices = sorted({slot["z_idx"] for slot in shelf_slots if slot.get("z_idx") is not None})
        if not row_indices:
            return {}

        deck = list(pool)
        if getattr(self.cfg, "shuffle_row_assignment", True):
            random.shuffle(deck)

        try:
            empty_weight = max(0.0, float(getattr(self.cfg, "empty_weight", 0.0) or 0.0))
        except (TypeError, ValueError):
            empty_weight = 0.0
        empty_object = ShelfObject(asset_name="", weight=empty_weight) if empty_weight > 0.0 else None

        assignment: dict[int, list[ShelfObject]] = {}
        for offset, z_idx in enumerate(row_indices):
            row_pool: list[ShelfObject] = []
            if offset < len(deck):
                row_pool.append(deck[offset])  # one distinct asset per row
            if empty_object is not None:
                row_pool.append(empty_object)
            assignment[int(z_idx)] = row_pool
        return assignment

    def _select_row_exclusive_object(
        self,
        slot: dict[str, Any],
        *,
        row_assignment: dict[int, list[ShelfObject]],
    ) -> ShelfObject:
        """Pick from the slot's row pool: the row's exclusive asset vs. shared empty,
        weighted. Falls back to empty when the row has no asset (surplus rows)."""
        z_idx = slot.get("z_idx")
        pool = row_assignment.get(int(z_idx)) if z_idx is not None else None
        if not pool:
            return ShelfObject(asset_name="")
        return self._weighted_choice(pool) or ShelfObject(asset_name="")

    def _select_object_for_slot(
        self,
        slot: dict[str, Any],
        *,
        object_pool: list[ShelfObject],
        mode: str,
        grid_x_count: int,
        shuffled_deck: list[ShelfObject] | None = None,
        slot_ordinal: int = 0,
        row_assignment: dict[int, list[ShelfObject]] | None = None,
    ) -> ShelfObject:
        if mode == "row_exclusive":
            return self._select_row_exclusive_object(slot, row_assignment=row_assignment or {})
        if not object_pool:
            return self._fallback_shelf_object()
        if mode == "grid_mapped":
            mapped = self._select_grid_mapped_object(slot, object_pool=object_pool)
            if mapped is not None:
                return mapped
        if mode == "shuffled_deck" and shuffled_deck:
            if 0 <= slot_ordinal < len(shuffled_deck):
                return shuffled_deck[slot_ordinal]
            return random.choice(object_pool)
        if mode == "round_robin":
            z_idx = slot.get("z_idx")
            x_idx = slot.get("x_idx")
            if z_idx is not None and x_idx is not None and grid_x_count > 0:
                slot_index = int(z_idx) * int(grid_x_count) + int(x_idx)
            else:
                slot_index = 0
            return object_pool[slot_index % len(object_pool)]
        # Default ``random`` mode: weighted draw so a obj's ``weight``
        # (including an empty-asset obj) sets its per-slot probability.
        return self._weighted_choice(object_pool) or object_pool[0]

    def _select_grid_mapped_object(
        self,
        slot: dict[str, Any],
        *,
        object_pool: list[ShelfObject],
    ) -> ShelfObject | None:
        """Pick a obj from ``grid_drink_regions`` using slot (z_idx, x_idx).

        A matched region is resolved into a weighted obj pool that may
        include empty-slot entries (empty ``asset_name``). Drawing an empty
        entry leaves the cell bare — so per-row asset type AND per-row empty
        probability are both expressed through the same weighted draw.
        """
        z_idx = slot.get("z_idx")
        x_idx = slot.get("x_idx")
        if z_idx is None or x_idx is None:
            return None

        objects_by_name = {item.asset_name: item for item in object_pool if item.asset_name}
        regions = getattr(self.cfg, "grid_drink_regions", None) or []
        for region in regions:
            if not isinstance(region, dict):
                continue
            z_indices = region.get("z_indices", region.get("z_idx"))
            x_indices = region.get("x_indices", region.get("x_idx"))
            try:
                z_set = {int(value) for value in (z_indices if isinstance(z_indices, (list, tuple)) else [z_indices])}
                x_set = {int(value) for value in (x_indices if isinstance(x_indices, (list, tuple)) else [x_indices])}
            except (TypeError, ValueError):
                continue
            if int(z_idx) not in z_set or int(x_idx) not in x_set:
                continue
            pool = self._region_object_pool(region, objects_by_name)
            if not pool:
                return None
            return self._weighted_choice(pool)
        return None

    def _region_object_pool(
        self,
        region: dict[str, Any],
        objects_by_name: dict[str, ShelfObject],
    ) -> list[ShelfObject]:
        """Resolve a matched region into a weighted obj pool.

        Two forms are supported:
          * ``object_pool``: a sub-list of asset_name/weight mappings; an empty
            ``asset_name`` entry is the empty-slot state with its own weight.
          * short form: a single ``asset_name`` (+ optional ``weight`` /
            ``semantic_tags``); empty ``asset_name`` means always-empty.
        """
        raw_objects = region.get("object_pool")
        pool: list[ShelfObject] = []
        if isinstance(raw_objects, (list, tuple)):
            for raw in raw_objects:
                if isinstance(raw, dict):
                    pool.append(self._resolve_region_object(raw, objects_by_name))
        else:
            pool.append(self._resolve_region_object(region, objects_by_name))
        return pool

    def _resolve_region_object(
        self,
        raw: dict[str, Any],
        objects_by_name: dict[str, ShelfObject],
    ) -> ShelfObject:
        """Build one region obj, inheriting semantic_tags from the named
        global obj when the region omits them. Empty ``asset_name`` is
        preserved as an empty-slot obj."""
        cand = self._object_from_mapping(raw)
        if cand.asset_name and not cand.semantic_tags:
            known = objects_by_name.get(cand.asset_name)
            if known is not None:
                return ShelfObject(
                    asset_name=cand.asset_name,
                    semantic_tags=list(known.semantic_tags),
                    weight=cand.weight,
                )
        return cand

    def _normalize_object_pool(self) -> list[ShelfObject]:
        normalized: list[ShelfObject] = []
        for raw in getattr(self.cfg, "object_pool", []) or []:
            if isinstance(raw, ShelfObject):
                normalized.append(raw)
                continue
            if isinstance(raw, dict):
                normalized.append(self._object_from_mapping(raw))
        for raw_name in getattr(self.cfg, "object_pool_asset_names", []) or []:
            if isinstance(raw_name, str) and raw_name:
                normalized.append(
                    ShelfObject(asset_name=raw_name, semantic_tags=list(getattr(self.cfg, "semantic_tags", []) or []))
                )
        return normalized

    @staticmethod
    def _object_from_mapping(raw: dict[str, Any]) -> ShelfObject:
        """Build a obj from a YAML mapping.

        An empty ``asset_name`` is kept as a valid *empty-slot* obj; its
        ``weight`` then governs how often a slot is left bare via the same
        weighted draw used for real assets.
        """
        asset_name = str(raw.get("asset_name", "") or "")
        semantic_tags = list(raw.get("semantic_tags", []) or [])
        try:
            weight = max(0.0, float(raw.get("weight", 1.0)))
        except (TypeError, ValueError):
            weight = 1.0
        return ShelfObject(asset_name=asset_name, semantic_tags=semantic_tags, weight=weight)

    @staticmethod
    def _weighted_choice(object_pool: list[ShelfObject]) -> ShelfObject | None:
        """Pick one obj by ``weight``; falls back to uniform when weights sum to 0."""
        if not object_pool:
            return None
        weights = [max(0.0, float(getattr(c, "weight", 1.0))) for c in object_pool]
        total = sum(weights)
        if total <= 0.0:
            return random.choice(object_pool)
        return random.choices(object_pool, weights=weights, k=1)[0]

    def _instantiate_object_asset(
        self,
        *,
        obj: ShelfObject,
        position: list[float],
        orientation: list[float],
        semantic_tags: list[str] | None = None,
    ) -> object | None:
        from isaaclab_arena.utils.pose import Pose

        asset_name = obj.asset_name
        if not asset_name:
            manaprint(f"WARNING: No name/tags specified for {self.__class__.__name__}")
            return None

        try:
            asset_class = self._asset_registry.get_asset_by_name(asset_name)
            runtime_semantic_tags = list(getattr(self.cfg, "semantic_tags", []) or []) + list(
                semantic_tags or obj.semantic_tags or []
            )
            runtime_plan = prepare_runtime_asset_class(asset_class, runtime_semantic_tags)
            asset_instance = runtime_plan.asset_class(**runtime_plan.init_kwargs)
            asset_instance.set_initial_pose(Pose(position_xyz=tuple(position), rotation_xyzw=tuple(orientation)))

            cfg_scale = _coerce_scale_triple(self.cfg.scale)
            if cfg_scale != (1.0, 1.0, 1.0) and hasattr(asset_instance, "scale"):
                asset_instance.scale = cfg_scale

            self._customize_asset(asset_instance)
            resolved_tags = list(semantic_tags) if semantic_tags is not None else list(obj.semantic_tags or [])
            self._apply_semantic_tags(asset_instance, resolved_tags)
            return asset_instance
        except Exception as e:
            manaprint(f"ERROR: Failed to instantiate procedural object '{asset_name}': {e}")
            return None

    # ------------------------------------------------------------------
    # Asset customisation hook
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_isaac_semantic_class(asset: object, class_tag: str) -> None:
        """Attach Isaac spawn ``("class", label)`` so sim/replicator can label prims at spawn."""
        if not class_tag:
            return
        cfg = getattr(asset, "object_cfg", None)
        spawn = getattr(cfg, "spawn", None) if cfg is not None else None
        if spawn is None or not hasattr(spawn, "semantic_tags"):
            return
        existing = list(getattr(spawn, "semantic_tags", None) or [])
        filtered: list[Any] = []
        for entry in existing:
            if isinstance(entry, (tuple, list)) and len(entry) >= 1 and entry[0] == "class":
                continue
            filtered.append(entry)
        filtered.append(("class", class_tag))
        spawn.semantic_tags = filtered

    def _resolve_spawn_class_tag(self, resolved_tags: list[str]) -> str | None:
        """Pick Isaac class string: ``isaac_semantic_class`` cfg, else first ``semantic_tags`` entry."""
        cfg_cls = getattr(self.cfg, "isaac_semantic_class", None)
        if isinstance(cfg_cls, str) and cfg_cls.strip():
            return cfg_cls.strip()
        if resolved_tags:
            head = str(resolved_tags[0]).strip()
            return head or None
        return None

    def _apply_semantic_tags(self, asset: object, semantic_tags: Any = None) -> None:
        sem = semantic_tags if semantic_tags is not None else (getattr(self.cfg, "semantic_tags", None) or [])
        resolved = apply_semantic_tags(asset, sem)
        class_tag = self._resolve_spawn_class_tag([tag.name for tag in resolved])
        if class_tag:
            self._apply_isaac_semantic_class(asset, class_tag)

    def _customize_asset(self, asset: object) -> None:
        """Customize object after creation."""
        init_object_cfg = getattr(asset, "_init_object_cfg", None)
        if callable(init_object_cfg):
            asset.object_cfg = init_object_cfg()
        self._apply_mass_override(asset)
        self._apply_friction_override(asset)
        self._apply_kinematic_override(asset)
        self._apply_strip_child_z_rotation_override(asset)
        self._apply_collision_approximation_override(asset)
        self._apply_semantic_tags(asset)

    def _apply_strip_child_z_rotation_override(self, asset: object) -> None:
        """Zero authored child ``rotateXYZ`` Z under visual/collision before PhysX cook.

        Some assets (e.g. ``rack_d``) put +90° on mesh children while layout yaws the
        root. Collision approximations cooked at the root then disagree with the
        rendered hang tube. Baking that Z into the root pose (scene YAML) and
        clearing child Z keeps visual and cooked collision in one frame.
        """
        if not bool(getattr(self.cfg, "strip_child_z_rotation", False)):
            return
        cfg = getattr(asset, "object_cfg", None)
        spawn = getattr(cfg, "spawn", None) if cfg is not None else None
        if spawn is None or not hasattr(spawn, "func"):
            manaprint("WARNING: strip_child_z_rotation ignored; asset has no spawn.func")
            return

        original_func = spawn.func
        asset_label = str(getattr(asset, "name", "") or getattr(self.cfg, "asset", "") or "?")

        def _spawn_and_strip(prim_path, spawn_cfg, *args, **kwargs):
            prim = original_func(prim_path, spawn_cfg, *args, **kwargs)
            n = _strip_child_z_rotations_under_prim(str(prim_path))
            if n:
                manaprint(
                    f"INFO: strip_child_z_rotation cleared {n} child Z rotate(s) on " f"'{asset_label}' ({prim_path})"
                )
            return prim

        spawn.func = _spawn_and_strip

    def _apply_collision_approximation_override(self, asset: object) -> None:
        """Wrap spawn so mesh collision approximation is rewritten before PhysX cooks.

        Particle cloth runs on GPU and cannot collide with non-GPU-compatible
        convex meshes (PhysX: "Non-GPU-compatible convex mesh is not able to
        collide with particle system"). Setting ``collision_approximation`` in
        scene YAML (e.g. ``convexHull`` / ``sdf``) forces a GPU-friendly cook.
        """
        approximation = str(getattr(self.cfg, "collision_approximation", "") or "").strip()
        if not approximation:
            return
        cfg = getattr(asset, "object_cfg", None)
        spawn = getattr(cfg, "spawn", None) if cfg is not None else None
        if spawn is None or not hasattr(spawn, "func"):
            manaprint(f"WARNING: collision_approximation={approximation} ignored; asset has no spawn.func")
            return
        try:
            from isaaclab.sim import schemas
            from isaaclab.sim.schemas import schemas_cfg
        except Exception as exc:
            manaprint(f"WARNING: collision_approximation skipped (isaaclab schemas unavailable): {exc}")
            return

        cfg_map = {
            "convexHull": schemas_cfg.ConvexHullPropertiesCfg,
            "convexDecomposition": schemas_cfg.ConvexDecompositionPropertiesCfg,
            "meshSimplification": schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
            "none": schemas_cfg.TriangleMeshPropertiesCfg,
            "sdf": schemas_cfg.SDFMeshPropertiesCfg,
            "boundingCube": schemas_cfg.BoundingCubePropertiesCfg,
            "boundingSphere": schemas_cfg.BoundingSpherePropertiesCfg,
        }
        cfg_cls = cfg_map.get(approximation)
        if cfg_cls is None:
            manaprint(f"WARNING: Unknown collision_approximation={approximation!r}; " f"valid={sorted(cfg_map)}")
            return

        mesh_cfg = cfg_cls()
        original_func = spawn.func
        asset_label = str(getattr(asset, "name", "") or getattr(self.cfg, "asset", "") or "?")

        def _spawn_with_approximation(prim_path, spawn_cfg, *args, **kwargs):
            prim = original_func(prim_path, spawn_cfg, *args, **kwargs)
            try:
                schemas.modify_mesh_collision_properties(str(prim_path), mesh_cfg)
                manaprint(
                    f"INFO: Set collision_approximation={approximation} on '{asset_label}' "
                    f"({prim_path}) for particle/GPU cloth compatibility"
                )
            except Exception as exc:
                manaprint(f"WARNING: Failed to set collision_approximation={approximation} " f"on {prim_path}: {exc}")
            return prim

        spawn.func = _spawn_with_approximation

    def _apply_mass_override(self, asset: object) -> None:
        """Apply spawn_mass from YAML config to the asset's spawn cfg."""
        spawn_mass = getattr(self.cfg, "spawn_mass", None)
        if spawn_mass is None:
            return
        cfg = getattr(asset, "object_cfg", None)
        if cfg is None:
            return
        spawn = getattr(cfg, "spawn", None)
        if spawn is None:
            return
        try:
            import isaaclab.sim as sim_utils

            mass_kg = float(spawn_mass)
            spawn.mass_props = sim_utils.MassPropertiesCfg(mass=mass_kg)
            asset_name = getattr(asset, "name", "<unnamed>")
            manaprint(f"INFO: spawn_mass override applied: name={asset_name} mass={mass_kg:.6g} kg")
        except Exception as e:
            manaprint(f"WARNING: Failed to apply spawn_mass={spawn_mass}: {e}")

    def _apply_kinematic_override(self, asset: object) -> None:
        """Spawn kinematic and disable world-anchored FixedJoints when ``kinematic`` is set.

        Matches the HSSD/static-prop pattern: ``kinematic_enabled`` +
        ``disable_gravity`` so PhysX never moves the body under contact or
        gravity. Semantic tag ``fixed`` alone does **not** freeze physics.

        Kinematic alone is also not enough for assets that ship a
        ``PhysicsFixedJoint`` with empty ``body0`` (weld to absolute world):
        that joint fights env-local pose writes and breaks collision with
        dynamic tabletop objects.

        The asset is marked with ``_mana_kinematic`` so downstream code that
        rebuilds ``object_cfg`` can re-assert flags via
        :func:`reapply_kinematic_override`.
        """
        if not getattr(self.cfg, "kinematic", False):
            return
        asset.__dict__["_mana_kinematic"] = True
        _set_kinematic_rigid_props(asset)
        _wrap_kinematic_spawn_disable_world_joints(asset, self.cfg)

    def _apply_friction_override(self, asset: object) -> None:
        """Apply spawn_static_friction / spawn_dynamic_friction from YAML."""
        sf = getattr(self.cfg, "spawn_static_friction", None)
        df = getattr(self.cfg, "spawn_dynamic_friction", None)
        if sf is None and df is None:
            return
        cfg = getattr(asset, "object_cfg", None)
        if cfg is None:
            return
        spawn = getattr(cfg, "spawn", None)
        if spawn is None:
            return
        try:
            import isaaclab.sim as sim_utils

            existing = getattr(spawn, "physics_material", None)
            if existing is not None:
                if sf is not None:
                    existing.static_friction = float(sf)
                if df is not None:
                    existing.dynamic_friction = float(df)
            else:
                spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
                    static_friction=float(sf if sf is not None else 1.0),
                    dynamic_friction=float(df if df is not None else 1.0),
                )
            manaprint(f"INFO: Friction override applied: static={sf}, dynamic={df}")
        except Exception as e:
            manaprint(f"WARNING: Failed to apply friction override: {e}")


def reapply_kinematic_override(asset: object) -> None:
    """Re-assert kinematic rigid props after an ``object_cfg`` rebuild.

    ``asset._init_object_cfg()`` rebuilds the spawn cfg from the asset class
    defaults and silently drops the kinematic override that
    ``ObjectsComponent`` applied earlier. Any caller that rebuilds
    ``object_cfg`` must call this afterwards.
    """
    if getattr(asset, "_mana_kinematic", False):
        _set_kinematic_rigid_props(asset)
        _wrap_kinematic_spawn_disable_world_joints(asset, cfg=None)


def _set_kinematic_rigid_props(asset: object) -> None:
    cfg = getattr(asset, "object_cfg", None)
    spawn = getattr(cfg, "spawn", None) if cfg is not None else None
    if spawn is None:
        manaprint(
            f"WARNING: kinematic=True but asset {getattr(asset, 'name', '?')!r} "
            f"has no object_cfg.spawn; screw/prop will stay dynamic"
        )
        return
    try:
        from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

        from manaenv.terms.events import bind_kinematic_pose_reset_event

        rigid = getattr(spawn, "rigid_props", None)
        if rigid is not None:
            rigid.kinematic_enabled = True
            if hasattr(rigid, "disable_gravity"):
                rigid.disable_gravity = True
        else:
            spawn.rigid_props = PhysxRigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            )
        bind_kinematic_pose_reset_event(asset)
        # Drop Arena's fixed-pose reset term entirely for kinematic props. That
        # term always zeros root velocity; ManaEnv ``reset_scene_to_default`` and
        # schema-vNext placement already restore pose without the PhysX error.
        disable_reset = getattr(asset, "disable_reset_pose", None)
        if callable(disable_reset):
            disable_reset()
        manaprint(
            f"INFO: kinematic override applied to {getattr(asset, 'name', '?')!r} "
            f"(kinematic_enabled + disable_gravity"
            f"{'; Arena pose-reset disabled' if callable(disable_reset) else ''})"
        )
    except Exception as e:
        manaprint(f"WARNING: Failed to apply kinematic override: {e}")


def _wrap_kinematic_spawn_disable_world_joints(asset: object, cfg: object | None) -> None:
    """Wrap spawn.func once to clear world-anchored FixedJoints after USD load."""
    object_cfg = getattr(asset, "object_cfg", None)
    spawn = getattr(object_cfg, "spawn", None) if object_cfg is not None else None
    if spawn is None or not hasattr(spawn, "func"):
        return
    if getattr(spawn, "_mana_kinematic_world_joint_wrap", False):
        return
    original_func = spawn.func
    asset_label = str(getattr(asset, "name", "") or "")
    if not asset_label and cfg is not None:
        asset_label = str(getattr(cfg, "asset", "") or "")
    if not asset_label:
        asset_label = "?"

    def _spawn_kinematic_and_disable_world_joints(prim_path, spawn_cfg, *args, **kwargs):
        prim = original_func(prim_path, spawn_cfg, *args, **kwargs)
        n = _disable_world_fixed_joints_under_prim(str(prim_path))
        if n:
            manaprint(
                f"INFO: kinematic spawn disabled {n} world FixedJoint(s) on "
                f"'{asset_label}' ({prim_path}) so env pose/collision work"
            )
        return prim

    spawn.func = _spawn_kinematic_and_disable_world_joints
    spawn._mana_kinematic_world_joint_wrap = True


@dataclass
class ObjectsComponentCFG(AssetBasedComponentCFG):
    """Configuration for scene objects component.

    Pose randomisation fields (optional):
        ``position_range_min/max``
            Axis-aligned box [min_xyz, max_xyz].  When both are set,
            each ``generate()`` samples a uniform random position.

        ``orientation_range_min/max``
            Euler angles in degrees [roll, pitch, yaw].  When both are
            set, a random Euler triplet is sampled and converted to an
            XYZW quaternion.

    """

    class_type: type = ObjectsComponent
    spawn_mass: float | None = None
    spawn_static_friction: float | None = None
    spawn_dynamic_friction: float | None = None
    #: Spawn the rigid body with physics:kinematicEnabled so physics never moves it.
    kinematic: bool = False
    #: Post-spawn mesh collision approximation rewrite (``convexHull``, ``sdf``, …).
    #: Needed when particle cloth must collide with an asset whose authored convex
    #: mesh is not GPU-compatible (PhysX: "Non-GPU-compatible convex mesh…").
    collision_approximation: str | None = None
    #: Clear authored child ``rotateXYZ`` Z under visual/collision before cook so
    #: collision approximations share the root frame with the rendered mesh
    #: (bake that yaw into the scene root pose instead).
    strip_child_z_rotation: bool = False
    semantic_tags: list[str] = field(default_factory=list)
    #: Isaac spawn ``semantic_tags`` class label (``("class", ...)``). When unset, the first
    #: entry in ``semantic_tags`` is used so YAML task tags still drive segmentation class.
    isaac_semantic_class: str | None = None
    # Pose randomisation — empty means "use fixed initial_position/orientation"
    position_range_min: list[float] = field(default_factory=list)
    position_range_max: list[float] = field(default_factory=list)
    orientation_range_min: list[float] = field(default_factory=list)
    orientation_range_max: list[float] = field(default_factory=list)
    # Procedural object generator, e.g. ``shelf_fill_objects_component``.
    generator: str = ""
    # Procedural shelf-fill fields used by
    # ``generator: shelf_fill_objects_component``
    bottles_per_shelf: int = 3
    bottles_per_shelf_min: int | None = None
    bottles_per_shelf_max: int | None = None
    selected_shelves_min: int = 1
    selected_shelves_max: int = 1
    slot_offset: list[float] = field(default_factory=lambda: [0.09, 0.0, 0.0])
    #: Per-slot uniform XY jitter (meters); each bottle samples independently in [-j, +j].
    slot_xy_jitter: float = 0.0
    shelf_end_positions: list[list[float]] = field(default_factory=list)
    shelf_grid: dict[str, Any] = field(default_factory=dict)
    object_pool_asset_names: list[str] = field(default_factory=list)
    object_pool: list[Any] = field(default_factory=list)
    #: ``random`` per slot; ``round_robin`` cycles by column; ``shuffled_deck`` evenly deals then shuffles;
    #: ``grid_mapped`` picks from ``grid_drink_regions`` by shelf (z_idx, x_idx);
    #: ``row_exclusive`` deals each ROW one distinct asset from ``object_pool`` without
    #: replacement (rows are mutually exclusive), with a shared empty option per row.
    slot_candidate_mode: str = "random"
    # Maps shelf grid cells to specific objects when ``slot_candidate_mode: grid_mapped``.
    grid_drink_regions: list[Any] = field(default_factory=list)
    #: ``row_exclusive`` only. Relative weight of the shared empty option that every
    #: row carries (so any row's slot can be left bare). The empty option is NOT drawn
    #: from the exclusive pool, so it never consumes a real asset. 0.0 = rows never empty.
    empty_weight: float = 0.0
    #: ``row_exclusive`` only. Shuffle the row->asset assignment each ``generate()`` call
    #: (per-episode variety). When False, assets are dealt to rows in declared order.
    shuffle_row_assignment: bool = True
    # 0-based shelf row indices (``shelf_grid`` z_idx) that receive ``drink_grasp_target``.
    grasp_target_row_indices: list[int] = field(default_factory=list)
    # Optional 0-based column filter (``shelf_grid`` x_idx). Empty means all columns in grasp rows.
    grasp_target_x_indices: list[int] = field(default_factory=list)
