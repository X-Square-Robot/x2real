"""Asset-based component for scene generation.

Provides common functionality for scene components that need to create
assets from the asset registry (backgrounds, objects, destinations).
"""

from __future__ import annotations

import importlib
import secrets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from manaenv.pipeline.generators.scene.component import SceneComponent, SceneComponentCFG
from manaenv.utils.debug_print import manaprint

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry


def _coerce_scale_triple(raw: object) -> tuple[float, float, float]:
    """Normalize YAML / cfg scale to a float3 (supports scalar ``0.8`` or ``[sx,sy,sz]``)."""
    if isinstance(raw, (int, float)):
        s = float(raw)
        return (s, s, s)
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    manaprint(f"WARNING: Invalid scene object scale {raw!r}; using (1.0, 1.0, 1.0)")
    return (1.0, 1.0, 1.0)


@dataclass
class AssetBasedComponentCFG(SceneComponentCFG):
    """Base configuration for asset-based components.

    Components that inherit from this configuration can create assets
    from the asset registry using Scene DSL fields, positions, orientations,
    and scales.
    """

    initial_position: list[float] = field(default_factory=lambda: [0, 0, 0])
    initial_orientation: list[float] = field(default_factory=lambda: [0, 0, 0, 1])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    asset_module: str = ""


class AssetBasedComponent(SceneComponent):
    """Base component for asset-based scene elements.

    Provides common asset creation logic for components that need to
    instantiate assets from the asset registry. All asset creation
    goes through the generate() interface.
    """

    def __init__(self, cfg: AssetBasedComponentCFG, asset_registry: AssetRegistry = None):
        super().__init__(cfg)
        self.cfg = cfg
        self._asset_registry = asset_registry

    def set_asset_registry(self, registry: AssetRegistry):
        """Set the asset registry for asset creation."""
        self._asset_registry = registry

    def generate(self, cfg: AssetBasedComponentCFG | None = None) -> object | None:
        """Generate asset based on Scene DSL fields from configuration."""
        if cfg is not None:
            self.cfg = cfg

        if self._asset_registry is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires an AssetRegistry injected from DataCenter; got None."
            )

        if self.cfg.generator:
            return self._generate_by_generator()
        if self.cfg.asset:
            self._preload_asset_module()
            return self._generate_by_asset()
        if self.cfg.asset_tag:
            self._preload_asset_module()
            return self._generate_by_asset_tag()
        manaprint(f"WARNING: No asset, asset_tag, or generator specified for {self.__class__.__name__}")
        return None

    def _preload_asset_module(self) -> None:
        """Import an explicitly declared asset module before registry lookup."""

        module_path = str(getattr(self.cfg, "asset_module", "") or "").strip()
        if not module_path:
            return
        try:
            module = importlib.import_module(module_path)
            register_assets = getattr(module, "register_assets", None)
            if callable(register_assets):
                register_assets()
        except Exception as exc:
            raise RuntimeError(
                f"{self.__class__.__name__} failed to preload asset_module={module_path!r}: {exc}"
            ) from exc

    def _generate_by_asset(self) -> object | None:
        """Generate asset by name lookup from registry."""
        if not self.cfg.asset:
            manaprint(f"WARNING: No asset specified for {self.__class__.__name__}")
            return None

        try:
            asset_class = self._asset_registry.get_asset_by_name(self.cfg.asset)
        except Exception as exc:
            raise RuntimeError(f"Failed to get asset by name '{self.cfg.asset}': {exc}") from exc
        return self._instantiate_asset(asset_class)

    def _generate_by_asset_tag(self) -> object | None:
        """Generate asset by tag lookup from registry.

        Randomly selects one matching asset from the pool so that
        repeated calls with the same tag can produce different results
        (essential for ``nums > 1`` scene variant diversity).
        """
        if not self.cfg.asset_tag:
            manaprint(f"WARNING: No asset_tag specified for {self.__class__.__name__}")
            return None

        try:
            assets = self._asset_registry.get_assets_by_tag(self.cfg.asset_tag)
        except Exception as exc:
            raise RuntimeError(f"Failed to get asset by tag '{self.cfg.asset_tag}': {exc}") from exc
        if not assets:
            manaprint(f"WARNING: No assets found with tag '{self.cfg.asset_tag}'")
            return None
        # Use OS-backed choice so tag picks are not coupled to global ``random`` state
        # (often seeded by sim/training code), which otherwise yields the same asset every run.
        asset_class = secrets.choice(assets)
        return self._instantiate_asset(asset_class)

    def _generate_by_generator(self) -> object | None:
        """Generate asset procedurally.

        Subclasses can override this to implement ``generator`` while
        still using :meth:`generate` as the single public entry point.
        """
        manaprint(f"WARNING: {self.__class__.__name__} does not implement generator={self.cfg.generator!r}")
        return None

    def _resolve_position(self) -> list[float]:
        """Return the position to apply. Subclasses may override."""
        return list(self.cfg.initial_position)

    def _resolve_orientation(self) -> list[float]:
        """Return the orientation to apply. Subclasses may override."""
        return list(self.cfg.initial_orientation)

    def _instantiate_asset(self, asset_class) -> object:
        """Instantiate asset class and apply transforms."""
        from isaaclab_arena.utils.pose import Pose

        asset_name = str(
            getattr(asset_class, "name", None)
            or self.cfg.asset
            or getattr(asset_class, "__name__", type(asset_class).__name__)
        )
        usd_path = str(getattr(asset_class, "usd_path", "") or "")
        try:
            asset_instance = asset_class()
        except Exception as exc:
            source = f" from {usd_path!r}" if usd_path else ""
            raise RuntimeError(f"Failed to instantiate asset '{asset_name}'{source}: {exc}") from exc
        try:
            self._apply_authored_name(asset_instance)

            position = self._resolve_position()
            orientation = self._resolve_orientation()
            asset_instance.set_initial_pose(Pose(position_xyz=tuple(position), rotation_xyzw=tuple(orientation)))

            # YAML scale overrides registry class scale when not default (1,1,1).
            cfg_scale = _coerce_scale_triple(self.cfg.scale)
            if cfg_scale != (1.0, 1.0, 1.0) and hasattr(asset_instance, "scale"):
                asset_instance.scale = cfg_scale

            # ``set_initial_pose`` / ``__init__`` may have built object_cfg with a stale spawn scale.
            if hasattr(asset_instance, "_init_object_cfg"):
                asset_instance.object_cfg = asset_instance._init_object_cfg()

            # Allow subclasses to customize
            self._customize_asset(asset_instance)

            return asset_instance

        except Exception as exc:
            raise RuntimeError(f"Failed to configure asset '{asset_name}': {exc}") from exc

    def _customize_asset(self, asset: object) -> None:
        """Customize asset after creation. Override in subclasses."""
        pass

    def _apply_authored_name(self, asset: object) -> None:
        """Apply stable YAML-authored object names for layout matching."""

        authored_name = str(getattr(self.cfg, "id", "") or "").strip()
        if not authored_name or not hasattr(asset, "name"):
            return
        asset.__dict__["name"] = authored_name
        if hasattr(asset, "_name"):
            asset._name = authored_name
        if hasattr(asset, "prim_path"):
            # Anchor convention: spawn under the "_"-prefixed anchor so contact
            # sensor/filter paths (asset_anchor_path consumers) match reality.
            try:
                from isaaclab_arena.assets.object_base import asset_anchor_path
            except ImportError as exc:
                detail = f"{getattr(exc, 'name', '')} {exc}"
                if not any(marker in detail for marker in ("isaaclab", "isaaclab_physx", "omni", "pxr")):
                    raise
                asset.prim_path = f"{{ENV_REGEX_NS}}/_{authored_name}"
            else:
                asset.prim_path = asset_anchor_path(authored_name)
