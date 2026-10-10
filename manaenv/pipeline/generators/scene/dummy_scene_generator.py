"""Dummy scene generator for testing purposes."""

from typing import TYPE_CHECKING, Any, Optional

from ..scene_generator import SceneGenerator, SceneGeneratorCFG

if TYPE_CHECKING:
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene


class DummySceneGenerator(SceneGenerator):
    """Dummy scene generator with minimal setup for testing."""

    def __init__(self, config: SceneGeneratorCFG, asset_registry: Optional["AssetRegistry"] = None):
        super().__init__(config)
        self.asset_registry = asset_registry
        self.scene = None

    def build_scene(self) -> "Scene":
        """Build an empty scene."""
        from isaaclab_arena.assets.registries import AssetRegistry
        from isaaclab_arena.scene.scene import Scene

        if self.asset_registry is None:
            self.asset_registry = AssetRegistry()

        # Create empty scene with no assets
        self.scene = Scene(assets=[])
        return self.scene

    def generate(self, **kwargs) -> dict[str, Any]:
        """Generate scene and return metadata."""
        scene = self.build_scene()
        return {"scene": scene, "metadata": self.config.metadata}

    def get_scene(self) -> "Scene":
        """Get the generated scene."""
        if self.scene is None:
            self.build_scene()
        return self.scene
