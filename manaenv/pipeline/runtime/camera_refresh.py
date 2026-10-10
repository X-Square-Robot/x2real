"""Camera refresh helpers for env-runner reset boundaries."""

from __future__ import annotations

from typing import Any

from manaenv.utils.camera_utils import refresh_camera_bundle_after_reset


def refresh_cameras_after_reset(gym_env: Any, env_ids: Any = None, camera_bundle: Any = None) -> int:
    """Refresh configured cameras after an env reset."""
    return refresh_camera_bundle_after_reset(gym_env, camera_bundle=camera_bundle, env_ids=env_ids)


def compute_post_reset_obs(gym_env: Any, env_ids: Any = None, camera_bundle: Any = None) -> Any:
    """Refresh cameras, capture home EE refs, then compute observations."""
    from manaenv.assets.robots.components.embodiment_obs import refresh_scene_and_store_home_ee_refs

    refresh_cameras_after_reset(gym_env, env_ids=env_ids, camera_bundle=camera_bundle)
    env = getattr(gym_env, "unwrapped", gym_env)
    refresh_scene_and_store_home_ee_refs(env, env_ids=env_ids)
    return gym_env.observation_manager.compute()
