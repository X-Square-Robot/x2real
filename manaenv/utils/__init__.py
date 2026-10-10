"""Utils module for manaenv.

Provides:
    - camera_constants        — tunables for semantic/depth sidecar export
    - camera_utils            — USD camera prim helpers (IsaacSim)
    - config_io               — shared YAML/JSON config loading helpers
    - debug_print             — coloured debug logging (manaprint)
    - gpu_capacity            — automatic num_envs estimation & VRAM profiling
    - math_utils              — quaternion / Euler / pose helpers (numpy, default XYZW)
    - motion_utils            — waypoint interpolation & trajectory builders (torch)
    - priority                — generic priority-based sorting
    - tensor_utils            — tensor ↔ numpy conversion & image encoding
"""
