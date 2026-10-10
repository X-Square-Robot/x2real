"""Constants for camera sidecar export (semantic preview MP4, depth MKV).

Semantic sidecars use deterministic pseudocolor from class id (preview only;
not the Isaac viewport LUT).

Depth sidecars store uint16 millimeters; Isaac Lab depth tensors are meters
(float32), comparable in spirit to ``manaenv.ros_utils.constants.DEPTH_MAX_MM``.
"""

import numpy as np

# Semantic sidecar (MP4 pseudocolor from class id)
SIDECAR_SEMANTIC_BACKGROUND_CLASS_ID = 0
SIDECAR_SEMANTIC_BACKGROUND_RGB = (32, 32, 32)
SIDECAR_SEMANTIC_HASH_RGB_MUL = (37, 17, 97)
SIDECAR_SEMANTIC_HASH_RGB_ADD = (40, 80, 120)
SIDECAR_SEMANTIC_UINT8_MOD = 256

# Depth sidecar (Isaac meters -> uint16 millimeters for lossless MKV)
SIDECAR_DEPTH_METERS_TO_MM = 1000.0
SIDECAR_DEPTH_UINT16_MAX_MM = float(np.iinfo(np.uint16).max)
