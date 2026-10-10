"""RTX 3D LiDAR helper (Livox Mid-360 profile).

Creates an RTX lidar sensor via ``IsaacSensorCreateRtxLidar`` and attaches
a ``IsaacCreateRTXLidarScanBuffer`` annotator for per-frame point cloud
retrieval.
"""

from __future__ import annotations

import numpy as np
from pxr import Sdf

MID360_SENSOR_ATTRIBUTES = {
    "omni:sensor:Core:scanType": "ROTARY",
    "omni:sensor:Core:intensityProcessing": "NORMALIZATION",
    "omni:sensor:Core:rotationDirection": "CW",
    "omni:sensor:Core:rayType": "IDEALIZED",
    "omni:sensor:Core:nearRangeM": 0.1,
    "omni:sensor:Core:farRangeM": 40.0,
    "omni:sensor:Core:rangeResolutionM": 0.004,
    "omni:sensor:Core:rangeAccuracyM": 0.025,
    "omni:sensor:Core:avgPowerW": 0.002,
    "omni:sensor:Core:minReflectance": 0.1,
    "omni:sensor:Core:minReflectanceRange": 70.0,
    "omni:sensor:Core:wavelengthNm": 905.0,
    "omni:sensor:Core:pulseTimeNs": 6,
    "omni:sensor:Core:azimuthErrorMean": 0.1,
    "omni:sensor:Core:azimuthErrorStd": 0.5,
    "omni:sensor:Core:elevationErrorMean": 0.1,
    "omni:sensor:Core:elevationErrorStd": 0.5,
    "omni:sensor:Core:maxReturns": 2,
    "omni:sensor:Core:scanRateBaseHz": 10.0,
    "omni:sensor:Core:reportRateBaseHz": 5000,
    "omni:sensor:Core:numberOfEmitters": 40,
    "omni:sensor:Core:numberOfChannels": 40,
    "omni:sensor:Core:rangeOffset": 0.03,
    "omni:sensor:Core:intensityMappingType": "LINEAR",
    "omni:sensor:Core:emitterState:s001:azimuthDeg": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:elevationDeg": [
        -7.0,
        -5.525,
        -4.050,
        -2.575,
        -1.1004,
        0.374,
        1.849,
        3.324,
        4.799,
        6.274,
        7.7494,
        9.2249,
        10.699,
        12.174,
        13.645,
        15.1243,
        16.5999,
        18.074,
        19.5499,
        21.024,
        22.493,
        23.9749,
        25.44,
        26.924,
        28.39,
        29.8743,
        31.3499,
        32.824,
        34.29,
        35.774,
        37.2486,
        38.724,
        40.19,
        41.674,
        43.14,
        44.624,
        46.09,
        47.574,
        49.048,
        50.524,
    ],
    "omni:sensor:Core:emitterState:s001:fireTimeNs": [i * 1000 for i in range(40)],
    "omni:sensor:Core:emitterState:s001:distanceCorrectionM": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:focalDistM": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:focalSlope": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:horOffsetM": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:reportRateDiv": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:vertOffsetM": [0.0] * 40,
    "omni:sensor:Core:emitterState:s001:channelId": list(range(1, 41)),
}

_SDF_TYPE_MAP = {
    bool: Sdf.ValueTypeNames.Bool,
    int: Sdf.ValueTypeNames.Int,
    float: Sdf.ValueTypeNames.Double,
    str: Sdf.ValueTypeNames.Token,
}


def _infer_sdf_type(value):
    if type(value) in _SDF_TYPE_MAP:
        return _SDF_TYPE_MAP[type(value)]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return Sdf.ValueTypeNames.FloatArray
        elem = value[0]
        if isinstance(elem, int):
            return Sdf.ValueTypeNames.IntArray
        if isinstance(elem, float):
            return Sdf.ValueTypeNames.DoubleArray
        return Sdf.ValueTypeNames.TokenArray
    return Sdf.ValueTypeNames.Token


class Rtx3dLidarHelper:
    """RTX 3D lidar that creates a sensor prim and reads point clouds."""

    def __init__(
        self,
        parent_prim_path: str,
        env_id: int = 0,
        sensor_name: str = "Mid360Rtx",
        frame_id: str = "ld_laser",
        translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self._parent_prim_path = parent_prim_path
        self._env_id = env_id
        self._sensor_name = sensor_name
        self._frame_id = frame_id
        self._translation_offset = translation_offset
        self._lidar_prim_path: str | None = None
        self._point_cloud_annotator = None
        self._initialized = False

    def initialize(self, scene) -> bool:
        import omni.kit.app
        import omni.kit.commands
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf

        parent_path = self._parent_prim_path.replace(
            "{ENV_REGEX_NS}",
            f"{scene.env_ns}/env_{self._env_id}",
        )
        self._lidar_prim_path = f"{parent_path}/{self._sensor_name}"

        _, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=self._sensor_name,
            parent=parent_path,
            config="Example_Rotary",
            translation=tuple(self._translation_offset),
            orientation=Gf.Quatd(1.0, 0, 0, 0.0),
        )
        if sensor is None:
            print(f"[Mid360Rtx] Failed to create RTX lidar under {parent_path}")
            return False

        if hasattr(sensor, "GetPath"):
            actual_path = str(sensor.GetPath())
            if actual_path != self._lidar_prim_path:
                print(f"[Mid360Rtx] Sensor created at {actual_path} (expected {self._lidar_prim_path})")
                self._lidar_prim_path = actual_path
            lidar_prim = sensor
        else:
            stage = omni.usd.get_context().get_stage()
            lidar_prim = stage.GetPrimAtPath(self._lidar_prim_path)

        if not lidar_prim.IsValid():
            print(f"[Mid360Rtx] Prim not valid at {self._lidar_prim_path}")
            return False

        print(f"[Mid360Rtx] Created RTX lidar at {self._lidar_prim_path}")
        self._apply_sensor_attributes(lidar_prim)

        hydra_texture = rep.create.render_product(
            lidar_prim.GetPath(),
            [1, 1],
            render_vars=["GenericModelOutput", "RtxSensorMetadata"],
        )
        self._point_cloud_annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        self._point_cloud_annotator.initialize(transformPoints=True, outputObjectId=False)
        self._point_cloud_annotator.attach([hydra_texture.path])

        self._warm_up()
        self._initialized = True
        return True

    @staticmethod
    def _apply_sensor_attributes(lidar_prim):
        applied = 0
        for attr_name, attr_value in MID360_SENSOR_ATTRIBUTES.items():
            attr = lidar_prim.GetAttribute(attr_name)
            if attr and attr.IsValid():
                attr.Set(attr_value)
            else:
                sdf_type = _infer_sdf_type(attr_value)
                lidar_prim.CreateAttribute(attr_name, sdf_type).Set(attr_value)
            applied += 1
        print(f"[Mid360Rtx] Applied {applied} sensor attributes")

    def _warm_up(self, frames: int = 180):
        import omni.kit.app
        import omni.kit.commands
        import omni.usd

        app = omni.kit.app.get_app()
        for _ in range(frames):
            app.update()
        print(f"[Mid360Rtx] Warm-up: rendered {frames} frames")

    @property
    def frame_id(self) -> str:
        return self._frame_id

    def get_point_cloud_data(self) -> dict | None:
        if not self._initialized or self._point_cloud_annotator is None:
            return None
        data = self._point_cloud_annotator.get_data()
        if data is None:
            return None
        buf = data.get("data")
        if buf is None or len(buf) == 0:
            return None

        points = np.array(buf, dtype=np.float32).reshape(-1, 3)
        intensities_raw = data.get("intensity")
        if intensities_raw is not None and len(intensities_raw) > 0:
            intensities = np.array(intensities_raw, dtype=np.float32)
        else:
            intensities = np.ones(len(points), dtype=np.float32)
        return {"points": points, "intensities": intensities}


def create_mid360_rtx_lidar(
    parent_prim_path: str,
    env_id: int = 0,
    sensor_name: str = "Mid360Rtx",
    frame_id: str = "ld_laser",
    translation_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Rtx3dLidarHelper:
    return Rtx3dLidarHelper(
        parent_prim_path=parent_prim_path,
        env_id=env_id,
        sensor_name=sensor_name,
        frame_id=frame_id,
        translation_offset=translation_offset,
    )
