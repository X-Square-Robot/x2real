from isaaclab.sensors.imu import ImuCfg

IMU_SAMPLE_RATE_HZ = 60

EX001_IMU_CFG = ImuCfg(
    update_period=1.0 / IMU_SAMPLE_RATE_HZ,
)
