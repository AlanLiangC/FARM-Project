from pathlib import Path
import sys

import numpy as np


MAPPING_ROOT = Path(__file__).resolve().parents[1] / "ros" / "mapping"
sys.path.insert(0, str(MAPPING_ROOT))

from mapping.lib.odin1_projection import OdinCalibration, T_IMU_LIDAR  # noqa: E402


def test_imu_camera_transform_includes_factory_lidar_offset():
    calibration = OdinCalibration(
        image_width=10,
        image_height=10,
        A11=5.0,
        A12=0.0,
        A22=5.0,
        u0=5.0,
        v0=5.0,
        k=[0.0] * 6,
        T_camera_base=np.eye(4),
    )
    np.testing.assert_allclose(calibration.T_imu_camera, T_IMU_LIDAR)


def test_imu_camera_composition_matches_camera_from_lidar_inverse():
    T_camera_lidar = np.eye(4)
    T_camera_lidar[:3, 3] = [0.1, -0.2, 0.3]
    calibration = OdinCalibration(
        image_width=10,
        image_height=10,
        A11=5.0,
        A12=0.0,
        A22=5.0,
        u0=5.0,
        v0=5.0,
        k=[0.0] * 6,
        T_camera_base=T_camera_lidar,
    )
    np.testing.assert_allclose(
        calibration.T_imu_camera,
        T_IMU_LIDAR @ np.linalg.inv(T_camera_lidar),
    )
