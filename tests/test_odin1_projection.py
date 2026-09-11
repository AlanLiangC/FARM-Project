from pathlib import Path
import sys

import numpy as np


MAPPING_ROOT = Path(__file__).resolve().parents[1] / "ros" / "mapping"
sys.path.insert(0, str(MAPPING_ROOT))

from mapping.lib.odin1_projection import (  # noqa: E402
    OdinCalibration,
    T_IMU_LIDAR,
    pointcloud2_to_xyz,
    project_world_pinhole,
    rasterize_zbuffer,
)


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


def test_pinhole_projection_applies_camera_pose_without_homogeneous_copy():
    points = np.array([[1.0, 2.0, 4.0], [-1.0, 0.0, 2.0]], dtype=np.float32)
    T_world_camera = np.eye(4)
    T_world_camera[:3, 3] = [0.0, 0.0, 1.0]
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])

    pixels, depth = project_world_pinhole(points, T_world_camera, K)

    np.testing.assert_allclose(depth, [3.0, 1.0])
    np.testing.assert_allclose(pixels, [[83.333333, 106.666667], [-50.0, 40.0]], rtol=1e-6)


def test_zbuffer_retains_nearest_duplicate_pixel():
    pixels = np.array([[2.1, 1.0], [2.4, 1.2], [0.0, 0.0]])
    depths = np.array([3.0, 1.5, 7.0])

    depth = rasterize_zbuffer(pixels, depths, 4, 3, min_depth=0.1, max_depth=5.0)

    assert depth[1, 2] == np.float32(1.5)
    assert np.isnan(depth[0, 0])  # 7 m was rejected by max_depth
    assert np.isnan(depth[2, 3])


def test_pointcloud_decode_supports_padded_records_without_field_copies():
    class Field:
        def __init__(self, name, offset):
            self.name = name
            self.offset = offset
            self.datatype = 7  # sensor_msgs/PointField.FLOAT32

    class Cloud:
        width = 3
        height = 1
        point_step = 20
        is_bigendian = False
        fields = [Field("x", 0), Field("y", 4), Field("z", 8), Field("intensity", 16)]

    packed = np.zeros(3, dtype={
        "names": ["x", "y", "z", "padding", "intensity"],
        "formats": ["<f4", "<f4", "<f4", "<u4", "<f4"],
        "offsets": [0, 4, 8, 12, 16],
        "itemsize": 20,
    })
    packed["x"] = [1.0, np.nan, 7.0]
    packed["y"] = [2.0, 5.0, 8.0]
    packed["z"] = [3.0, 6.0, 9.0]
    cloud = Cloud()
    cloud.data = bytearray(packed.tobytes())

    points = pointcloud2_to_xyz(cloud)

    np.testing.assert_array_equal(points, [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])
