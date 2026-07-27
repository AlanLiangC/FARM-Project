"""Launch the Odin1 (fisheye + LiDAR) online mapping chain.

Odin1 has no depth image, so ``odin1_depth_pub`` synthesises one from the LiDAR
cloud and republishes standard RGBD topics. From there the pipeline is identical
to the multi-camera Spot path: one ``frame_pub`` (camera ``odin1``) feeds the
shared ``streaming_mapper``. This is the single-camera instance of the same
"one interface, any number of cameras" contract as ``mapping_five_cam``.

The odometry topic is remapped into ``frame_pub``'s pose listener
(``/spot/platform/odom``) so no code differs between Spot and Odin1.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from mapping.lib.paths import (
    default_scene_graph_json_save_path,
    default_scene_graph_snapshot_dir,
    default_scene_state_path,
    default_storage_image_dir,
)


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("logger_level", default_value="INFO"),
        DeclareLaunchArgument("caption_enabled", default_value="false"),
        DeclareLaunchArgument("calibration_path", default_value="/odin1_data/calib.yaml"),
        DeclareLaunchArgument("image_topic", default_value="/odin1/image/compressed"),
        DeclareLaunchArgument("cloud_topic", default_value="/odin1/cloud_slam"),
        DeclareLaunchArgument("odom_topic", default_value="/odin1/odometry_highfreq"),
        DeclareLaunchArgument("output_pinhole", default_value="false"),
        DeclareLaunchArgument("depth_dilate_px", default_value="2"),
        DeclareLaunchArgument("scan_window_s", default_value="0.15"),
        DeclareLaunchArgument("max_scans_per_frame", default_value="3"),
        DeclareLaunchArgument("depth_max_m", default_value="60.0"),
        DeclareLaunchArgument("base_frame", default_value="odin1_base_link"),
        DeclareLaunchArgument("optical_frame", default_value="odin1_optical"),
        # frame_pub needs an odom frame as its TF target; the Odin bag's odometry
        # header frame is typically "odom".
        DeclareLaunchArgument("global_frame", default_value="odom"),
        DeclareLaunchArgument("latest_only", default_value="true"),
        DeclareLaunchArgument("scene_state_save_path", default_value=default_scene_state_path()),
        DeclareLaunchArgument("scene_state_load_path", default_value=default_scene_state_path()),
        DeclareLaunchArgument("scene_state_save_on_shutdown", default_value="true"),
        DeclareLaunchArgument("segmenter_device", default_value=""),
    ]

    depth_pub = Node(
        package="mapping",
        executable="odin1_depth_pub",
        name="odin1_depth_pub",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "calibration_path": LaunchConfiguration("calibration_path"),
            "image_topic": LaunchConfiguration("image_topic"),
            "cloud_topic": LaunchConfiguration("cloud_topic"),
            "odom_topic": LaunchConfiguration("odom_topic"),
            "output_pinhole": LaunchConfiguration("output_pinhole"),
            "depth_dilate_px": LaunchConfiguration("depth_dilate_px"),
            "scan_window_s": LaunchConfiguration("scan_window_s"),
            "max_scans_per_frame": LaunchConfiguration("max_scans_per_frame"),
            "depth_max_m": LaunchConfiguration("depth_max_m"),
            "base_frame": LaunchConfiguration("base_frame"),
            "optical_frame": LaunchConfiguration("optical_frame"),
        }],
    )

    frame_pub = Node(
        package="mapping",
        executable="frame_pub",
        name="odin1_frame_pub",
        output="screen",
        parameters=[{
            "camera": "odin1",
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "global_frame": LaunchConfiguration("global_frame"),
            "tf_buffer_cache_time_sec": 300.0,
            "tf_lookup_timeout_sec": 1.0,
            "sync_slop_sec": 0.05,
            "max_rgb_depth_skew_sec": 0.05,
            "max_tf_skew_sec": 0.05,
        }],
        # Point the pose listener's odom subscription at the Odin odometry topic.
        remappings=[("/spot/platform/odom", LaunchConfiguration("odom_topic"))],
    )

    mapper = Node(
        package="mapping",
        executable="streaming_mapper",
        name="mapping_streaming_mapper",
        output="screen",
        parameters=[{
            "logger_level": LaunchConfiguration("logger_level"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "camera_names": ["odin1"],
            "expected_batch": 1,
            "batch_timeout": 1.0,
            "latest_only": LaunchConfiguration("latest_only"),
            "caption_enabled": LaunchConfiguration("caption_enabled"),
            "segmenter_device": LaunchConfiguration("segmenter_device"),
            "scene_state_load_path": LaunchConfiguration("scene_state_load_path"),
            "scene_state_save_path": LaunchConfiguration("scene_state_save_path"),
            "scene_state_save_on_shutdown": LaunchConfiguration("scene_state_save_on_shutdown"),
            "scene_state_save_observations": True,
            "scene_graph_json_save_enabled": False,
            "scene_graph_snapshot_save_enabled": False,
            "scene_graph_json_save_path": default_scene_graph_json_save_path(),
            "scene_graph_snapshot_dir": default_scene_graph_snapshot_dir(),
            "storage_image_dir": default_storage_image_dir(),
        }],
    )

    return LaunchDescription([*args, depth_pub, frame_pub, mapper])
