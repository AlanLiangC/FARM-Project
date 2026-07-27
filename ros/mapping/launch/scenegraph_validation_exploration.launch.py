"""Launch scene-graph validation workflow without visual_search_goal_publisher."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from mapping.lib.paths import (
    default_adaptation_results_dir,
    default_scene_graph_json_save_path,
    default_scene_graph_snapshot_dir,
    default_scene_state_path,
    default_storage_image_dir,
)


def generate_launch_description() -> LaunchDescription:
    scene_state_path_arg = DeclareLaunchArgument(
        "scene_state_path",
        default_value=default_scene_state_path(),
        description="Scene state file path used for both load and save.",
    )
    scene_graph_json_save_path_arg = DeclareLaunchArgument(
        "scene_graph_json_save_path",
        default_value=default_scene_graph_json_save_path(),
        description="Path for scene graph JSON output.",
    )
    scene_graph_snapshot_dir_arg = DeclareLaunchArgument(
        "scene_graph_snapshot_dir",
        default_value=default_scene_graph_snapshot_dir(),
        description="Directory for scene graph snapshot versions (vXXXXXX).",
    )
    storage_image_dir_arg = DeclareLaunchArgument(
        "storage_image_dir",
        default_value=default_storage_image_dir(),
        description="Directory for saved per-frame image_store outputs.",
    )
    caption_enabled_arg = DeclareLaunchArgument(
        "caption_enabled",
        default_value="true",
        description="Enable caption worker for online captions.",
    )
    logger_level_arg = DeclareLaunchArgument(
        "logger_level",
        default_value="DEBUG",
        description="ROS logger severity for the streaming mapper.",
    )
    debug_queue_status_arg = DeclareLaunchArgument(
        "debug_queue_status",
        default_value="true",
        description="Enable per-camera queue depth logging in streaming mapper.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Set to true when /clock is available.",
    )
    latest_only_arg = DeclareLaunchArgument(
        "latest_only",
        default_value="true",
        description="Keep only the newest pending frame per camera.",
    )
    max_rgb_depth_skew_arg = DeclareLaunchArgument(
        "max_rgb_depth_skew_sec",
        default_value="0.05",
        description="Drop RGB/depth pairs when |delta_t| exceeds this threshold.",
    )
    max_pair_skew_arg = DeclareLaunchArgument(
        "max_pair_skew_sec",
        default_value="0.05",
        description="Drop frames when RGB/depth/meta skew exceeds this threshold.",
    )
    global_frame_arg = DeclareLaunchArgument(
        "global_frame",
        default_value="spot/odom",
        description="TF target frame for camera poses.",
    )
    tf_lookup_timeout_arg = DeclareLaunchArgument(
        "tf_lookup_timeout_sec",
        default_value="1.0",
        description="TF2 lookup timeout in seconds.",
    )
    tf_buffer_cache_arg = DeclareLaunchArgument(
        "tf_buffer_cache_time_sec",
        default_value="600.0",
        description="TF2 cache duration in seconds.",
    )
    scene_state_save_observations_arg = DeclareLaunchArgument(
        "scene_state_save_observations",
        default_value="true",
        description="Include per-object RGB observations in saved scene state.",
    )
    scene_state_view_limit_arg = DeclareLaunchArgument(
        "scene_state_save_observation_view_limit",
        default_value="1",
        description="Max number of observation views saved per object.",
    )
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="adaptation",
        description="Visual search mode for YOLOE text prompt node.",
    )
    debug_save_enable_arg = DeclareLaunchArgument(
        "debug_save_enable",
        default_value="true",
        description="Save visual search debug images.",
    )
    debug_save_dir_arg = DeclareLaunchArgument(
        "debug_save_dir",
        default_value="log/visual_search_results_yoloe",
        description="Directory for visual search debug outputs.",
    )
    adaptation_results_dir_arg = DeclareLaunchArgument(
        "adaptation_results_dir",
        default_value=default_adaptation_results_dir(),
        description="Directory for adaptation snapshots and manifest outputs.",
    )

    mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("mapping"), "launch", "mapping_five_cam.launch.py")
        ),
        launch_arguments={
            "caption_enabled": LaunchConfiguration("caption_enabled"),
            "logger_level": LaunchConfiguration("logger_level"),
            "debug_queue_status": LaunchConfiguration("debug_queue_status"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "latest_only": LaunchConfiguration("latest_only"),
            "max_rgb_depth_skew_sec": LaunchConfiguration("max_rgb_depth_skew_sec"),
            "max_pair_skew_sec": LaunchConfiguration("max_pair_skew_sec"),
            "global_frame": LaunchConfiguration("global_frame"),
            "tf_lookup_timeout_sec": LaunchConfiguration("tf_lookup_timeout_sec"),
            "tf_buffer_cache_time_sec": LaunchConfiguration("tf_buffer_cache_time_sec"),
            "scene_state_load_path": LaunchConfiguration("scene_state_path"),
            "scene_state_save_path": LaunchConfiguration("scene_state_path"),
            "scene_state_save_observations": LaunchConfiguration("scene_state_save_observations"),
            "scene_state_save_observation_view_limit": LaunchConfiguration("scene_state_save_observation_view_limit"),
            "scene_graph_json_save_path": LaunchConfiguration("scene_graph_json_save_path"),
            "scene_graph_snapshot_dir": LaunchConfiguration("scene_graph_snapshot_dir"),
            "storage_image_dir": LaunchConfiguration("storage_image_dir"),
        }.items(),
    )

    visual_search_yoloe_text_prompt = Node(
        package="mapping",
        executable="visual_search_yoloe_text_prompt",
        name="visual_search_yoloe_text_prompt",
        output="screen",
        parameters=[{
            "mode": LaunchConfiguration("mode"),
            "debug_save_enable": LaunchConfiguration("debug_save_enable"),
            "debug_save_dir": LaunchConfiguration("debug_save_dir"),
            "adaptation_results_dir": LaunchConfiguration("adaptation_results_dir"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return LaunchDescription([
        scene_state_path_arg,
        scene_graph_json_save_path_arg,
        scene_graph_snapshot_dir_arg,
        storage_image_dir_arg,
        caption_enabled_arg,
        logger_level_arg,
        debug_queue_status_arg,
        use_sim_time_arg,
        latest_only_arg,
        max_rgb_depth_skew_arg,
        max_pair_skew_arg,
        global_frame_arg,
        tf_lookup_timeout_arg,
        tf_buffer_cache_arg,
        scene_state_save_observations_arg,
        scene_state_view_limit_arg,
        mode_arg,
        debug_save_enable_arg,
        debug_save_dir_arg,
        adaptation_results_dir_arg,
        mapping_launch,
        visual_search_yoloe_text_prompt,
    ])
