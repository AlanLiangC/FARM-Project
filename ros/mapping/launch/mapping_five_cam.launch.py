"""Launch file for five-camera batching feeding the mapping pipeline (mapping package)."""

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
    config_arg = DeclareLaunchArgument(
        "config_path",
        default_value="",
        description="Optional path to mapping YAML config (used for storage overrides).",
    )
    logger_level_arg = DeclareLaunchArgument(
        "logger_level",
        default_value="INFO",
        description="ROS logger severity for the streaming mapper.",
    )
    debug_queue_arg = DeclareLaunchArgument(
        "debug_queue_status",
        default_value="false",
        description="Enable per-camera queue depth logging.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Set to true when playing back bags with /clock.",
    )
    caption_enabled_arg = DeclareLaunchArgument(
        "caption_enabled",
        default_value="false",
        description="Enable caption worker for online captions.",
    )
    segmenter_device_arg = DeclareLaunchArgument(
        "segmenter_device",
        default_value="",
        description="Device string for YOLOE segmenter (e.g., cuda:0). Empty means auto.",
    )
    segmenter_mask_erosion_arg = DeclareLaunchArgument(
        "segmenter_mask_erosion_px",
        default_value="3",
        description="Erode segmentation masks by N pixels before 3D stats (0 disables).",
    )
    segmenter_mahalanobis_arg = DeclareLaunchArgument(
        "segmenter_mahalanobis_thresh",
        default_value="2.0",
        description="Mahalanobis distance threshold for outlier rejection (<=0 disables).",
    )
    global_frame_arg = DeclareLaunchArgument(
        "global_frame",
        default_value="spot/odom",
        description="TF target frame for camera poses in frame_pub (e.g., spot/odom or map).",
    )
    tf_buffer_cache_arg = DeclareLaunchArgument(
        "tf_buffer_cache_time_sec",
        default_value="300.0",
        description="TF2 buffer cache duration (seconds) for each frame_pub.",
    )
    tf_lookup_timeout_arg = DeclareLaunchArgument(
        "tf_lookup_timeout_sec",
        default_value="1.0",
        description="TF2 lookup timeout (seconds) for each frame_pub.",
    )
    sync_slop_arg = DeclareLaunchArgument(
        "sync_slop_sec",
        default_value="0.05",
        description="Approximate sync slop (seconds) for pairing RGB/depth in frame publishers.",
    )
    max_rgb_depth_skew_arg = DeclareLaunchArgument(
        "max_rgb_depth_skew_sec",
        default_value="0.05",
        description="Drop RGB/depth pairs when |Δt| exceeds this (seconds).",
    )
    max_tf_skew_arg = DeclareLaunchArgument(
        "max_tf_skew_sec",
        default_value="0.05",
        description="Drop frames when |tf_stamp - rgb_stamp| exceeds this (seconds).",
    )
    max_pair_skew_arg = DeclareLaunchArgument(
        "max_pair_skew_sec",
        default_value="0.05",
        description="Drop frames in streaming_mapper when RGB/depth/meta skew exceeds this (seconds).",
    )
    latest_only_arg = DeclareLaunchArgument(
        "latest_only",
        default_value="true",
        description="When true, keep only the newest not-yet-processed frame per camera in streaming_mapper.",
    )
    scene_state_load_arg = DeclareLaunchArgument(
        "scene_state_load_path",
        default_value=default_scene_state_path(),
        description="Optional path to load a saved scene state at startup.",
    )
    scene_state_save_arg = DeclareLaunchArgument(
        "scene_state_save_path",
        default_value=default_scene_state_path(),
        description="Optional path to save scene state on shutdown (defaults to load path when unset).",
    )
    scene_state_observations_arg = DeclareLaunchArgument(
        "scene_state_save_observations",
        default_value="true",
        description="When true, include per-object RGB observations in the saved scene state.",
    )
    scene_state_view_limit_arg = DeclareLaunchArgument(
        "scene_state_save_observation_view_limit",
        default_value="1",
        description="Max number of per-object observation views to save (-1 for unlimited).",
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
    scene_graph_publish_interval_arg = DeclareLaunchArgument(
        "scene_graph_publish_interval",
        default_value="60.0",
        description="Interval in seconds for publishing scene graph JSON.",
    )
    scene_graph_json_save_enabled_arg = DeclareLaunchArgument(
        "scene_graph_json_save_enabled",
        default_value="true",
        description="Enable scene graph JSON persistence.",
    )
    scene_graph_snapshot_save_enabled_arg = DeclareLaunchArgument(
        "scene_graph_snapshot_save_enabled",
        default_value="true",
        description="Enable snapshot persistence to disk.",
    )
    scene_graph_snapshot_publish_interval_arg = DeclareLaunchArgument(
        "scene_graph_snapshot_publish_interval",
        default_value="0.0",
        description="Interval in seconds for publishing scene graph snapshots.",
    )

    camera_names = ["head_left", "head_right", "left", "right", "rear"]
    frame_publishers = []
    for camera in camera_names:
        frame_publishers.append(
            Node(
                package="mapping",
                executable="frame_pub",
                name=f"{camera}_frame_pub",
                output="screen",
                parameters=[{
                    "camera": camera,
                    "global_frame": LaunchConfiguration("global_frame"),
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "tf_buffer_cache_time_sec": LaunchConfiguration("tf_buffer_cache_time_sec"),
                    "tf_lookup_timeout_sec": LaunchConfiguration("tf_lookup_timeout_sec"),
                    "max_frame_age_sec": 5.0,
                    "sync_queue_size": 100,
                    "sync_slop_sec": LaunchConfiguration("sync_slop_sec"),
                    "max_rgb_depth_skew_sec": LaunchConfiguration("max_rgb_depth_skew_sec"),
                    "max_tf_skew_sec": LaunchConfiguration("max_tf_skew_sec"),
                }],
            )
        )

    mapper_node = Node(
        package="mapping",
        executable="streaming_mapper",
        name="mapping_streaming_mapper",
        output="screen",
        parameters=[{
            "config_path": LaunchConfiguration("config_path"),
            "logger_level": LaunchConfiguration("logger_level"),
            "debug_queue_status": LaunchConfiguration("debug_queue_status"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "camera_names": camera_names,
            "expected_batch": len(camera_names),
            "max_pair_skew_sec": LaunchConfiguration("max_pair_skew_sec"),
            "latest_only": LaunchConfiguration("latest_only"),
            "caption_enabled": LaunchConfiguration("caption_enabled"),
            "segmenter_device": LaunchConfiguration("segmenter_device"),
            "segmenter_mask_erosion_px": LaunchConfiguration("segmenter_mask_erosion_px"),
            "segmenter_mahalanobis_thresh": LaunchConfiguration("segmenter_mahalanobis_thresh"),
            "scene_state_load_path": LaunchConfiguration("scene_state_load_path"),
            "scene_state_save_path": LaunchConfiguration("scene_state_save_path"),
            "scene_state_save_observations": LaunchConfiguration("scene_state_save_observations"),
            "scene_state_save_observation_view_limit": LaunchConfiguration("scene_state_save_observation_view_limit"),
            "scene_graph_json_save_path": LaunchConfiguration("scene_graph_json_save_path"),
            "scene_graph_snapshot_dir": LaunchConfiguration("scene_graph_snapshot_dir"),
            "scene_graph_publish_interval": LaunchConfiguration("scene_graph_publish_interval"),
            "scene_graph_json_save_enabled": LaunchConfiguration("scene_graph_json_save_enabled"),
            "scene_graph_snapshot_save_enabled": LaunchConfiguration("scene_graph_snapshot_save_enabled"),
            "scene_graph_snapshot_publish_interval": LaunchConfiguration("scene_graph_snapshot_publish_interval"),
            "storage_image_dir": LaunchConfiguration("storage_image_dir"),
        }],
    )

    return LaunchDescription([
        config_arg,
        logger_level_arg,
        debug_queue_arg,
        use_sim_time_arg,
        caption_enabled_arg,
        segmenter_device_arg,
        segmenter_mask_erosion_arg,
        segmenter_mahalanobis_arg,
        global_frame_arg,
        tf_buffer_cache_arg,
        tf_lookup_timeout_arg,
        sync_slop_arg,
        max_rgb_depth_skew_arg,
        max_tf_skew_arg,
        max_pair_skew_arg,
        latest_only_arg,
        scene_state_load_arg,
        scene_state_save_arg,
        scene_state_observations_arg,
        scene_state_view_limit_arg,
        scene_graph_json_save_path_arg,
        scene_graph_snapshot_dir_arg,
        storage_image_dir_arg,
        scene_graph_publish_interval_arg,
        scene_graph_json_save_enabled_arg,
        scene_graph_snapshot_save_enabled_arg,
        scene_graph_snapshot_publish_interval_arg,
        *frame_publishers,
        mapper_node,
    ])
