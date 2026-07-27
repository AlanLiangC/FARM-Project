"""Launch the YOLOE text-prompt visual search node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from mapping.lib.paths import default_adaptation_results_dir


def generate_launch_description() -> LaunchDescription:
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Path to YOLOE checkpoint; empty uses the mapping package default.",
    )
    device_arg = DeclareLaunchArgument(
        "device",
        default_value="",
        description="Device for YOLOE inference (e.g., cuda, cuda:0, cpu).",
    )
    goal_topic_arg = DeclareLaunchArgument(
        "goal_topic",
        default_value="/spot/mapping/visual_search_goal",
        description="Topic for visual search text prompts.",
    )
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="validation",
        description="Operating mode: validation|adaptation.",
    )
    mode_topic_arg = DeclareLaunchArgument(
        "mode_topic",
        default_value="/spot/mapping/visual_search_mode",
        description="Topic used to switch mode at runtime (std_msgs/String).",
    )
    results_topic_arg = DeclareLaunchArgument(
        "results_topic",
        default_value="/spot/mapping/visual_search_results",
        description="Topic for detection results.",
    )
    run_when_goal_empty_arg = DeclareLaunchArgument(
        "run_when_goal_empty",
        default_value="false",
        description="Run inference even when no goal is set.",
    )
    debug_save_enable_arg = DeclareLaunchArgument(
        "debug_save_enable",
        default_value="false",
        description="Save per-detection debug images with bounding boxes.",
    )
    debug_save_dir_arg = DeclareLaunchArgument(
        "debug_save_dir",
        default_value="log/visual_search_detections",
        description="Directory for saving debug images.",
    )
    debug_save_jpeg_quality_arg = DeclareLaunchArgument(
        "debug_save_jpeg_quality",
        default_value="95",
        description="JPEG quality (1-100) for debug images.",
    )
    adaptation_results_dir_arg = DeclareLaunchArgument(
        "adaptation_results_dir",
        default_value=default_adaptation_results_dir(),
        description="Directory for adaptation snapshots and manifest outputs.",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time when /clock is available.",
    )

    node = Node(
        package="mapping",
        executable="visual_search_yoloe_text_prompt",
        name="visual_search_yoloe_text_prompt",
        output="screen",
        parameters=[{
            "model_path": LaunchConfiguration("model_path"),
            "device": LaunchConfiguration("device"),
            "mode": LaunchConfiguration("mode"),
            "mode_topic": LaunchConfiguration("mode_topic"),
            "goal_topic": LaunchConfiguration("goal_topic"),
            "results_topic": LaunchConfiguration("results_topic"),
            "run_when_goal_empty": LaunchConfiguration("run_when_goal_empty"),
            "debug_save_enable": LaunchConfiguration("debug_save_enable"),
            "debug_save_dir": LaunchConfiguration("debug_save_dir"),
            "debug_save_jpeg_quality": LaunchConfiguration("debug_save_jpeg_quality"),
            "adaptation_results_dir": LaunchConfiguration("adaptation_results_dir"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
        }],
    )

    return LaunchDescription([
        model_path_arg,
        device_arg,
        mode_arg,
        mode_topic_arg,
        goal_topic_arg,
        results_topic_arg,
        run_when_goal_empty_arg,
        debug_save_enable_arg,
        debug_save_dir_arg,
        debug_save_jpeg_quality_arg,
        adaptation_results_dir_arg,
        use_sim_time_arg,
        node,
    ])
