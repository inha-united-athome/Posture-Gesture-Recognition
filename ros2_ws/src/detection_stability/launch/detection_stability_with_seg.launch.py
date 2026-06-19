from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config_file = PathJoinSubstitution([
        FindPackageShare("detection_stability"),
        "config",
        "detection_stability_with_seg.yaml",
    ])
    config_file = LaunchConfiguration("config_file")

    return LaunchDescription([
        DeclareLaunchArgument(
            "config_file",
            default_value=default_config_file,
            description="Path to the ROS 2 parameter file for both nodes.",
        ),
        Node(
            package="detection_stability",
            executable="yolo_instance_seg_node",
            name="yolo_instance_seg_node",
            output="screen",
            parameters=[config_file],
        ),
        Node(
            package="detection_stability",
            executable="detection_stability_node",
            name="detection_stability_node",
            output="screen",
            parameters=[config_file],
        ),
    ])
