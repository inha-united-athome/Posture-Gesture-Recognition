from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_topic = LaunchConfiguration("input_topic")
    image_transport = LaunchConfiguration("image_transport")
    recorder_service = LaunchConfiguration("recorder_service")
    recorder_output_dir = LaunchConfiguration("recorder_output_dir")
    recorder_fps = LaunchConfiguration("recorder_fps")

    return LaunchDescription([
        DeclareLaunchArgument(
            "input_topic",
            default_value="/camera/camera_head/color/image_raw/compressed",
            description="Raw camera image topic shared by the detector and the recorder.",
        ),
        DeclareLaunchArgument(
            "image_transport",
            default_value="compressed",
            description="'compressed' or 'raw' image transport.",
        ),
        DeclareLaunchArgument(
            "recorder_service",
            default_value="/action_recorder/set_recording",
            description="SetBool service the detector calls to start/stop recording.",
        ),
        DeclareLaunchArgument(
            "recorder_output_dir",
            default_value="/home/thor/inha_log/module/posture_and_gesture/action_recordings",
            description="Root directory for raw action recordings.",
        ),
        DeclareLaunchArgument(
            "recorder_fps",
            default_value="15.0",
            description="Assumed FPS for the recorded mp4 (timing is recoverable via frames.txt).",
        ),
        Node(
            package="Posture_and_Gesture",
            executable="posture_detect_node",
            name="posture_detect_node",
            output="screen",
            parameters=[{
                "input_topic": input_topic,
                "image_transport": image_transport,
                "recorder_enable_service": recorder_service,
                "recorder_trigger_enabled": True,
            }],
        ),
        Node(
            package="Posture_and_Gesture",
            executable="action_recorder_node",
            name="action_recorder",
            output="screen",
            parameters=[{
                "input_topic": input_topic,
                "image_transport": image_transport,
                "service_name": "set_recording",
                "output_root_dir": recorder_output_dir,
                "video_fps": recorder_fps,
                "video_fourcc": "mp4v",
            }],
        ),
    ])
