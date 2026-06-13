from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    detections_topic = LaunchConfiguration("detections_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    lidar_topic = LaunchConfiguration("lidar_topic")
    image_topic = LaunchConfiguration("image_topic")
    instance_mask_topic = LaunchConfiguration("instance_mask_topic")
    output_topic = LaunchConfiguration("output_topic")
    selected_point_topic = LaunchConfiguration("selected_point_topic")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_device = LaunchConfiguration("yolo_device")
    yolo_conf = LaunchConfiguration("yolo_conf")
    yolo_iou = LaunchConfiguration("yolo_iou")
    yolo_imgsz = LaunchConfiguration("yolo_imgsz")
    yolo_erode_pixels = LaunchConfiguration("yolo_erode_pixels")

    return LaunchDescription([
        DeclareLaunchArgument(
            "detections_topic",
            default_value="/gesture_and_posture/detection/detections",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera_head/color/camera_info",
        ),
        DeclareLaunchArgument("lidar_topic", default_value="/livox/lidar"),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/camera_head/color/image_raw/compressed",
        ),
        DeclareLaunchArgument(
            "instance_mask_topic",
            default_value="/gesture_and_posture/person_instance_mask",
        ),
        DeclareLaunchArgument(
            "output_topic",
            default_value="/gesture_and_posture/detection_stability",
        ),
        DeclareLaunchArgument(
            "selected_point_topic",
            default_value="/gesture_and_posture/selected_person/point",
        ),
        DeclareLaunchArgument("yolo_model_path", default_value="yolo26n-seg.pt"),
        DeclareLaunchArgument("yolo_device", default_value="cuda"),
        DeclareLaunchArgument("yolo_conf", default_value="0.35"),
        DeclareLaunchArgument("yolo_iou", default_value="0.50"),
        DeclareLaunchArgument("yolo_imgsz", default_value="640"),
        DeclareLaunchArgument("yolo_erode_pixels", default_value="3"),
        Node(
            package="detection_stability",
            executable="yolo_instance_seg_node",
            name="yolo_instance_seg_node",
            output="screen",
            parameters=[{
                "input_topic": image_topic,
                "output_topic": instance_mask_topic,
                "model_path": yolo_model_path,
                "device": yolo_device,
                "conf": ParameterValue(yolo_conf, value_type=float),
                "iou": ParameterValue(yolo_iou, value_type=float),
                "imgsz": ParameterValue(yolo_imgsz, value_type=int),
                "erode_pixels": ParameterValue(yolo_erode_pixels, value_type=int),
            }],
        ),
        Node(
            package="detection_stability",
            executable="detection_stability_node",
            name="detection_stability_node",
            output="screen",
            parameters=[{
                "detections_topic": detections_topic,
                "camera_info_topic": camera_info_topic,
                "lidar_topic": lidar_topic,
                "image_topic": image_topic,
                "instance_mask_topic": instance_mask_topic,
                "output_topic": output_topic,
                "selected_point_topic": selected_point_topic,
                "use_instance_mask_depth": True,
            }],
        ),
    ])
