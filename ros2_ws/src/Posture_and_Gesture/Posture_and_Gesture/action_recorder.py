import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import SetBool


def _current_time_for_filename():
    now = time.time()
    local_time = time.localtime(now)
    milliseconds = int((now - int(now)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", local_time) + f"_{milliseconds:03d}"


class ActionRecorderNode(Node):
    """Records the raw camera stream as a video while an action is active.

    Detection nodes (gesture/posture) call the ``set_recording`` SetBool service
    when their action service is started/stopped. While recording is enabled the
    node subscribes to the raw image topic and writes every frame to an mp4 file
    plus a ``frames.txt`` timestamp log, so the footage can later be re-processed
    offline (e.g. multi-person RTMPose labelling).
    """

    def __init__(self):
        super().__init__('action_recorder')

        self.declare_parameter('input_topic', '/camera/camera_head/color/image_raw/compressed')
        self.declare_parameter('image_transport', 'compressed')
        self.declare_parameter('service_name', 'set_recording')
        self.declare_parameter('output_root_dir', '/home/thor/inha_log/module/posture_and_gesture/action_recordings')
        self.declare_parameter('video_fps', 15.0)
        self.declare_parameter('video_fourcc', 'mp4v')

        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.image_transport = self.get_parameter('image_transport').get_parameter_value().string_value
        self.service_name = self.get_parameter('service_name').get_parameter_value().string_value
        self.output_root_dir = Path(
            self.get_parameter('output_root_dir').get_parameter_value().string_value
        )
        self.video_fps = max(1.0, self.get_parameter('video_fps').get_parameter_value().double_value)
        self.video_fourcc = self.get_parameter('video_fourcc').get_parameter_value().string_value or 'mp4v'

        self.qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._recording = False
        self._sub = None
        self._video_writer = None
        self._video_path = None
        self._frame_index = 0
        self._timestamp_file = None

        self.service = self.create_service(
            SetBool, self.service_name, self.set_recording_callback
        )

        self.get_logger().info(
            f"ActionRecorderNode ready. service='{self.service_name}', "
            f"input_topic='{self.input_topic}', output_root_dir='{self.output_root_dir}'"
        )

    def set_recording_callback(self, request, response):
        if request.data:
            started = self._start_recording()
            response.success = True
            response.message = "Recording started." if started else "Recording already active."
        else:
            self._stop_recording()
            response.success = True
            response.message = "Recording stopped."
        return response

    def _start_recording(self):
        if self._recording:
            return False
        self._recording = True
        self._video_writer = None
        self._video_path = None
        self._frame_index = 0
        self._timestamp_file = None
        self._sub = self.create_subscription(
            CompressedImage if self.image_transport == 'compressed' else Image,
            self.input_topic,
            self.image_callback,
            self.qos_best_effort,
        )
        self.get_logger().info("Action recording enabled.")
        return True

    def _stop_recording(self):
        if not self._recording:
            return
        self._recording = False
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._release_writer()
        self.get_logger().info("Action recording disabled.")

    def image_callback(self, msg):
        if not self._recording:
            return

        if self.image_transport == 'compressed':
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image frame.")
                return
        else:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)

        if self._video_writer is None:
            self._open_writer(frame)
            if self._video_writer is None:
                return

        self._video_writer.write(frame)
        if self._timestamp_file is not None:
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._timestamp_file.write(
                f"{self._frame_index} {time.time():.6f} {stamp:.6f}\n"
            )
        self._frame_index += 1

    def _open_writer(self, frame):
        action_dir = self.output_root_dir / f"action_{_current_time_for_filename()}"
        try:
            action_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.get_logger().warn(f"Failed to create recording directory {action_dir}: {exc}")
            return

        height, width = frame.shape[:2]
        video_path = action_dir / f"raw_{_current_time_for_filename()}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)
        writer = cv2.VideoWriter(str(video_path), fourcc, self.video_fps, (width, height))
        if not writer.isOpened():
            self.get_logger().warn(f"Failed to open video writer: {video_path}")
            return

        self._video_writer = writer
        self._video_path = video_path
        self._frame_index = 0
        try:
            self._timestamp_file = open(action_dir / "frames.txt", "w")
            self._timestamp_file.write("# index wall_time stamp_sec\n")
        except OSError as exc:
            self.get_logger().warn(f"Failed to open frame timestamp log: {exc}")
            self._timestamp_file = None
        self.get_logger().info(f"Raw action video will be saved to {video_path}")

    def _release_writer(self):
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
        self._video_path = None
        self._frame_index = 0
        if self._timestamp_file is not None:
            try:
                self._timestamp_file.close()
            except OSError:
                pass
            self._timestamp_file = None


def main(args=None):
    rclpy.init(args=args)
    node = ActionRecorderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
