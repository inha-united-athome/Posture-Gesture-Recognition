import torch
import cv2
import numpy as np

import sys
import os
import threading
from pathlib import Path

def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (
            (parent / 'models' / 'tcn.py').exists()
            and (parent / 'runtime' / 'frame_inferencer.py').exists()
            and (parent / 'data').exists()
        ):
            return parent
    raise RuntimeError(f"Could not locate repository root from {current}")


REPO_ROOT = _find_repo_root()
VENDORED_RTMLIB_ROOT = REPO_ROOT / 'rtmlib'
for path in (REPO_ROOT, VENDORED_RTMLIB_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


from models.tcn import TCN
from runtime.frame_inferencer import TCNFrameInferencer

from rclpy.node import Node
import rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from inha_interfaces.srv import GestureDetection
from vision_msgs.msg import BoundingBox2D, Detection2D, Detection2DArray, ObjectHypothesisWithPose
from Posture_and_Gesture.image_msg_utils import bgr8_to_jpeg_compressed_image


CLASS_COLORS = {
    "idle": (128, 128, 128),
    "waving": (0, 165, 255),
    "hands_up_single": (0, 0, 255),
    "hands_up_both": (0, 100, 255),
    "pointing": (0, 255, 0),
    "stop": (255, 0, 0),
    "unknown": (80, 80, 80),
}


def _bbox_from_detection(det, frame_shape, min_score=0.2, margin=8.0):
    if det.bbox_xyxy is not None:
        x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy]
    else:
        points = np.asarray(det.skeleton_xy, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2 or points.size == 0:
            return None

        valid = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
        if det.keypoint_scores:
            scores = np.asarray(det.keypoint_scores, dtype=np.float32)
            score_valid = np.zeros(len(points), dtype=bool)
            usable_scores = min(len(scores), len(points))
            score_valid[:usable_scores] = scores[:usable_scores] >= min_score
            valid &= score_valid
        valid &= (points[:, 0] > 1.0) & (points[:, 1] > 1.0)

        visible_points = points[valid]
        if visible_points.size == 0:
            return None

        x1 = float(np.min(visible_points[:, 0]) - margin)
        y1 = float(np.min(visible_points[:, 1]) - margin)
        x2 = float(np.max(visible_points[:, 0]) + margin)
        y2 = float(np.max(visible_points[:, 1]) + margin)

    height, width = frame_shape[:2]
    x1 = max(0.0, min(x1, float(width - 1)))
    y1 = max(0.0, min(y1, float(height - 1)))
    x2 = max(0.0, min(x2, float(width - 1)))
    y2 = max(0.0, min(y2, float(height - 1)))
    if x2 <= x1 + 1.0 or y2 <= y1 + 1.0:
        return None
    return x1, y1, x2, y2


def _draw_readable_text(frame, text, origin, font_scale=0.65, thickness=2):
    if not text:
        return

    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 6
    baseline_pad = 4
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = int(origin[0])
    y = int(origin[1])
    x = max(0, min(x, max(0, width - text_w - 2 * pad - 1)))
    y = max(text_h + pad + baseline_pad, min(y, max(text_h + pad + baseline_pad, height - pad)))

    top_left = (x, y - text_h - pad - baseline_pad)
    bottom_right = (x + text_w + 2 * pad, y + baseline + pad)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 0), -1)
    cv2.rectangle(frame, top_left, bottom_right, (255, 220, 0), 1)
    cv2.putText(frame, text, (x + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_readable_gesture_overlay(frame, detections):
    for det, bbox_xyxy in detections:
        if bbox_xyxy is None:
            if det.center_xy is None:
                continue
            label = f"TID:{det.track_id} {det.class_name} {det.confidence:.0%}"
            _draw_readable_text(frame, label, det.center_xy, font_scale=0.7, thickness=2)
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 220, 0), 2)

        label = f"TID:{det.track_id} {det.class_name} {det.confidence:.0%}"
        _draw_readable_text(frame, label, (x1, y1 - 8), font_scale=0.7, thickness=2)

        if det.probabilities:
            top_probs = sorted(det.probabilities.items(), key=lambda item: item[1], reverse=True)[:3]
            for row, (class_name, probability) in enumerate(top_probs):
                prob_text = f"{class_name}: {probability:.0%}"
                _draw_readable_text(
                    frame,
                    prob_text,
                    (x1, y2 + 24 + row * 26),
                    font_scale=0.55,
                    thickness=1,
                )


class GestureDetectNode(Node):
    def __init__(self):
        super().__init__('gesture_detection_node')

        # ROS QoS Profiles
        self.qos_best_effort = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Parameters
        self.declare_parameter('device','cuda')
        self.declare_parameter('det_frequency',1)
        self.declare_parameter('buffer_size', 60)
        self.declare_parameter('track_ttl', 45)
        self.declare_parameter('track_match_distance', 120)
        self.declare_parameter('mode', 'balanced')
        self.declare_parameter('max_persons_inference', 8)

        # ROS2 Parameters
        self.declare_parameter('package_name', 'Posture_and_Gesture')
        self.declare_parameter('input_topic','/camera/camera_head/color/image_raw/compressed')
        self.declare_parameter('image_transport','compressed')
        self.declare_parameter('output_topic', '/gesture_and_posture/detection')
        self.declare_parameter('output_image_topic', '')
        self.declare_parameter('output_detection_topic', '')
        self.declare_parameter('output_jpeg_quality', 80)

        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.det_frequency = self.get_parameter('det_frequency').get_parameter_value().integer_value
        self.track_ttl = self.get_parameter('track_ttl').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.track_match_distance = self.get_parameter('track_match_distance').get_parameter_value().integer_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        self.max_persons_inference = self.get_parameter('max_persons_inference').get_parameter_value().integer_value

        valid_modes = {'performance', 'balanced', 'lightweight'}
        if self.mode not in valid_modes:
            self.get_logger().warn(
                f"Unsupported mode '{self.mode}', falling back to 'balanced'. "
                f"Valid values: {sorted(valid_modes)}"
            )
            self.mode = 'balanced'

        self.package_name = self.get_parameter('package_name').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.image_transport = self.get_parameter('image_transport').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        output_detection_topic = self.get_parameter('output_detection_topic').get_parameter_value().string_value
        self.output_image_topic = output_image_topic or f'{self.output_topic}/image/compressed'
        self.output_detection_topic = output_detection_topic or f'{self.output_topic}/detections'
        self.output_jpeg_quality = self.get_parameter('output_jpeg_quality').get_parameter_value().integer_value

        self.get_logger().info(
            f"""PostureDetectNode initialized with device={self.device},
            det_frequency={self.det_frequency}, 
            track_ttl={self.track_ttl}, 
            track_match_distance={self.track_match_distance}, 
            mode={self.mode}, 
            max_persons_inference={self.max_persons_inference}"""
        )

        self.image_pub = self.create_publisher(
            CompressedImage,
            self.output_image_topic,
            self.qos_best_effort
        )

        self.detection_pub = self.create_publisher(
            Detection2DArray,
            self.output_detection_topic,
            self.qos_best_effort
        )

        self.gesture_service = self.create_service(
            GestureDetection,
            'gesture_detection_service',
            self.gesture_detection_callback
        )

        self.get_logger().info(
            f"""Subscribing to {self.input_topic} 
            with transport {self.image_transport}.
            Publishing compressed image to {self.output_image_topic}.
            Publishing detections to {self.output_detection_topic}."""
        )

        # Initialize model
        model_path = REPO_ROOT / 'models' / 'best_tcn_xsub.pth'

        if torch.cuda.is_available() and self.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.get_logger().warn("CUDA not available, using CPU instead.")

        checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=True)
        self.model = TCN(
            input_dim=checkpoint.get("input_dim", 130),
            num_classes=checkpoint.get("num_classes", 6),
            hidden_dims=checkpoint.get("hidden_dims", [64, 128, 128, 256]),
            kernel_size=checkpoint.get("kernel_size", 5),
            dropout=checkpoint.get("dropout", 0.3),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.inferencer = TCNFrameInferencer(
            model=self.model,
            device=self.device,
            num_classes=checkpoint.get("num_classes", 6),
            max_frames=checkpoint.get("max_frames", 120),
            num_joints=checkpoint.get("num_joints", 65),
            pose_device=str(self.device),
            class_colors=CLASS_COLORS,
        )
        
        # Event Trigger
        self._action_active = False
        self.target_class = []
        
        
    def start_subscription(self):
        self.sub = self.create_subscription(
            CompressedImage if self.image_transport == 'compressed' else Image,
            self.input_topic,
            self.image_callback,
            self.qos_best_effort
        )

    def image_callback(self, msg):
        if not self._action_active:
            return
        
        if self.image_transport == 'compressed':
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warn("Failed to decode compressed image frame.")
                return
        else:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        

        # Inference Posture with MLP
        result = self.inferencer.infer(frame, return_debug_image=True)

        ### result -> detections, debug_image, keypoints, scores
        ### result.detections -> track_id, det_index, class_name, confidence, keypoint_scores
        ###                      probabilities, center_xy, bbox_xyxy, skeleton_xy

        header = msg.header
        header.frame_id = "camera_head_color_optical_frame"

        detection_array = Detection2DArray()
        detection_array.header = header
        visible_debug_detections = []

        for det in result.detections:
            if self.target_class and det.class_name not in self.target_class:
                continue

            bbox_xyxy = _bbox_from_detection(det, result.debug_image.shape)
            if bbox_xyxy is None:
                continue

            visible_debug_detections.append((det, bbox_xyxy))

            bb = BoundingBox2D()
            bb.center.position.x = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)
            bb.center.position.y = float((bbox_xyxy[1] + bbox_xyxy[3]) / 2.0)
            bb.size_x = float(bbox_xyxy[2] - bbox_xyxy[0])
            bb.size_y = float(bbox_xyxy[3] - bbox_xyxy[1])

            detection = Detection2D()
            detection.header = header
            detection.bbox = bb
            detection.id = f"track:{det.track_id};det:{det.det_index}"

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(det.class_name)
            hypothesis.hypothesis.score = float(det.confidence)
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)

        _draw_readable_gesture_overlay(result.debug_image, visible_debug_detections)
        image_msg = bgr8_to_jpeg_compressed_image(
            header,
            result.debug_image,
            self.output_jpeg_quality,
        )

        if image_msg is None:
            self.get_logger().warn("Failed to encode debug image as JPEG.")
        else:
            self.image_pub.publish(image_msg)
        self.detection_pub.publish(detection_array)


    

    def gesture_detection_callback(self, request, response):
        """
        GestureDetection Service:
        - Request:
            bool start
            string[] target_class
        - Response:
            bool success
            string message
        """
        if request.start:
            self.target_class = list(request.target_class)
            self.get_logger().info(f"Received gesture detection request for class: {self.target_class}")
            self._start_action() 
        else:
            self.get_logger().info("Received gesture detection stop request.")
            self._stop_action()
        
        response.success = True
        response.message = "Gesture detection started." if request.start else "Gesture detection stopped."
        return response

    def _start_action(self):
        if not self._action_active:
            self.get_logger().info("Starting gesture detection service.")
            self._action_active = True
            self.start_subscription()

    def _stop_action(self):
        if self._action_active:
            self.get_logger().info("Stopping gesture detection service.")
            self._action_active = False
            self.target_class = []
            self.destroy_subscription(self.sub)

def main(args=None):
    rclpy.init(args=args)
    node = GestureDetectNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
