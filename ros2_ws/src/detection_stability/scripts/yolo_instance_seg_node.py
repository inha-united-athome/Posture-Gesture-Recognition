#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from inha_interfaces.srv import SetEnable
from sensor_msgs.msg import CompressedImage, Image


class YoloInstanceSegNode(Node):
    def __init__(self):
        super().__init__("yolo_instance_seg_node")

        self.declare_parameter("input_topic", "/camera/camera_head/color/image_raw/compressed")
        self.declare_parameter("output_topic", "/gesture_and_posture/person_instance_mask")
        self.declare_parameter(
            "debug_image_topic",
            "/gesture_and_posture/person_instance_mask/debug/compressed",
        )
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("initially_enabled", False)
        self.declare_parameter("enable_service_name", "~/set_enable")
        self.declare_parameter("debug_alpha", 0.45)
        self.declare_parameter("debug_jpeg_quality", 80)
        self.declare_parameter("model_path", "yolo26n-seg.pt")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.50)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("person_class_id", 0)
        self.declare_parameter("mask_threshold", 0.50)
        self.declare_parameter("erode_pixels", 3)
        self.declare_parameter("max_instances", 32)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.enabled = bool(self.get_parameter("initially_enabled").value)
        enable_service_name = self.get_parameter("enable_service_name").value
        self.debug_alpha = float(self.get_parameter("debug_alpha").value)
        self.debug_jpeg_quality = int(self.get_parameter("debug_jpeg_quality").value)
        model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.person_class_id = int(self.get_parameter("person_class_id").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.erode_pixels = max(0, int(self.get_parameter("erode_pixels").value))
        self.max_instances = max(1, min(65535, int(self.get_parameter("max_instances").value)))
        self.debug_alpha = min(1.0, max(0.0, self.debug_alpha))
        self.debug_jpeg_quality = min(100, max(1, self.debug_jpeg_quality))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for yolo_instance_seg_node. "
                "Install it in the Python environment used to run ROS."
            ) from exc

        self.model = YOLO(model_path)
        self.sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        label_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.publisher = self.create_publisher(Image, self.output_topic, label_qos)
        self.debug_publisher = None
        if self.publish_debug_image:
            self.debug_publisher = self.create_publisher(
                CompressedImage,
                self.debug_image_topic,
                self.sensor_qos,
            )
        self.subscription = None
        self.enable_service = self.create_service(
            SetEnable,
            enable_service_name,
            self.on_set_enable,
        )
        if self.enabled:
            self.start_subscription()

        self.get_logger().info(
            f"YOLO instance segmentation ready: input={self.input_topic}, "
            f"output={self.output_topic}, debug={self.debug_image_topic}, "
            f"model={model_path}, device={self.device}, enabled={self.enabled}, "
            f"enable_service={enable_service_name}"
        )

    def on_set_enable(self, request, response):
        self.set_enabled(bool(request.enable))
        response.success = True
        response.message = "YOLO instance segmentation enabled." if self.enabled else (
            "YOLO instance segmentation disabled."
        )
        return response

    def set_enabled(self, enabled):
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if self.enabled:
            self.start_subscription()
            self.get_logger().info("YOLO instance segmentation enabled.")
        else:
            self.stop_subscription()
            self.get_logger().info("YOLO instance segmentation disabled.")

    def start_subscription(self):
        if self.subscription is not None:
            return
        self.subscription = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.on_image,
            self.sensor_qos,
        )

    def stop_subscription(self):
        if self.subscription is None:
            return
        self.destroy_subscription(self.subscription)
        self.subscription = None

    def on_image(self, msg):
        if not self.enabled:
            return
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn("Failed to decode compressed image.")
            return

        height, width = frame.shape[:2]
        label_map = np.zeros((height, width), dtype=np.uint16)

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if results:
            self.fill_person_instances(label_map, results[0], width, height)

        out = Image()
        out.header = msg.header
        out.height = height
        out.width = width
        out.encoding = "mono16"
        out.is_bigendian = 0
        out.step = width * 2
        out.data = label_map.tobytes()
        self.publisher.publish(out)

        if self.debug_publisher is not None:
            self.publish_debug_image_msg(msg.header, frame, label_map)

    def fill_person_instances(self, label_map, result, width, height):
        if result.masks is None or result.boxes is None:
            return

        masks = result.masks.data
        classes = result.boxes.cls
        confidences = result.boxes.conf
        if masks is None or classes is None or confidences is None:
            return

        masks = masks.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy().astype(np.int32)
        confidences = confidences.detach().cpu().numpy()

        order = np.argsort(-confidences)
        label = 1
        kernel = None
        if self.erode_pixels > 0:
            size = 2 * self.erode_pixels + 1
            kernel = np.ones((size, size), dtype=np.uint8)

        for idx in order:
            if label > self.max_instances:
                break
            if classes[idx] != self.person_class_id:
                continue

            mask = masks[idx]
            if mask.shape[0] != height or mask.shape[1] != width:
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)

            mask_u8 = (mask >= self.mask_threshold).astype(np.uint8)
            if kernel is not None:
                mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
            if not np.any(mask_u8):
                continue

            empty = label_map == 0
            label_map[(mask_u8 > 0) & empty] = label
            label += 1

    def publish_debug_image_msg(self, header, frame, label_map):
        overlay = frame.copy()
        labels = np.unique(label_map)
        labels = labels[labels > 0]

        for label in labels:
            mask = label_map == label
            overlay[mask] = self.label_color(int(label))

        if labels.size > 0:
            debug = cv2.addWeighted(
                overlay,
                self.debug_alpha,
                frame,
                1.0 - self.debug_alpha,
                0.0,
            )
        else:
            debug = frame

        ok, encoded = cv2.imencode(
            ".jpg",
            debug,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.debug_jpeg_quality],
        )
        if not ok:
            self.get_logger().warn("Failed to encode debug segmentation image.")
            return

        out = CompressedImage()
        out.header = header
        out.format = "bgr8; jpeg compressed bgr8"
        out.data = encoded.tobytes()
        self.debug_publisher.publish(out)

    @staticmethod
    def label_color(label):
        hue = (label * 47) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def main(args=None):
    rclpy.init(args=args)
    node = YoloInstanceSegNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
