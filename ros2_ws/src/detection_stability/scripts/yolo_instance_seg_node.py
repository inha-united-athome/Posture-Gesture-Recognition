#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class YoloInstanceSegNode(Node):
    def __init__(self):
        super().__init__("yolo_instance_seg_node")

        self.declare_parameter("input_topic", "/camera/camera_head/color/image_raw/compressed")
        self.declare_parameter("output_topic", "/gesture_and_posture/person_instance_mask")
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
        model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.person_class_id = int(self.get_parameter("person_class_id").value)
        self.mask_threshold = float(self.get_parameter("mask_threshold").value)
        self.erode_pixels = max(0, int(self.get_parameter("erode_pixels").value))
        self.max_instances = max(1, min(65535, int(self.get_parameter("max_instances").value)))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for yolo_instance_seg_node. "
                "Install it in the Python environment used to run ROS."
            ) from exc

        self.model = YOLO(model_path)
        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.on_image,
            10,
        )

        self.get_logger().info(
            f"YOLO instance segmentation ready: input={self.input_topic}, "
            f"output={self.output_topic}, model={model_path}, device={self.device}"
        )

    def on_image(self, msg):
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
