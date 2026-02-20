#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
from collections import deque
import threading

# Custom messages
from qcar2_object_detections.msg import (
    PersonDetection,
    TrafficLightDetection,
    StopSignDetection,
    ZebraCrossingDetection
)


class DetectionFilterNode(Node):

    def __init__(self):
        super().__init__('detection_filter_node')

        self.bridge = CvBridge()
        self.current_image = None
        self.zebra_image = None
        self.lock = threading.Lock()

        # ============================================================
        # PARAMETERS (ORIGINAL)
        # ============================================================
        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('detections_input_topic', '/detections_output')
        self.declare_parameter('zebra_image_topic', '/camera/csi_image_3')

        self.declare_parameter('person_output_topic', '/detections/person')
        self.declare_parameter('traffic_light_output_topic', '/detections/traffic_light')
        self.declare_parameter('stop_sign_output_topic', '/detections/stop_sign')
        self.declare_parameter('zebra_output_topic', '/detections/zebra_crossing')

        self.declare_parameter('person_class_id', '0')
        self.declare_parameter('traffic_light_class_id', '9')
        self.declare_parameter('stop_sign_class_id', '11')

        self.declare_parameter('person_roi', [0, 0, 640, 640])
        self.declare_parameter('traffic_light_roi', [0, 0, 640, 640])
        self.declare_parameter('stop_sign_roi', [0, 0, 640, 640])

        self.declare_parameter('min_confidence', 0.5)

        # Zebra params
        self.declare_parameter('zebra_enabled', True)
        self.declare_parameter('zebra_roi_top', 0.80)
        self.declare_parameter('zebra_roi_bottom', 0.985)
        self.declare_parameter('zebra_roi_width', 0.40)
        self.declare_parameter('zebra_min_stripes', 4)
        self.declare_parameter('zebra_max_stripes', 7)
        self.declare_parameter('zebra_vote_threshold', 5)
        self.declare_parameter('zebra_vote_window', 7)

        # ============================================================
        # GET PARAMETERS
        # ============================================================
        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_input_topic').value
        self.zebra_image_topic = self.get_parameter('zebra_image_topic').value

        self.person_output_topic = self.get_parameter('person_output_topic').value
        self.traffic_light_output_topic = self.get_parameter('traffic_light_output_topic').value
        self.stop_sign_output_topic = self.get_parameter('stop_sign_output_topic').value
        self.zebra_output_topic = self.get_parameter('zebra_output_topic').value

        self.person_class_id = self.get_parameter('person_class_id').value
        self.traffic_light_class_id = self.get_parameter('traffic_light_class_id').value
        self.stop_sign_class_id = self.get_parameter('stop_sign_class_id').value

        self.person_roi = self.get_parameter('person_roi').value
        self.traffic_light_roi = self.get_parameter('traffic_light_roi').value
        self.stop_sign_roi = self.get_parameter('stop_sign_roi').value

        self.min_confidence = self.get_parameter('min_confidence').value

        # Zebra
        self.zebra_enabled = self.get_parameter('zebra_enabled').value
        self.zebra_roi_top = self.get_parameter('zebra_roi_top').value
        self.zebra_roi_bottom = self.get_parameter('zebra_roi_bottom').value
        self.zebra_roi_width = self.get_parameter('zebra_roi_width').value
        self.zebra_min_stripes = self.get_parameter('zebra_min_stripes').value
        self.zebra_max_stripes = self.get_parameter('zebra_max_stripes').value
        self.zebra_vote_threshold = self.get_parameter('zebra_vote_threshold').value
        self.zebra_vote_window = self.get_parameter('zebra_vote_window').value

        self.zebra_votes = deque(maxlen=self.zebra_vote_window)
        self.last_zebra_state = None

        # ============================================================
        # 🔥 TU DETECTOR AVANZADO
        # ============================================================
        self.tl_last_state = "NONE"

        self.roi_x_min = 0.25
        self.roi_x_max = 0.65
        self.roi_y_min = 0.00
        self.roi_y_max = 0.45

        self.min_bbox_area = 600

        self.confirm_frames = 3
        self.candidate = "NONE"
        self.candidate_count = 0

        self.debug_view = False
        self.window_name = "traffic_light_debug"

        # ============================================================
        # SUBSCRIBERS
        # ============================================================
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10)

        self.detection_sub = self.create_subscription(
            Detection2DArray, self.detections_topic,
            self.detection_callback, 10)

        if self.zebra_enabled and self.zebra_image_topic != self.image_topic:
            self.zebra_image_sub = self.create_subscription(
                Image, self.zebra_image_topic,
                self.zebra_image_callback, 10)

        # ============================================================
        # PUBLISHERS
        # ============================================================
        self.person_pub = self.create_publisher(
            PersonDetection, self.person_output_topic, 10)

        self.traffic_light_pub = self.create_publisher(
            TrafficLightDetection, self.traffic_light_output_topic, 10)

        self.stop_sign_pub = self.create_publisher(
            StopSignDetection, self.stop_sign_output_topic, 10)

        self.zebra_pub = self.create_publisher(
            ZebraCrossingDetection, self.zebra_output_topic, 10)

        if self.zebra_enabled:
            self.zebra_thread = threading.Thread(
                target=self._zebra_loop, daemon=True)
            self.zebra_thread.start()

        self.get_logger().info("Detection Filter Node Initialized")

    # ============================================================
    # CALLBACKS
    # ============================================================
    def image_callback(self, msg: Image):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.zebra_image_topic == self.image_topic:
                with self.lock:
                    self.zebra_image = self.current_image.copy()
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def zebra_image_callback(self, msg: Image):
        try:
            with self.lock:
                self.zebra_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Zebra image conversion error: {e}')

    def detection_callback(self, msg: Detection2DArray):
        best_person = None
        best_traffic_light = None
        best_stop_sign = None

        for det in msg.detections:
            if not det.results:
                continue

            class_id = str(det.results[0].hypothesis.class_id)
            score = det.results[0].hypothesis.score

            if score < self.min_confidence:
                continue

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y

            if class_id == self.person_class_id:
                if self._is_in_roi(cx, cy, self.person_roi):
                    if best_person is None or score > best_person[1]:
                        best_person = (det, score)

            elif class_id == self.traffic_light_class_id:
                if self._is_in_roi(cx, cy, self.traffic_light_roi):
                    if best_traffic_light is None or score > best_traffic_light[1]:
                        best_traffic_light = (det, score)

            elif class_id == self.stop_sign_class_id:
                if self._is_in_roi(cx, cy, self.stop_sign_roi):
                    if best_stop_sign is None or score > best_stop_sign[1]:
                        best_stop_sign = (det, score)

        self._publish_person(best_person, msg.header)
        self._publish_traffic_light(best_traffic_light, msg.header)
        self._publish_stop_sign(best_stop_sign, msg.header)

    # ============================================================
    # ROI HELPER
    # ============================================================
    def _is_in_roi(self, cx: float, cy: float, roi: list) -> bool:
        x1, y1, x2, y2 = roi
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # ============================================================
    # TRAFFIC LIGHT PUBLISHER (USA TU DETECTOR)
    # ============================================================
    def _publish_traffic_light(self, detection_data, header):
        msg = TrafficLightDetection()
        msg.header = header

        if detection_data is not None and self.current_image is not None:
            det, score = detection_data
            msg.detected = True
            msg.confidence = float(score)
            msg.state = self._analyze_traffic_light_color(det)
        else:
            msg.detected = False
            msg.confidence = 0.0
            msg.state = "unknown"

        self.traffic_light_pub.publish(msg)

    # ============================================================
    # 🔥 TU ANALIZADOR COMPLETO
    # ============================================================
    def _analyze_traffic_light_color(self, detection) -> str:
        try:
            import cv2

            if self.current_image is None:
                return "unknown"

            img = self.current_image
            h, w = img.shape[:2]

            rx1 = int(self.roi_x_min * w)
            rx2 = int(self.roi_x_max * w)
            ry1 = int(self.roi_y_min * h)
            ry2 = int(self.roi_y_max * h)
            roi_rect = (rx1, ry1, rx2, ry2)

            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            sx = detection.bbox.size_x
            sy = detection.bbox.size_y

            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                self.publish_stable("NONE")
                return self._map_state(self.tl_last_state)

            area = float(sx * sy)
            if area < self.min_bbox_area:
                self.publish_stable("NONE")
                return self._map_state(self.tl_last_state)

            x1 = int(cx - sx / 2)
            y1 = int(cy - sy / 2)
            x2 = int(cx + sx / 2)
            y2 = int(cy + sy / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            roi_tl = img[y1:y2, x1:x2]
            if roi_tl.size == 0:
                self.publish_stable("NONE")
                return self._map_state(self.tl_last_state)

            state = self.analyze_red_green(roi_tl)

            if state not in ("RED", "GREEN"):
                state = "UNKNOWN"

            self.publish_stable(state)

            if self.debug_view:
                self.show_debug(img, roi_rect, (x1, y1, x2, y2),
                                self.tl_last_state, area)

            return self._map_state(self.tl_last_state)

        except Exception as e:
            self.get_logger().error(f'Traffic light analysis error: {e}')
            return "unknown"

    # ============================================================
    # 🔥 FUNCIONES TUYAS
    # ============================================================
    def publish_stable(self, raw_state: str):
        if raw_state == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = raw_state
            self.candidate_count = 1

        if self.candidate_count >= self.confirm_frames:
            if self.tl_last_state != self.candidate:
                self.tl_last_state = self.candidate

    def analyze_red_green(self, roi_bgr):
        import cv2

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 160, 160])
        upper_red = np.array([10, 255, 255])

        lower_yellow_kill = np.array([20, 30, 240])
        upper_yellow_kill = np.array([45, 160, 255])

        lower_green = np.array([46, 140, 140])
        upper_green = np.array([90, 255, 255])

        mask_yellow = cv2.inRange(hsv, lower_yellow_kill, upper_yellow_kill)
        roi_clean = cv2.bitwise_and(
            hsv, hsv, mask=cv2.bitwise_not(mask_yellow))

        mask_r = cv2.inRange(roi_clean, lower_red, upper_red)
        mask_g = cv2.inRange(roi_clean, lower_green, upper_green)

        r_px = cv2.countNonZero(mask_r)
        g_px = cv2.countNonZero(mask_g)

        umbral_sensibilidad = 8

        if r_px > umbral_sensibilidad and r_px > g_px:
            return "RED"
        elif g_px > umbral_sensibilidad and g_px > r_px:
            return "GREEN"

        return "UNKNOWN"

    def _map_state(self, state_str: str) -> str:
        state_str = state_str.upper()

        if state_str == "RED":
            return "red"
        elif state_str == "GREEN":
            return "green"
        elif state_str == "NONE":
            return "unknown"
        else:
            return "yellow"

    def show_debug(self, img_bgr, roi_rect, bbox_rect, state, area):
        import cv2
        dbg = img_bgr.copy()

        rx1, ry1, rx2, ry2 = roi_rect
        cv2.rectangle(dbg, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)

        if bbox_rect is not None:
            x1, y1, x2, y2 = bbox_rect
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.putText(dbg, f"{state} | area={int(area)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)

        cv2.imshow(self.window_name, dbg)
        cv2.waitKey(1)

    # ============================================================
    # (ZEBRA Y PERSON/STOP SIGUEN IGUAL EN TU PROYECTO)
    # ============================================================


def main(args=None):
    rclpy.init(args=args)
    node = DetectionFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()