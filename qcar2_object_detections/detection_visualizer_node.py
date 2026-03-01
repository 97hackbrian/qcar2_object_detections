#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection Visualizer Node for QCar2 Object Detection

Draws annotated bounding boxes on the YOLO-processed image for all classes
of interest (person, traffic_light, stop_sign) with color differentiation:
  - GREEN  : detection passes all filters (ROI + min area + min confidence)
  - GRAY   : detection is of a relevant class but does NOT pass filters

Also overlays a mini-view of the zebra camera with detection state.

Publishes the annotated image to /detections/debug_image.

Author: QCar2 Developer
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading

from qcar2_object_detections.msg import (
    PersonDetection,
    TrafficLightDetection,
    StopSignDetection,
    ZebraCrossingDetection,
)

# ── Color palette (BGR) ─────────────────────────────────────────────────────
COLOR_PASS = (0, 255, 0)        # green  – passes filters
COLOR_FAIL = (128, 128, 128)    # gray   – does NOT pass filters
COLOR_PERSON = (0, 255, 0)      # green
COLOR_STOP = (0, 0, 255)        # red
COLOR_TL = (0, 255, 255)        # yellow / cyan
COLOR_ZEBRA = (255, 255, 0)     # cyan-ish
COLOR_TEXT_BG = (0, 0, 0)       # black background for text


class DetectionVisualizerNode(Node):
    """Annotates the YOLO image with bounding boxes and publishes it."""

    def __init__(self):
        super().__init__('detection_visualizer_node')

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        # ── Cached frames ────────────────────────────────────────────────
        self._yolo_image = None       # /image  (preprocessed YOLO image)
        self._zebra_image = None      # /camera/csi_image_3

        # ── Latest stable states from detection_filter_node ──────────────
        self._person_state = PersonDetection()
        self._tl_state = TrafficLightDetection()
        self._stop_state = StopSignDetection()
        self._zebra_state = ZebraCrossingDetection()

        # =================================================================
        # PARAMETERS
        # =================================================================

        # -- Topics IN --
        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('detections_input_topic', '/detections_output')
        self.declare_parameter('zebra_image_topic', '/camera/csi_image_3')

        # Filtered state topics
        self.declare_parameter('person_state_topic', '/detections/person')
        self.declare_parameter('traffic_light_state_topic', '/detections/traffic_light')
        self.declare_parameter('stop_sign_state_topic', '/detections/stop_sign')
        self.declare_parameter('zebra_state_topic', '/detections/zebra_crossing')

        # -- Topic OUT --
        self.declare_parameter('output_image_topic', '/detections/debug_image')

        # -- Class IDs (must match detection_filter_node) --
        self.declare_parameter('person_class_id', '0')
        self.declare_parameter('traffic_light_class_id', '9')
        self.declare_parameter('stop_sign_class_id', '11')

        # -- Filter thresholds (mirror of detection_filter_node) --
        self.declare_parameter('min_confidence', 0.60)

        # Person ROI (%)
        self.declare_parameter('person_roi_x_min', 0.2)
        self.declare_parameter('person_roi_x_max', 0.8)
        self.declare_parameter('person_roi_y_min', 0.20)
        self.declare_parameter('person_roi_y_max', 0.85)
        self.declare_parameter('person_min_bbox_area', 10500)

        # Stop Sign ROI (%)
        self.declare_parameter('stop_sign_roi_x_min', 0.95)
        self.declare_parameter('stop_sign_roi_x_max', 1.0)
        self.declare_parameter('stop_sign_roi_y_min', 0.25)
        self.declare_parameter('stop_sign_roi_y_max', 0.45)
        self.declare_parameter('stop_sign_min_bbox_area', 2000)

        # Traffic Light ROI (%)
        self.declare_parameter('traffic_light_roi_x_min', 0.25)
        self.declare_parameter('traffic_light_roi_x_max', 0.75)
        self.declare_parameter('traffic_light_roi_y_min', 0.13)
        self.declare_parameter('traffic_light_roi_y_max', 0.45)
        self.declare_parameter('traffic_light_min_bbox_area', 600)

        # Zebra mini-view
        self.declare_parameter('zebra_miniview_scale', 0.25)
        self.declare_parameter('zebra_miniview_position', 'bottom_right')

        # =================================================================
        # GET PARAMETERS
        # =================================================================
        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_input_topic').value
        self.zebra_image_topic = self.get_parameter('zebra_image_topic').value

        self.person_state_topic = self.get_parameter('person_state_topic').value
        self.tl_state_topic = self.get_parameter('traffic_light_state_topic').value
        self.stop_state_topic = self.get_parameter('stop_sign_state_topic').value
        self.zebra_state_topic = self.get_parameter('zebra_state_topic').value

        self.output_topic = self.get_parameter('output_image_topic').value

        self.person_class_id = self.get_parameter('person_class_id').value
        self.tl_class_id = self.get_parameter('traffic_light_class_id').value
        self.stop_class_id = self.get_parameter('stop_sign_class_id').value

        self.min_confidence = float(self.get_parameter('min_confidence').value)

        # Person filter params
        self.p_roi_xmin = float(self.get_parameter('person_roi_x_min').value)
        self.p_roi_xmax = float(self.get_parameter('person_roi_x_max').value)
        self.p_roi_ymin = float(self.get_parameter('person_roi_y_min').value)
        self.p_roi_ymax = float(self.get_parameter('person_roi_y_max').value)
        self.p_min_area = float(self.get_parameter('person_min_bbox_area').value)

        # Stop filter params
        self.s_roi_xmin = float(self.get_parameter('stop_sign_roi_x_min').value)
        self.s_roi_xmax = float(self.get_parameter('stop_sign_roi_x_max').value)
        self.s_roi_ymin = float(self.get_parameter('stop_sign_roi_y_min').value)
        self.s_roi_ymax = float(self.get_parameter('stop_sign_roi_y_max').value)
        self.s_min_area = float(self.get_parameter('stop_sign_min_bbox_area').value)

        # Traffic light filter params
        self.tl_roi_xmin = float(self.get_parameter('traffic_light_roi_x_min').value)
        self.tl_roi_xmax = float(self.get_parameter('traffic_light_roi_x_max').value)
        self.tl_roi_ymin = float(self.get_parameter('traffic_light_roi_y_min').value)
        self.tl_roi_ymax = float(self.get_parameter('traffic_light_roi_y_max').value)
        self.tl_min_area = float(self.get_parameter('traffic_light_min_bbox_area').value)

        # Zebra mini-view
        self.zebra_mv_scale = float(self.get_parameter('zebra_miniview_scale').value)
        self.zebra_mv_pos = self.get_parameter('zebra_miniview_position').value

        # Class name map for labels
        self._class_names = {
            self.person_class_id: 'person',
            self.tl_class_id: 'traffic_light',
            self.stop_class_id: 'stop_sign',
        }

        # =================================================================
        # SUBSCRIBERS
        # =================================================================
        self.create_subscription(Image, self.image_topic,
                                 self._cb_image, 10)
        self.create_subscription(Detection2DArray, self.detections_topic,
                                 self._cb_detections, 10)
        self.create_subscription(Image, self.zebra_image_topic,
                                 self._cb_zebra_image, 10)

        # Stable-state subscribers
        self.create_subscription(PersonDetection, self.person_state_topic,
                                 self._cb_person_state, 10)
        self.create_subscription(TrafficLightDetection, self.tl_state_topic,
                                 self._cb_tl_state, 10)
        self.create_subscription(StopSignDetection, self.stop_state_topic,
                                 self._cb_stop_state, 10)
        self.create_subscription(ZebraCrossingDetection, self.zebra_state_topic,
                                 self._cb_zebra_state, 10)

        # =================================================================
        # PUBLISHER
        # =================================================================
        self.pub_debug = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(
            f'DetectionVisualizerNode ready | '
            f'image={self.image_topic} | detections={self.detections_topic} | '
            f'output={self.output_topic}'
        )

    # =====================================================================
    # CALLBACKS – cache latest data
    # =====================================================================
    def _cb_image(self, msg: Image):
        try:
            with self.lock:
                self._yolo_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def _cb_zebra_image(self, msg: Image):
        try:
            with self.lock:
                self._zebra_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Zebra image conversion error: {e}')

    def _cb_person_state(self, msg: PersonDetection):
        with self.lock:
            self._person_state = msg

    def _cb_tl_state(self, msg: TrafficLightDetection):
        with self.lock:
            self._tl_state = msg

    def _cb_stop_state(self, msg: StopSignDetection):
        with self.lock:
            self._stop_state = msg

    def _cb_zebra_state(self, msg: ZebraCrossingDetection):
        with self.lock:
            self._zebra_state = msg

    # =====================================================================
    # MAIN DETECTION CALLBACK – draw & publish
    # =====================================================================
    def _cb_detections(self, msg: Detection2DArray):
        with self.lock:
            img = self._yolo_image.copy() if self._yolo_image is not None else None
            zebra_img = self._zebra_image.copy() if self._zebra_image is not None else None
            p_state = self._person_state
            tl_state = self._tl_state
            s_state = self._stop_state
            z_state = self._zebra_state

        if img is None:
            return

        h, w = img.shape[:2]
        canvas = img.copy()

        # ── Pixel ROIs ───────────────────────────────────────────────────
        p_roi = self._pct_to_px(self.p_roi_xmin, self.p_roi_xmax,
                                self.p_roi_ymin, self.p_roi_ymax, w, h)
        s_roi = self._pct_to_px(self.s_roi_xmin, self.s_roi_xmax,
                                self.s_roi_ymin, self.s_roi_ymax, w, h)
        tl_roi = self._pct_to_px(self.tl_roi_xmin, self.tl_roi_xmax,
                                 self.tl_roi_ymin, self.tl_roi_ymax, w, h)

        # ── Draw each detection of interest ──────────────────────────────
        for det in msg.detections:
            if not det.results:
                continue

            class_id = str(det.results[0].hypothesis.class_id)
            score = float(det.results[0].hypothesis.score)

            if class_id not in self._class_names:
                continue

            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            sx = float(det.bbox.size_x)
            sy = float(det.bbox.size_y)
            area = sx * sy

            x1 = max(0, int(cx - sx / 2))
            y1 = max(0, int(cy - sy / 2))
            x2 = min(w, int(cx + sx / 2))
            y2 = min(h, int(cy + sy / 2))

            # Determine if detection passes the filters
            passes = self._check_filters(class_id, cx, cy, area, score,
                                         p_roi, s_roi, tl_roi)

            # Choose colours
            if passes:
                if class_id == self.person_class_id:
                    box_color = COLOR_PERSON
                elif class_id == self.stop_class_id:
                    box_color = COLOR_STOP
                else:
                    box_color = COLOR_TL
            else:
                box_color = COLOR_FAIL

            thickness = 2 if passes else 1
            label = f'{self._class_names[class_id]} {score:.2f}'

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            # Label background + text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x1, max(0, y1 - th - 6)),
                          (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(canvas, label, (x1 + 2, max(th + 2, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)

        # ── Stable-state overlay (bottom-left) ───────────────────────────
        self._draw_state_panel(canvas, p_state, tl_state, s_state, z_state)

        # ── Zebra mini-view (bottom-right) ───────────────────────────────
        if zebra_img is not None:
            self._draw_zebra_miniview(canvas, zebra_img, z_state)

        # ── Publish ──────────────────────────────────────────────────────
        try:
            out_msg = self.bridge.cv2_to_imgmsg(canvas, 'bgr8')
            out_msg.header = msg.header
            self.pub_debug.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'Publish error: {e}')

    # =====================================================================
    # FILTER CHECK – mirrors detection_filter_node logic
    # =====================================================================
    def _check_filters(self, class_id, cx, cy, area, score,
                       p_roi, s_roi, tl_roi):
        """Return True if the detection passes all filters for its class."""
        if score < self.min_confidence:
            return False

        if class_id == self.person_class_id:
            rx1, ry1, rx2, ry2 = p_roi
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                return False
            if area < self.p_min_area:
                return False
            return True

        elif class_id == self.stop_class_id:
            rx1, ry1, rx2, ry2 = s_roi
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                return False
            if area < self.s_min_area:
                return False
            return True

        elif class_id == self.tl_class_id:
            rx1, ry1, rx2, ry2 = tl_roi
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                return False
            if area < self.tl_min_area:
                return False
            return True

        return False

    # =====================================================================
    # DRAWING HELPERS
    # =====================================================================
    @staticmethod
    def _pct_to_px(xmin, xmax, ymin, ymax, w, h):
        return (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))

    @staticmethod
    def _draw_state_panel(canvas, p_state, tl_state, s_state, z_state):
        """Draw a small status panel at the bottom-left corner."""
        h, w = canvas.shape[:2]

        lines = [
            f'PERSON : {"YES" if p_state.detected else "no"}  ({p_state.confidence:.2f})',
            f'TL     : {tl_state.state if tl_state.detected else "---"}  ({tl_state.confidence:.2f})',
            f'STOP   : {"YES" if s_state.detected else "no"}  ({s_state.confidence:.2f})',
            f'ZEBRA  : {"YES" if z_state.detected else "no"}  (stripes={z_state.stripe_count})',
        ]

        line_colors = [
            COLOR_PERSON if p_state.detected else COLOR_FAIL,
            _tl_state_color(tl_state),
            COLOR_STOP if s_state.detected else COLOR_FAIL,
            COLOR_ZEBRA if z_state.detected else COLOR_FAIL,
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.50
        thick = 1
        line_h = 20
        pad = 6

        panel_h = len(lines) * line_h + pad * 2
        panel_w = 310
        y0 = h - panel_h
        x0 = 0

        # Semi-transparent background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, (line, color) in enumerate(zip(lines, line_colors)):
            ty = y0 + pad + (i + 1) * line_h - 4
            cv2.putText(canvas, line, (x0 + pad, ty),
                        font, scale, color, thick, cv2.LINE_AA)

    def _draw_zebra_miniview(self, canvas, zebra_img, z_state):
        """Insert a scaled-down zebra camera image in the bottom-right corner."""
        h_c, w_c = canvas.shape[:2]
        h_z, w_z = zebra_img.shape[:2]

        new_w = max(1, int(w_z * self.zebra_mv_scale))
        new_h = max(1, int(h_z * self.zebra_mv_scale))
        mini = cv2.resize(zebra_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Border colour depends on detection state
        border_color = COLOR_ZEBRA if z_state.detected else COLOR_FAIL
        mini = cv2.copyMakeBorder(mini, 2, 2, 2, 2,
                                  cv2.BORDER_CONSTANT, value=border_color)
        mh, mw = mini.shape[:2]

        # Position: bottom-right
        x_off = w_c - mw - 4
        y_off = h_c - mh - 4

        if x_off < 0 or y_off < 0:
            return  # canvas too small

        canvas[y_off:y_off + mh, x_off:x_off + mw] = mini

        # Label above mini-view
        label = f'ZEBRA: {"DETECTED" if z_state.detected else "---"}'
        cv2.putText(canvas, label, (x_off, max(12, y_off - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, border_color, 1,
                    cv2.LINE_AA)


# ─── Utility ─────────────────────────────────────────────────────────────────
def _tl_state_color(tl: TrafficLightDetection):
    """Return BGR colour matching the traffic-light state string."""
    if not tl.detected:
        return COLOR_FAIL
    s = tl.state.upper()
    if s == 'RED':
        return (0, 0, 255)
    elif s == 'GREEN':
        return (0, 255, 0)
    elif s == 'YELLOW':
        return (0, 200, 255)
    return COLOR_FAIL   # UNKNOWN / NONE


# ─── Main ────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = DetectionVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
