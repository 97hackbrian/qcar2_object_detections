#!/usr/bin/env python3
# -*- coding: utf-8 -*-




# --------------- DETECCION CUANDO EL OBJETO ESTA CERCA Y DENTRO DEL ROI ------------

# semaforo, persona, cebra, stop

# FIXED = DIF VISTA POSTERIOR Y FRONTAL DEL STOP

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge

import cv2
import numpy as np

from collections import deque
import threading
import time

from qcar2_object_detections.msg import (
    PersonDetection,
    TrafficLightDetection,
    StopSignDetection,
    ZebraCrossingDetection
)


class DetectionFilterNode(Node):
    def __init__(self):
        super().__init__('detection_filter_node')

        # Runtime state
        self.bridge = CvBridge()
        self.current_image = None          # Imagen usada por YOLO (misma geometría que detections)
        self.zebra_image = None            # Imagen para zebra (csi_image_3)
        self.lock = threading.Lock()

        # =====================================================================
        # 1) DECLARE PARAMETERS
        # =====================================================================

        # ----------------- Topics IN -----------------
        # IMPORTANTE: esta debe ser la imagen que corresponde a las detecciones YOLO
        self.declare_parameter('image_topic', '/yolo_v8_proseed_image')
        self.declare_parameter('detections_input_topic', '/detections_output')

        # Zebra sigue usando csi3
        self.declare_parameter('zebra_image_topic', '/camera/csi_image_3')

        # ----------------- Topics OUT ----------------
        self.declare_parameter('person_output_topic', '/detections/person')
        self.declare_parameter('traffic_light_output_topic', '/detections/traffic_light')
        self.declare_parameter('stop_sign_output_topic', '/detections/stop_sign')
        self.declare_parameter('zebra_output_topic', '/detections/zebra_crossing')

        # ----------------- Class IDs -----------------
        self.declare_parameter('person_class_id', '0')
        self.declare_parameter('traffic_light_class_id', '9')
        self.declare_parameter('stop_sign_class_id', '11')

        # ----------------- Confidence ----------------
        self.declare_parameter('min_confidence', 0.6)

        # =====================================================================
        # 2) STOP SIGN ROI (EN % - DERECHA CENTRO) + CERCA + DEBUG
        # =====================================================================
        self.declare_parameter('stop_sign_roi_x_min', 0.95)
        self.declare_parameter('stop_sign_roi_x_max', 1.0)

        self.declare_parameter('stop_sign_roi_y_min', 0.25)
        self.declare_parameter('stop_sign_roi_y_max', 0.45)

        # Cerca del auto
        self.declare_parameter('stop_sign_min_bbox_area', 2000) #2500

        self.declare_parameter('stop_sign_debug_view', True)
        self.declare_parameter('stop_sign_window_name', 'stop_debug')

        # =====================================================================
        # 3) ZEBRA PARAMETERS
        # =====================================================================
        self.declare_parameter('zebra_enabled', True)
        self.declare_parameter('zebra_roi_top', 0.74)
        self.declare_parameter('zebra_roi_bottom', 0.955)
        self.declare_parameter('zebra_roi_width', 0.40)
        self.declare_parameter('zebra_min_stripes', 4)
        self.declare_parameter('zebra_max_stripes', 7)
        self.declare_parameter('zebra_vote_threshold', 5)
        self.declare_parameter('zebra_vote_window', 7)

        self.declare_parameter('zebra_debug_view', True)
        self.declare_parameter('zebra_window_name', "zebra_debug")

        # =====================================================================
        # 4) TRAFFIC LIGHT PARAMETERS
        # =====================================================================
        self.declare_parameter('traffic_light_roi_x_min', 0.25)
        self.declare_parameter('traffic_light_roi_x_max', 0.75)
        self.declare_parameter('traffic_light_roi_y_min', 0.13)
        self.declare_parameter('traffic_light_roi_y_max', 0.45)

        self.declare_parameter('traffic_light_min_bbox_area', 600)
        self.declare_parameter('traffic_light_confirm_frames', 3)

        self.declare_parameter('traffic_light_debug_view', True)
        self.declare_parameter('traffic_light_sensitivity', 8)

        self.declare_parameter('traffic_light_red_lower', [0, 160, 160])
        self.declare_parameter('traffic_light_red_upper', [10, 255, 255])
        self.declare_parameter('traffic_light_yellow_lower', [20, 30, 240])
        self.declare_parameter('traffic_light_yellow_upper', [45, 160, 255])
        self.declare_parameter('traffic_light_green_lower', [46, 140, 140])
        self.declare_parameter('traffic_light_green_upper', [90, 255, 255])

        # =====================================================================
        # 5) PERSON PARAMETERS (con histeresis)
        # =====================================================================
        self.declare_parameter('person_roi_x_min', 0.25)
        self.declare_parameter('person_roi_x_max', 0.76)
        self.declare_parameter('person_roi_y_min', 0.20)
        self.declare_parameter('person_roi_y_max', 0.85)

        self.declare_parameter('person_min_bbox_area', 10500)

        self.declare_parameter('person_confirm_frames_on', 4)
        self.declare_parameter('person_confirm_frames_off', 8)

        self.declare_parameter('person_debug_view', True)
        self.declare_parameter('person_window_name', "person_debug")

        # =====================================================================
        # 6) GET PARAMETERS
        # =====================================================================
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

        self.min_confidence = float(self.get_parameter('min_confidence').value)

        # STOP ROI % + cerca
        self.stop_roi_x_min = float(self.get_parameter('stop_sign_roi_x_min').value)
        self.stop_roi_x_max = float(self.get_parameter('stop_sign_roi_x_max').value)
        self.stop_roi_y_min = float(self.get_parameter('stop_sign_roi_y_min').value)
        self.stop_roi_y_max = float(self.get_parameter('stop_sign_roi_y_max').value)
        self.stop_min_bbox_area = float(self.get_parameter('stop_sign_min_bbox_area').value)

        self.stop_debug_view = bool(self.get_parameter('stop_sign_debug_view').value)
        self.stop_window_name = str(self.get_parameter('stop_sign_window_name').value)

        # Zebra
        self.zebra_enabled = bool(self.get_parameter('zebra_enabled').value)
        self.zebra_roi_top = float(self.get_parameter('zebra_roi_top').value)
        self.zebra_roi_bottom = float(self.get_parameter('zebra_roi_bottom').value)
        self.zebra_roi_width = float(self.get_parameter('zebra_roi_width').value)
        self.zebra_min_stripes = int(self.get_parameter('zebra_min_stripes').value)
        self.zebra_max_stripes = int(self.get_parameter('zebra_max_stripes').value)
        self.zebra_vote_threshold = int(self.get_parameter('zebra_vote_threshold').value)
        self.zebra_vote_window = int(self.get_parameter('zebra_vote_window').value)
        self.zebra_debug_view = bool(self.get_parameter('zebra_debug_view').value)
        self.zebra_window_name = str(self.get_parameter('zebra_window_name').value)

        # Traffic light
        self.tl_roi_x_min = float(self.get_parameter('traffic_light_roi_x_min').value)
        self.tl_roi_x_max = float(self.get_parameter('traffic_light_roi_x_max').value)
        self.tl_roi_y_min = float(self.get_parameter('traffic_light_roi_y_min').value)
        self.tl_roi_y_max = float(self.get_parameter('traffic_light_roi_y_max').value)

        self.tl_min_bbox_area = float(self.get_parameter('traffic_light_min_bbox_area').value)
        self.tl_confirm_frames = int(self.get_parameter('traffic_light_confirm_frames').value)

        self.tl_debug_view = bool(self.get_parameter('traffic_light_debug_view').value)
        self.tl_sensitivity = int(self.get_parameter('traffic_light_sensitivity').value)

        self.red_lower = np.array(self.get_parameter('traffic_light_red_lower').value)
        self.red_upper = np.array(self.get_parameter('traffic_light_red_upper').value)
        self.yellow_lower = np.array(self.get_parameter('traffic_light_yellow_lower').value)
        self.yellow_upper = np.array(self.get_parameter('traffic_light_yellow_upper').value)
        self.green_lower = np.array(self.get_parameter('traffic_light_green_lower').value)
        self.green_upper = np.array(self.get_parameter('traffic_light_green_upper').value)

        # Person
        self.person_roi_x_min = float(self.get_parameter('person_roi_x_min').value)
        self.person_roi_x_max = float(self.get_parameter('person_roi_x_max').value)
        self.person_roi_y_min = float(self.get_parameter('person_roi_y_min').value)
        self.person_roi_y_max = float(self.get_parameter('person_roi_y_max').value)
        self.person_min_bbox_area = float(self.get_parameter('person_min_bbox_area').value)
        self.person_confirm_frames_on = int(self.get_parameter('person_confirm_frames_on').value)
        self.person_confirm_frames_off = int(self.get_parameter('person_confirm_frames_off').value)
        self.person_debug_view = bool(self.get_parameter('person_debug_view').value)
        self.person_window_name = str(self.get_parameter('person_window_name').value)

        # =====================================================================
        # 7) INTERNAL STATES
        # =====================================================================
        self.zebra_votes = deque(maxlen=self.zebra_vote_window)
        self.last_zebra_state = None

        self.tl_last_state = "NONE"
        self.tl_candidate = "NONE"
        self.tl_candidate_count = 0
        self.tl_window_name = "traffic_light_debug"

        self.person_last_state = False
        self.person_on_count = 0
        self.person_off_count = 0

        # =====================================================================
        # 8) SUBSCRIBERS
        # =====================================================================
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.detection_sub = self.create_subscription(Detection2DArray, self.detections_topic, self.detection_callback, 10)

        if self.zebra_enabled:
            self.zebra_image_sub = self.create_subscription(Image, self.zebra_image_topic, self.zebra_image_callback, 10)

        # =====================================================================
        # 9) PUBLISHERS
        # =====================================================================
        self.person_pub = self.create_publisher(PersonDetection, self.person_output_topic, 10)
        self.traffic_light_pub = self.create_publisher(TrafficLightDetection, self.traffic_light_output_topic, 10)
        self.stop_sign_pub = self.create_publisher(StopSignDetection, self.stop_sign_output_topic, 10)
        self.zebra_pub = self.create_publisher(ZebraCrossingDetection, self.zebra_output_topic, 10)

        # =====================================================================
        # 10) ZEBRA THREAD
        # =====================================================================
        if self.zebra_enabled:
            self.zebra_thread = threading.Thread(target=self._zebra_loop, daemon=True)
            self.zebra_thread.start()

        self.get_logger().info(
            f"DetectionFilterNode listo | image_topic={self.image_topic} | detections_topic={self.detections_topic} | "
            f"STOP ROI x=[{self.stop_roi_x_min:.2f},{self.stop_roi_x_max:.2f}] "
            f"y=[{self.stop_roi_y_min:.2f},{self.stop_roi_y_max:.2f}] "
            f"stop_min_area={int(self.stop_min_bbox_area)}"
        )

    # =============================================================================
    # IMAGE CALLBACKS
    # =============================================================================
    def image_callback(self, msg: Image):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')

    def zebra_image_callback(self, msg: Image):
        try:
            with self.lock:
                self.zebra_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Zebra image conversion error: {e}')

    # =============================================================================
    # STOP:  TRUE solo si hay ROJO presente en el bbox
    # =============================================================================
    def _stop_has_red_present(self, crop_bgr: np.ndarray) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        crop = cv2.resize(crop_bgr, (160, 160), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Rojo en HSV (dos rangos por el wrap de Hue)
        lower1 = np.array([0, 90, 70], dtype=np.uint8)
        upper1 = np.array([10, 255, 255], dtype=np.uint8)
        lower2 = np.array([170, 90, 70], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k, iterations=1)

        red_px = float(cv2.countNonZero(red_mask))
        total_px = float(red_mask.size)

        red_ratio = red_px / total_px  # 0..1

        # Si el bbox es gris/blanco predominante = FALSE
        # Umbral DE ROJO
        return red_ratio >= 0.03

    # =============================================================================
    # DETECTIONS CALLBACK
    # =============================================================================
    def detection_callback(self, msg: Detection2DArray):
        best_person = None
        best_person_bbox = None
        best_person_area = 0.0

        best_stop_sign = None
        best_stop_bbox = None
        best_stop_area = 0.0

        tl_best_bbox = None
        tl_best_area = 0.0
        tl_best_score = 0.0

        if self.current_image is None:
            self._publish_person(None, msg.header)
            self._publish_stop_sign(None, msg.header)
            self._publish_traffic_light_from_bbox(None, 0.0, (0, 0, 0, 0), msg.header)
            return

        img = self.current_image
        h, w = img.shape[:2]

        # STOP ROI sobre la imagen de detección
        sx1 = int(self.stop_roi_x_min * w)
        sx2 = int(self.stop_roi_x_max * w)
        sy1 = int(self.stop_roi_y_min * h)
        sy2 = int(self.stop_roi_y_max * h)
        stop_roi_rect = (sx1, sy1, sx2, sy2)

        # Traffic light ROI
        tl_rx1 = int(self.tl_roi_x_min * w)
        tl_rx2 = int(self.tl_roi_x_max * w)
        tl_ry1 = int(self.tl_roi_y_min * h)
        tl_ry2 = int(self.tl_roi_y_max * h)
        tl_roi_rect = (tl_rx1, tl_ry1, tl_rx2, tl_ry2)

        # Person ROI
        prx1 = int(self.person_roi_x_min * w)
        prx2 = int(self.person_roi_x_max * w)
        pry1 = int(self.person_roi_y_min * h)
        pry2 = int(self.person_roi_y_max * h)
        person_roi_rect = (prx1, pry1, prx2, pry2)

        for det in msg.detections:
            if not det.results:
                continue

            class_id = str(det.results[0].hypothesis.class_id)
            score = float(det.results[0].hypothesis.score)

            if score < self.min_confidence:
                continue

            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)

            # PERSON
            if class_id == self.person_class_id:
                sx = float(det.bbox.size_x)
                sy = float(det.bbox.size_y)

                if not (prx1 <= cx <= prx2 and pry1 <= cy <= pry2):
                    continue

                area = float(sx * sy)
                if area < self.person_min_bbox_area:
                    continue

                if area > best_person_area:
                    best_person_area = area
                    best_person = (det, score)

                    x1 = int(cx - sx / 2); y1 = int(cy - sy / 2)
                    x2 = int(cx + sx / 2); y2 = int(cy + sy / 2)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    best_person_bbox = (x1, y1, x2, y2)

            # STOP: dentro ROI + cerca (área mínima)
            elif class_id == self.stop_sign_class_id:
                sx_bb = float(det.bbox.size_x)
                sy_bb = float(det.bbox.size_y)

                # 1) dentro del ROI
                if not ((sx1 <= cx <= sx2) and (sy1 <= cy <= sy2)):
                    continue

                # 2) cerca del auto: área mínima
                area = float(sx_bb * sy_bb)
                if area < self.stop_min_bbox_area:
                    continue

                # elegir el mejor (por score); si prefieres por área: usa area > best_stop_area
                if best_stop_sign is None or score > best_stop_sign[1]:
                    best_stop_sign = (det, score)
                    best_stop_area = area

                    bx1 = int(cx - sx_bb / 2); by1 = int(cy - sy_bb / 2)
                    bx2 = int(cx + sx_bb / 2); by2 = int(cy + sy_bb / 2)
                    bx1 = max(0, bx1); by1 = max(0, by1)
                    bx2 = min(w, bx2); by2 = min(h, by2)
                    best_stop_bbox = (bx1, by1, bx2, by2)

            # TRAFFIC LIGHT
            elif class_id == self.traffic_light_class_id:
                sx = float(det.bbox.size_x)
                sy = float(det.bbox.size_y)

                if not (tl_rx1 <= cx <= tl_rx2 and tl_ry1 <= cy <= tl_ry2):
                    continue

                area = float(sx * sy)
                if area < self.tl_min_bbox_area:
                    continue

                if area > tl_best_area:
                    tl_best_area = area
                    tl_best_score = score

                    x1 = int(cx - sx / 2); y1 = int(cy - sy / 2)
                    x2 = int(cx + sx / 2); y2 = int(cy + sy / 2)
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w, x2); y2 = min(h, y2)
                    tl_best_bbox = (x1, y1, x2, y2)

        # ====== STOP: MEJORA (SOLO AÑADIDO) ======
        if best_stop_bbox is not None:
            bx1, by1, bx2, by2 = best_stop_bbox
            stop_crop = img[by1:by2, bx1:bx2]
            if not self._stop_has_red_present(stop_crop):
                best_stop_sign = None
                best_stop_bbox = None
                best_stop_area = 0.0
        # =========================================

        # Publish
        self._publish_person(best_person, msg.header)
        self._publish_stop_sign(best_stop_sign, msg.header)
        self._publish_traffic_light_from_bbox(tl_best_bbox, tl_best_score, tl_roi_rect, msg.header)

        # Debug
        if self.tl_debug_view:
            self._tl_show_debug(img, tl_roi_rect, tl_best_bbox, self.tl_last_state, tl_best_area)

        if self.person_debug_view:
            self._person_show_debug(img, person_roi_rect, best_person_bbox, best_person)

        if self.stop_debug_view:
            self._stop_show_debug(img, stop_roi_rect, best_stop_bbox, best_stop_sign, best_stop_area)

    # =============================================================================
    # PERSON - histeresis
    # =============================================================================
    def _person_publish_stable(self, raw_detected: bool) -> bool:
        if raw_detected:
            self.person_on_count += 1
            self.person_off_count = 0
            if (not self.person_last_state) and (self.person_on_count >= self.person_confirm_frames_on):
                self.person_last_state = True
                self.get_logger().info("PERSON STATE -> True")
        else:
            self.person_off_count += 1
            self.person_on_count = 0
            if self.person_last_state and (self.person_off_count >= self.person_confirm_frames_off):
                self.person_last_state = False
                self.get_logger().info("PERSON STATE -> False")
        return self.person_last_state

    def _publish_person(self, detection_data, header):
        msg = PersonDetection()
        msg.header = header
        msg.detected = bool(self._person_publish_stable(detection_data is not None))
        msg.confidence = float(detection_data[1]) if (msg.detected and detection_data is not None) else 0.0
        self.person_pub.publish(msg)

    def _person_show_debug(self, img_bgr, roi_rect, bbox_rect, best_person):
        dbg = img_bgr.copy()
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(dbg, "PERSON ROI", (x1 + 5, max(20, y1 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if bbox_rect is not None:
            bx1, by1, bx2, by2 = bbox_rect
            cv2.rectangle(dbg, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

        conf = 0.0 if best_person is None else float(best_person[1])
        txt = (
            f"stable={self.person_last_state} conf={conf:.2f} "
            f"on={self.person_on_count}/{self.person_confirm_frames_on} "
            f"off={self.person_off_count}/{self.person_confirm_frames_off}"
        )
        cv2.putText(dbg, txt, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow(self.person_window_name, dbg)
        cv2.waitKey(1)

    # =============================================================================
    # STOP SIGN
    # =============================================================================
    def _publish_stop_sign(self, detection_data, header):
        msg = StopSignDetection()
        msg.header = header
        msg.detected = detection_data is not None
        msg.confidence = float(detection_data[1]) if detection_data is not None else 0.0
        self.stop_sign_pub.publish(msg)

    def _stop_show_debug(self, img_bgr, roi_rect, bbox_rect, best_stop, best_stop_area):
        dbg = img_bgr.copy()
        x1, y1, x2, y2 = roi_rect
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(dbg, "STOP ROI (RIGHT-CENTER)", (x1 + 5, max(20, y1 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        if bbox_rect is not None:
            bx1, by1, bx2, by2 = bbox_rect
            cv2.rectangle(dbg, (bx1, by1), (bx2, by2), (255, 255, 255), 2)

        conf = 0.0 if best_stop is None else float(best_stop[1])
        raw = "YES" if best_stop is not None else "NO"
        cv2.putText(
            dbg,
            f"raw={raw} conf={conf:.2f} area={int(best_stop_area)} min={int(self.stop_min_bbox_area)}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

        cv2.imshow(self.stop_window_name, dbg)
        cv2.waitKey(1)

    # =============================================================================
    # TRAFFIC LIGHT
    # =============================================================================
    def _publish_traffic_light_from_bbox(self, best_bbox, best_score, roi_rect, header):
        out = TrafficLightDetection()
        out.header = header

        img = self.current_image
        if best_bbox is None or img is None:
            stable = self._tl_publish_stable("NONE")
            out.detected = False
            out.confidence = 0.0
            out.state = stable
            self.traffic_light_pub.publish(out)
            return

        x1, y1, x2, y2 = best_bbox
        roi_tl = img[y1:y2, x1:x2]
        if roi_tl.size == 0:
            stable = self._tl_publish_stable("NONE")
            out.detected = False
            out.confidence = 0.0
            out.state = stable
            self.traffic_light_pub.publish(out)
            return

        state = self._tl_analyze_red_green(roi_tl)
        stable = self._tl_publish_stable(state)

        out.detected = True
        out.confidence = float(best_score)
        out.state = stable
        self.traffic_light_pub.publish(out)

    def _tl_publish_stable(self, raw_state: str) -> str:
        if raw_state == self.tl_candidate:
            self.tl_candidate_count += 1
        else:
            self.tl_candidate = raw_state
            self.tl_candidate_count = 1

        if self.tl_candidate_count >= self.tl_confirm_frames:
            if self.tl_last_state != self.tl_candidate:
                self.tl_last_state = self.tl_candidate
                self.get_logger().info(f"TL STATE -> {self.tl_last_state}")

        return self.tl_last_state

    def _tl_analyze_red_green(self, roi_bgr: np.ndarray) -> str:
        try:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
            roi_clean = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(mask_yellow))

            mask_r = cv2.inRange(roi_clean, self.red_lower, self.red_upper)
            mask_g = cv2.inRange(roi_clean, self.green_lower, self.green_upper)

            r_px = cv2.countNonZero(mask_r)
            g_px = cv2.countNonZero(mask_g)

            if self.tl_debug_view:
                scale = 6
                cv2.imshow("mask_red", cv2.resize(mask_r, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
                cv2.imshow("mask_green", cv2.resize(mask_g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
                cv2.imshow("mask_yellow_kill", cv2.resize(mask_yellow, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
                cv2.imshow("tl_crop", cv2.resize(roi_bgr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
                cv2.waitKey(1)

            total = int(roi_bgr.shape[0] * roi_bgr.shape[1])
            min_px = max(int(self.tl_sensitivity), int(0.005 * total))
            dominance = 1.30

            if r_px >= min_px and r_px >= dominance * g_px:
                return "RED"
            elif g_px >= min_px and g_px >= dominance * r_px:
                return "GREEN"
            return "UNKNOWN"

        except Exception as e:
            self.get_logger().error(f"Traffic light color analysis error: {e}")
            return "UNKNOWN"

    def _tl_show_debug(self, img_bgr, roi_rect, bbox_rect, state, area):
        dbg = img_bgr.copy()
        rx1, ry1, rx2, ry2 = roi_rect

        cv2.rectangle(dbg, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
        cv2.putText(dbg, "TL ROI", (rx1 + 5, max(20, ry1 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if bbox_rect is not None:
            x1, y1, x2, y2 = bbox_rect
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 255, 255), 2)

        cv2.putText(dbg, f"{state} | area={int(area)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)

        cv2.imshow(self.tl_window_name, dbg)
        cv2.waitKey(1)

    # =============================================================================
    # ZEBRA (thread)
    # =============================================================================
    def _zebra_loop(self):
        while rclpy.ok():
            with self.lock:
                img = None if self.zebra_image is None else self.zebra_image.copy()

            if img is not None:
                present_raw, stripe_count, dbg = self._detect_zebra(img)

                self.zebra_votes.append(1 if present_raw else 0)
                present = sum(self.zebra_votes) >= self.zebra_vote_threshold

                if self.zebra_debug_view and dbg is not None:
                    cv2.imshow(self.zebra_window_name, dbg)
                    cv2.waitKey(1)

                if present != self.last_zebra_state:
                    self.last_zebra_state = present
                    self._publish_zebra(present, stripe_count)

            time.sleep(0.05)

    def _zebra_make_debug(self, img_bgr, roi_rect, present, stripe_count):
        try:
            dbg = img_bgr.copy()
            x1, y1, x2, y2 = roi_rect
            cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(dbg, "ZEBRA ROI", (x1 + 5, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            votes = sum(self.zebra_votes)
            txt = f"present_raw={present} stripes={stripe_count} votes={votes}/{self.zebra_vote_window}"
            cv2.putText(dbg, txt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            return dbg
        except Exception:
            return None

    def _detect_zebra(self, img_bgr: np.ndarray) -> tuple:
        try:
            h, w = img_bgr.shape[:2]

            rt = int(h * self.zebra_roi_top)
            rb = int(h * self.zebra_roi_bottom)
            roi_w = int(w * self.zebra_roi_width)
            cx = w // 2
            rl = max(0, cx - roi_w // 2)
            rr = min(w, cx + roi_w // 2)

            roi_rect = (rl, rt, rr, rb)
            roi = img_bgr[rt:rb, rl:rr]
            if roi.size == 0:
                dbg = self._zebra_make_debug(img_bgr, roi_rect, False, -1)
                return False, -1, dbg

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            band_top = int(gray.shape[0] * 0.20)
            band = gray[band_top:, :]

            band_blur = cv2.GaussianBlur(band, (5, 5), 0)
            _, bw = cv2.threshold(band_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

            H = bw.shape[0]
            sample_rows = np.linspace(int(H * 0.65), int(H * 0.95), 8).astype(int)

            trans_counts = []
            for y in sample_rows:
                if y >= H:
                    continue
                row = (bw[y, :] > 0).astype(np.uint8)
                transitions = int(np.sum(row[1:] != row[:-1]))
                trans_counts.append(transitions)

            if not trans_counts:
                dbg = self._zebra_make_debug(img_bgr, roi_rect, False, -1)
                if self.zebra_debug_view:
                    cv2.imshow("zebra_roi_crop", roi)
                    cv2.imshow("zebra_bw", bw)
                    cv2.waitKey(1)
                return False, -1, dbg

            trans_med = float(np.median(trans_counts))
            stripe_count = int(np.floor(trans_med + 1))
            present = self.zebra_min_stripes <= stripe_count <= self.zebra_max_stripes

            if self.zebra_debug_view:
                cv2.imshow("zebra_roi_crop", roi)
                cv2.imshow("zebra_bw", bw)
                cv2.waitKey(1)

            dbg = self._zebra_make_debug(img_bgr, roi_rect, present, stripe_count)
            return present, stripe_count, dbg

        except Exception as e:
            self.get_logger().error(f'Zebra detection error: {e}')
            return False, -1, None

    def _publish_zebra(self, detected: bool, stripe_count: int):
        msg = ZebraCrossingDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.detected = detected
        msg.stripe_count = stripe_count
        self.zebra_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()