#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection Filter Node for QCar2 Object Detection

This node filters YoloV8 detections by ROI, analyzes traffic light colors,
detects stop signs, and processes zebra crossings.

Publishes only the highest confidence detection per class within its ROI.

Author: QCar2 Developer
License: MIT
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import numpy as np
from collections import deque
import threading

# Import custom messages (generated after build)
from qcar2_object_detections.msg import (
    PersonDetection,
    TrafficLightDetection,
    StopSignDetection,
    ZebraCrossingDetection
)


class DetectionFilterNode(Node):
    """
    ROS2 Node that filters YoloV8 detections by ROI and publishes
    specialized detection messages for each object class.
    
    Features:
    - ROI-based filtering for person, traffic light, stop sign
    - Traffic light color analysis using HSV
    - Zebra crossing detection using image processing
    - Publishes only highest confidence detection per class
    """

    def __init__(self):
        super().__init__('detection_filter_node')
        
        self.bridge = CvBridge()
        self.current_image = None
        self.zebra_image = None
        self.lock = threading.Lock()
        
        # =====================================================================
        # DECLARE PARAMETERS
        # =====================================================================
        
        # Input topics
        self.declare_parameter('image_topic', '/image')
        self.declare_parameter('detections_input_topic', '/detections_output')
        self.declare_parameter('zebra_image_topic', '/camera/csi_image_3')
        
        # Output topics
        self.declare_parameter('person_output_topic', '/detections/person')
        self.declare_parameter('traffic_light_output_topic', '/detections/traffic_light')
        self.declare_parameter('stop_sign_output_topic', '/detections/stop_sign')
        self.declare_parameter('zebra_output_topic', '/detections/zebra_crossing')
        
        # Class IDs (COCO dataset)
        self.declare_parameter('person_class_id', '0')
        self.declare_parameter('traffic_light_class_id', '9')
        self.declare_parameter('stop_sign_class_id', '11')
        
        # ROIs [x1, y1, x2, y2]
        self.declare_parameter('person_roi', [0, 0, 640, 640])
        self.declare_parameter('traffic_light_roi', [0, 0, 640, 640])
        self.declare_parameter('stop_sign_roi', [0, 0, 640, 640])
        
        # Confidence threshold
        self.declare_parameter('min_confidence', 0.5)
        
        # Zebra crossing parameters
        self.declare_parameter('zebra_enabled', True)
        self.declare_parameter('zebra_roi_top', 0.80)
        self.declare_parameter('zebra_roi_bottom', 0.985)
        self.declare_parameter('zebra_roi_width', 0.40)
        self.declare_parameter('zebra_min_stripes', 4)
        self.declare_parameter('zebra_max_stripes', 7)
        self.declare_parameter('zebra_vote_threshold', 5)
        self.declare_parameter('zebra_vote_window', 7)
        
        # Traffic light HSV thresholds
        self.declare_parameter('traffic_light_red_lower', [0, 160, 160])
        self.declare_parameter('traffic_light_red_upper', [10, 255, 255])
        self.declare_parameter('traffic_light_yellow_lower', [20, 30, 240])
        self.declare_parameter('traffic_light_yellow_upper', [45, 160, 255])
        self.declare_parameter('traffic_light_green_lower', [46, 140, 140])
        self.declare_parameter('traffic_light_green_upper', [90, 255, 255])
        self.declare_parameter('traffic_light_sensitivity', 8)
        
        # =====================================================================
        # GET PARAMETERS
        # =====================================================================
        
        # Topics
        self.image_topic = self.get_parameter('image_topic').value
        self.detections_topic = self.get_parameter('detections_input_topic').value
        self.zebra_image_topic = self.get_parameter('zebra_image_topic').value
        
        self.person_output_topic = self.get_parameter('person_output_topic').value
        self.traffic_light_output_topic = self.get_parameter('traffic_light_output_topic').value
        self.stop_sign_output_topic = self.get_parameter('stop_sign_output_topic').value
        self.zebra_output_topic = self.get_parameter('zebra_output_topic').value
        
        # Class IDs
        self.person_class_id = self.get_parameter('person_class_id').value
        self.traffic_light_class_id = self.get_parameter('traffic_light_class_id').value
        self.stop_sign_class_id = self.get_parameter('stop_sign_class_id').value
        
        # ROIs
        self.person_roi = self.get_parameter('person_roi').value
        self.traffic_light_roi = self.get_parameter('traffic_light_roi').value
        self.stop_sign_roi = self.get_parameter('stop_sign_roi').value
        
        # Confidence
        self.min_confidence = self.get_parameter('min_confidence').value
        
        # Zebra parameters
        self.zebra_enabled = self.get_parameter('zebra_enabled').value
        self.zebra_roi_top = self.get_parameter('zebra_roi_top').value
        self.zebra_roi_bottom = self.get_parameter('zebra_roi_bottom').value
        self.zebra_roi_width = self.get_parameter('zebra_roi_width').value
        self.zebra_min_stripes = self.get_parameter('zebra_min_stripes').value
        self.zebra_max_stripes = self.get_parameter('zebra_max_stripes').value
        self.zebra_vote_threshold = self.get_parameter('zebra_vote_threshold').value
        self.zebra_vote_window = self.get_parameter('zebra_vote_window').value
        
        # Traffic light HSV
        self.red_lower = np.array(self.get_parameter('traffic_light_red_lower').value)
        self.red_upper = np.array(self.get_parameter('traffic_light_red_upper').value)
        self.yellow_lower = np.array(self.get_parameter('traffic_light_yellow_lower').value)
        self.yellow_upper = np.array(self.get_parameter('traffic_light_yellow_upper').value)
        self.green_lower = np.array(self.get_parameter('traffic_light_green_lower').value)
        self.green_upper = np.array(self.get_parameter('traffic_light_green_upper').value)
        self.tl_sensitivity = self.get_parameter('traffic_light_sensitivity').value
        
        # Zebra voting buffer
        self.zebra_votes = deque(maxlen=self.zebra_vote_window)
        self.last_zebra_state = None
        
        # =====================================================================
        # SUBSCRIBERS
        # =====================================================================
        
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )
        
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            self.detections_topic,
            self.detection_callback,
            10
        )
        
        if self.zebra_enabled and self.zebra_image_topic != self.image_topic:
            self.zebra_image_sub = self.create_subscription(
                Image,
                self.zebra_image_topic,
                self.zebra_image_callback,
                10
            )
        
        # =====================================================================
        # PUBLISHERS
        # =====================================================================
        
        self.person_pub = self.create_publisher(
            PersonDetection,
            self.person_output_topic,
            10
        )
        
        self.traffic_light_pub = self.create_publisher(
            TrafficLightDetection,
            self.traffic_light_output_topic,
            10
        )
        
        self.stop_sign_pub = self.create_publisher(
            StopSignDetection,
            self.stop_sign_output_topic,
            10
        )
        
        self.zebra_pub = self.create_publisher(
            ZebraCrossingDetection,
            self.zebra_output_topic,
            10
        )
        
        # Start zebra detection thread if enabled
        if self.zebra_enabled:
            self.zebra_thread = threading.Thread(target=self._zebra_loop, daemon=True)
            self.zebra_thread.start()
        
        self._log_configuration()

    def _log_configuration(self):
        """Log current configuration."""
        self.get_logger().info(
            f'\n{"="*60}\n'
            f'Detection Filter Node Initialized\n'
            f'{"="*60}\n'
            f'Input Topics:\n'
            f'  - Image: {self.image_topic}\n'
            f'  - Detections: {self.detections_topic}\n'
            f'  - Zebra Image: {self.zebra_image_topic}\n'
            f'\nOutput Topics:\n'
            f'  - Person: {self.person_output_topic}\n'
            f'  - Traffic Light: {self.traffic_light_output_topic}\n'
            f'  - Stop Sign: {self.stop_sign_output_topic}\n'
            f'  - Zebra: {self.zebra_output_topic}\n'
            f'\nClass IDs (COCO):\n'
            f'  - Person: {self.person_class_id}\n'
            f'  - Traffic Light: {self.traffic_light_class_id}\n'
            f'  - Stop Sign: {self.stop_sign_class_id}\n'
            f'\nROIs [x1, y1, x2, y2]:\n'
            f'  - Person: {self.person_roi}\n'
            f'  - Traffic Light: {self.traffic_light_roi}\n'
            f'  - Stop Sign: {self.stop_sign_roi}\n'
            f'\nMin Confidence: {self.min_confidence}\n'
            f'Zebra Detection: {"Enabled" if self.zebra_enabled else "Disabled"}\n'
            f'{"="*60}'
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    
    def image_callback(self, msg: Image):
        """Store current image for traffic light analysis."""
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # If zebra uses same topic, update zebra image too
            if self.zebra_image_topic == self.image_topic:
                with self.lock:
                    self.zebra_image = self.current_image.copy()
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
    
    def zebra_image_callback(self, msg: Image):
        """Store zebra detection image (if different from main image)."""
        try:
            with self.lock:
                self.zebra_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Zebra image conversion error: {e}')
    
    def detection_callback(self, msg: Detection2DArray):
        """
        Process YoloV8 detections and publish filtered results.
        
        Only publishes the highest confidence detection per class
        that falls within its respective ROI.
        """
        # Initialize best detections per class
        best_person = None
        best_traffic_light = None
        best_stop_sign = None
        
        for det in msg.detections:
            if not det.results:
                continue
            
            class_id = str(det.results[0].hypothesis.class_id)
            score = det.results[0].hypothesis.score
            
            # Skip low confidence detections
            if score < self.min_confidence:
                continue
            
            # Get bounding box center
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            
            # Check each class
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
        
        # Publish results
        self._publish_person(best_person, msg.header)
        self._publish_traffic_light(best_traffic_light, msg.header)
        self._publish_stop_sign(best_stop_sign, msg.header)

    # =========================================================================
    # ROI HELPER
    # =========================================================================
    
    def _is_in_roi(self, cx: float, cy: float, roi: list) -> bool:
        """Check if point (cx, cy) is inside ROI [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = roi
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # =========================================================================
    # PUBLISHERS
    # =========================================================================
    
    def _publish_person(self, detection_data, header):
        """Publish person detection message."""
        msg = PersonDetection()
        msg.header = header
        
        if detection_data is not None:
            msg.detected = True
            msg.confidence = float(detection_data[1])
            self.get_logger().info(
                f'PERSON detected - confidence: {msg.confidence:.2f}'
            )
        else:
            msg.detected = False
            msg.confidence = 0.0
        
        self.person_pub.publish(msg)
    
    def _publish_traffic_light(self, detection_data, header):
        """Publish traffic light detection with color analysis."""
        msg = TrafficLightDetection()
        msg.header = header
        
        if detection_data is not None and self.current_image is not None:
            det, score = detection_data
            msg.detected = True
            msg.confidence = float(score)
            
            # Analyze color
            msg.state = self._analyze_traffic_light_color(det)
            
            self.get_logger().info(
                f'TRAFFIC LIGHT detected - state: {msg.state}, '
                f'confidence: {msg.confidence:.2f}'
            )
        else:
            msg.detected = False
            msg.confidence = 0.0
            msg.state = "unknown"
        
        self.traffic_light_pub.publish(msg)
    
    def _publish_stop_sign(self, detection_data, header):
        """Publish stop sign detection message."""
        msg = StopSignDetection()
        msg.header = header
        
        if detection_data is not None:
            msg.detected = True
            msg.confidence = float(detection_data[1])
            self.get_logger().info(
                f'STOP SIGN detected - confidence: {msg.confidence:.2f}'
            )
        else:
            msg.detected = False
            msg.confidence = 0.0
        
        self.stop_sign_pub.publish(msg)

    # =========================================================================
    # TRAFFIC LIGHT COLOR ANALYSIS
    # =========================================================================
    
    def _analyze_traffic_light_color(self, detection) -> str:
        """
        Analyze traffic light color using HSV color space.
        
        Based on color_rv.py logic - excludes yellow and compares red vs green pixels.
        
        Args:
            detection: Detection2D message with bounding box
            
        Returns:
            Color state: "red", "green", "yellow", or "unknown"
        """
        try:
            # Import cv2 here to avoid issues if not installed
            import cv2
            
            # Extract bounding box
            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            sx = detection.bbox.size_x
            sy = detection.bbox.size_y
            
            x1 = int(cx - sx / 2)
            y1 = int(cy - sy / 2)
            x2 = int(cx + sx / 2)
            y2 = int(cy + sy / 2)
            
            # Clip to image bounds
            h_img, w_img = self.current_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            # Extract ROI
            roi = self.current_image[y1:y2, x1:x2]
            if roi.size == 0:
                return "unknown"
            
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Create yellow exclusion mask
            mask_yellow = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
            
            # Clean ROI (remove yellow)
            roi_clean = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(mask_yellow))
            
            # Detect red and green
            mask_red = cv2.inRange(roi_clean, self.red_lower, self.red_upper)
            mask_green = cv2.inRange(roi_clean, self.green_lower, self.green_upper)
            
            red_pixels = cv2.countNonZero(mask_red)
            green_pixels = cv2.countNonZero(mask_green)
            
            # Decision logic
            if red_pixels > self.tl_sensitivity and red_pixels > green_pixels:
                return "red"
            elif green_pixels > self.tl_sensitivity and green_pixels > red_pixels:
                return "green"
            else:
                # Yellow was dominant or no clear light
                return "yellow"
                
        except Exception as e:
            self.get_logger().error(f'Traffic light color analysis error: {e}')
            return "unknown"

    # =========================================================================
    # ZEBRA CROSSING DETECTION
    # =========================================================================
    
    def _zebra_loop(self):
        """
        Background thread for zebra crossing detection.
        
        Based on s_cebra.py logic - detects parallel stripes pattern.
        """
        import time
        
        while rclpy.ok():
            with self.lock:
                img = None if self.zebra_image is None else self.zebra_image.copy()
            
            if img is not None:
                present_raw, stripe_count = self._detect_zebra(img)
                
                # Voting for stability
                self.zebra_votes.append(1 if present_raw else 0)
                present = sum(self.zebra_votes) >= self.zebra_vote_threshold
                
                # Publish if state changed
                if present != self.last_zebra_state:
                    self.last_zebra_state = present
                    self._publish_zebra(present, stripe_count)
            
            time.sleep(0.05)  # 20 Hz
    
    def _detect_zebra(self, img_bgr: np.ndarray) -> tuple:
        """
        Detect zebra crossing using stripe pattern analysis.
        
        Args:
            img_bgr: Input BGR image
            
        Returns:
            Tuple of (detected: bool, stripe_count: int)
        """
        try:
            import cv2
            
            h, w = img_bgr.shape[:2]
            
            # Calculate ROI coordinates
            rt = int(h * self.zebra_roi_top)
            rb = int(h * self.zebra_roi_bottom)
            roi_w = int(w * self.zebra_roi_width)
            cx = w // 2
            rl = max(0, cx - roi_w // 2)
            rr = min(w, cx + roi_w // 2)
            
            # Extract ROI
            roi = img_bgr[rt:rb, rl:rr]
            if roi.size == 0:
                return False, -1
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Use bottom band of ROI
            band_top = int(gray.shape[0] * 0.20)
            band = gray[band_top:, :]
            
            # Blur and binarize
            band_blur = cv2.GaussianBlur(band, (5, 5), 0)
            _, bw = cv2.threshold(
                band_blur, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Morphological opening to remove noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Sample multiple rows to count transitions
            H = bw.shape[0]
            sample_rows = np.linspace(
                int(H * 0.65), int(H * 0.95), 8
            ).astype(int)
            
            trans_counts = []
            for y in sample_rows:
                if y >= H:
                    continue
                row = (bw[y, :] > 0).astype(np.uint8)
                transitions = int(np.sum(row[1:] != row[:-1]))
                trans_counts.append(transitions)
            
            if not trans_counts:
                return False, -1
            
            # Median of transitions
            trans_med = float(np.median(trans_counts))
            stripe_count = int(np.floor(trans_med + 1))
            
            # Decision based on stripe count range
            present = self.zebra_min_stripes <= stripe_count <= self.zebra_max_stripes
            
            return present, stripe_count
            
        except Exception as e:
            self.get_logger().error(f'Zebra detection error: {e}')
            return False, -1
    
    def _publish_zebra(self, detected: bool, stripe_count: int):
        """Publish zebra crossing detection message."""
        msg = ZebraCrossingDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.detected = detected
        msg.stripe_count = stripe_count
        
        if detected:
            self.get_logger().info(
                f'ZEBRA CROSSING detected - stripes: {stripe_count}'
            )
        
        self.zebra_pub.publish(msg)


def main(args=None):
    """Main entry point."""
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
