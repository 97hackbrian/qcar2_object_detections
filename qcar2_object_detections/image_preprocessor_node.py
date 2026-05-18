#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Preprocessor Node for QCar2 Object Detection

This node resizes input images to a target size with padding (letterbox)
for YoloV8 inference compatibility.

Author: QCar2 Developer
License: MIT
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
import numpy as np
import cv2


class ImagePreprocessorNode(Node):
    """
    ROS2 Node that preprocesses images by resizing with padding.
    
    Subscribes to an input image topic and publishes resized images
    with letterbox padding to maintain aspect ratio.
    """

    def __init__(self):
        super().__init__('image_preprocessor_node')
        
        # Declare parameters
        self.declare_parameter('input_image_topic', '/camera/csi_image_2')
        self.declare_parameter('output_image_topic', '/image')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        self.declare_parameter('target_width', 640)
        self.declare_parameter('target_height', 640)
        self.declare_parameter('padding_color', [0, 0, 0])
        self.declare_parameter('input_encoding', 'bgr8')
        self.declare_parameter('clahe_enabled', True)
        self.declare_parameter('clahe_clip_limit', 5.0)
        self.declare_parameter('clahe_tile_size', 15)
        self.declare_parameter('brightness_adjustment', -55.0)
        self.declare_parameter('gamma_correction', 3)
        self.declare_parameter('sharpen_image', True)
        
        # Get parameters
        self.input_topic = self.get_parameter('input_image_topic').value
        self.output_topic = self.get_parameter('output_image_topic').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.padding_color = self.get_parameter('padding_color').value
        self.input_encoding = self.get_parameter('input_encoding').value
        self.clahe_enabled = self.get_parameter('clahe_enabled').value
        self.clahe_clip_limit = self.get_parameter('clahe_clip_limit').value
        self.clahe_tile_size = self.get_parameter('clahe_tile_size').value
        self.brightness_adjustment = self.get_parameter('brightness_adjustment').value
        self.gamma_correction = self.get_parameter('gamma_correction').value
        self.sharpen_image = self.get_parameter('sharpen_image').value
        
        # Calculate padding offsets for centered letterbox
        self._calculate_padding()
        
        # Initialize CLAHE if enabled
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )
        else:
            self.clahe = None
        
        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            10
        )
        
        # Add parameter callback for dynamic reconfiguration
        self.add_on_set_parameters_callback(self._on_parameters_changed)
        
        self.get_logger().info(
            f'Image Preprocessor initialized:\n'
            f'  Input: {self.input_topic} ({self.input_width}x{self.input_height})\n'
            f'  Output: {self.output_topic} ({self.target_width}x{self.target_height})\n'
            f'  Padding: {self.padding_color}\n'
            f'  CLAHE: {"Enabled (clip={}, tile={}x{})".format(self.clahe_clip_limit, self.clahe_tile_size, self.clahe_tile_size) if self.clahe_enabled else "Disabled"}\n'
            f'  Brightness adjustment: {self.brightness_adjustment}\n'
            f'  Gamma correction: {self.gamma_correction}\n'
            f'  Sharpen image: {self.sharpen_image}'
        )

    def _on_parameters_changed(self, params) -> SetParametersResult:
        """
        Callback for dynamic parameter reconfiguration.
        
        Called when parameters are changed via ros2 param set or rqt_gui.
        Allows real-time adjustment of CLAHE settings without node restart.
        
        Args:
            params: List of Parameter objects that changed
            
        Returns:
            SetParametersResult indicating success/failure
        """
        success = True
        for param in params:
            try:
                if param.name == 'clahe_enabled':
                    self.clahe_enabled = param.value
                    self._reinitialize_clahe()
                    self.get_logger().info(f'CLAHE: {"Enabled" if self.clahe_enabled else "Disabled"}')
                    
                elif param.name == 'clahe_clip_limit':
                    if param.value < 1.0 or param.value > 8.0:
                        self.get_logger().warning(f'clip_limit out of range [1.0, 8.0]: {param.value}')
                        success = False
                        continue
                    self.clahe_clip_limit = param.value
                    self._reinitialize_clahe()
                    self.get_logger().info(f'CLAHE clip_limit: {self.clahe_clip_limit}')
                    
                elif param.name == 'clahe_tile_size':
                    if param.value not in [4, 8, 16, 32, 64]:
                        self.get_logger().warning(f'tile_size should be power of 2: {param.value}')
                        success = False
                        continue
                    self.clahe_tile_size = param.value
                    self._reinitialize_clahe()
                    self.get_logger().info(f'CLAHE tile_size: {self.clahe_tile_size}x{self.clahe_tile_size}')
                    
                elif param.name == 'brightness_adjustment':
                    if param.value < -50.0 or param.value > 50.0:
                        self.get_logger().warning(f'brightness_adjustment out of range [-50, 50]: {param.value}')
                        success = False
                        continue
                    self.brightness_adjustment = param.value
                    self.get_logger().info(f'Brightness adjustment: {self.brightness_adjustment}')
                    
                elif param.name == 'gamma_correction':
                    if param.value <= 0.0 or param.value > 5.0:
                        self.get_logger().warning(f'gamma_correction out of range (0.0, 5.0]: {param.value}')
                        success = False
                        continue
                    self.gamma_correction = param.value
                    self.get_logger().info(f'Gamma correction: {self.gamma_correction}')
                    
                elif param.name == 'sharpen_image':
                    self.sharpen_image = param.value
                    self.get_logger().info(f'Sharpen image: {self.sharpen_image}')
                    
            except Exception as e:
                self.get_logger().error(f'Error updating parameter {param.name}: {e}')
                success = False
        
        return SetParametersResult(successful=success)

    def _reinitialize_clahe(self):
        """Reinitialize CLAHE object with current parameters."""
        if self.clahe_enabled:
            try:
                self.clahe = cv2.createCLAHE(
                    clipLimit=self.clahe_clip_limit,
                    tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
                )
                self.get_logger().debug(f'CLAHE reinitialized: clip={self.clahe_clip_limit}, tile={self.clahe_tile_size}')
            except Exception as e:
                self.get_logger().error(f'Error reinitializing CLAHE: {e}')
                self.clahe = None
        else:
            self.clahe = None

    def _calculate_padding(self):
        """Calculate padding offsets for centered letterbox."""
        # Calculate scale to fit image in target while maintaining aspect ratio
        scale_w = self.target_width / self.input_width
        scale_h = self.target_height / self.input_height
        self.scale = min(scale_w, scale_h)
        
        # New dimensions after scaling
        self.new_width = int(self.input_width * self.scale)
        self.new_height = int(self.input_height * self.scale)
        
        # Padding to center the image
        self.pad_left = (self.target_width - self.new_width) // 2
        self.pad_top = (self.target_height - self.new_height) // 2
        self.pad_right = self.pad_left + self.new_width
        self.pad_bottom = self.pad_top + self.new_height
        
        self.get_logger().debug(
            f'Padding calculated: scale={self.scale:.3f}, '
            f'offset=({self.pad_left}, {self.pad_top})'
        )

    def image_callback(self, msg: Image):
        """
        Process incoming image and publish resized version.
        
        Args:
            msg: Input Image message
        """
        # Validate message data
        expected_size = self.input_width * self.input_height * 3
        if not msg.data or len(msg.data) < expected_size:
            self.get_logger().warning(
                f'Invalid image data: expected {expected_size} bytes, '
                f'got {len(msg.data) if msg.data else 0}'
            )
            return
        
        try:
            # Convert to numpy array
            img_data = np.frombuffer(msg.data, dtype=np.uint8)
            img = img_data.reshape((self.input_height, self.input_width, 3))
            
            # Apply CLAHE for contrast enhancement
            if self.clahe_enabled:
                img = self._apply_clahe(img)
                # Enhance color saturation to recover lost colors in bright areas
                img = self._enhance_color_saturation(img, saturation_scale=1.3)
            
            # Apply brightness adjustment if needed
            if self.brightness_adjustment != 0.0:
                img = self._adjust_brightness(img, self.brightness_adjustment)
            
            # Create canvas with padding color
            canvas = np.full(
                (self.target_height, self.target_width, 3),
                self.padding_color,
                dtype=np.uint8
            )
            
            # If scale is 1.0 and dimensions match, direct copy
            if self.scale == 1.0 and self.new_width == self.input_width:
                canvas[self.pad_top:self.pad_bottom, 
                       self.pad_left:self.pad_right] = img
            else:
                # Resize image using simple nearest neighbor (no cv2 dependency)
                # For better quality, cv2.resize would be preferred
                resized = self._resize_nearest(img, self.new_width, self.new_height)
                canvas[self.pad_top:self.pad_bottom, 
                       self.pad_left:self.pad_right] = resized
            
            # Create output message
            out_msg = Image()
            out_msg.header = msg.header
            out_msg.height = self.target_height
            out_msg.width = self.target_width
            out_msg.encoding = self.input_encoding
            out_msg.step = self.target_width * 3
            out_msg.data = canvas.tobytes()
            
            self.publisher.publish(out_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        CLAHE enhances local contrast which helps detect objects that are
        too bright or too dim. Particularly useful for traffic lights and
        bright objects that lose detail.
        
        Args:
            img: Input BGR image array
            
        Returns:
            CLAHE-enhanced image with increased contrast
        """
        try:
            # Convert BGR to RGB then to HSV for better contrast control
            # HSV allows us to enhance contrast in the Value (brightness) channel
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Extract V (Value/brightness) channel
            v_channel = hsv[:, :, 2]
            
            # Apply CLAHE to V channel for local contrast enhancement
            v_enhanced = self.clahe.apply(v_channel)
            
            # Replace V channel
            hsv[:, :, 2] = v_enhanced
            
            # Convert back to BGR
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Reduce glare and separate bright lights from bright backgrounds (like yellow boxes)
            # by applying gamma correction instead of global histogram equalization which washes out highlights.
            # Gamma > 1.0 darkens midtones while preserving peak highlights.
            if hasattr(self, 'gamma_correction') and self.gamma_correction != 1.0:
                table = np.array([((i / 255.0) ** self.gamma_correction) * 255 for i in np.arange(0, 256)]).astype("uint8")
                result = cv2.LUT(result, table)
                
            # Apply unsharp mask to crisp up the edges of the lights
            # which helps YOLO distinguish the circular lights from the rectangular box
            if hasattr(self, 'sharpen_image') and self.sharpen_image:
                blurred = cv2.GaussianBlur(result, (0, 0), 2.0)
                result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
            
            return result
        except Exception as e:
            self.get_logger().warning(f'Error applying CLAHE: {e}. Returning original image.')
            return img

    def _adjust_brightness(self, img: np.ndarray, adjustment: float) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            img: Input image array
            adjustment: Brightness adjustment value (-50 to +50)
            
        Returns:
            Brightness-adjusted image (clipped to valid range)
        """
        try:
            # Convert to float for arithmetic
            img_float = img.astype(np.float32)
            
            # Apply brightness adjustment
            img_adjusted = img_float + adjustment
            
            # Clip to valid range [0, 255]
            img_adjusted = np.clip(img_adjusted, 0, 255)
            
            return img_adjusted.astype(np.uint8)
        except Exception as e:
            self.get_logger().warning(f'Error adjusting brightness: {e}. Returning original image.')
            return img

    def _enhance_color_saturation(self, img: np.ndarray, saturation_scale: float = 1.2) -> np.ndarray:
        """
        Enhance color saturation to recover lost colors in bright/washed-out areas.
        
        Increases saturation in HSV space to make colors more vivid,
        particularly useful for recovering traffic light colors.
        
        Args:
            img: Input BGR image array
            saturation_scale: Multiplier for saturation (1.0 = no change, 1.5 = 50% more vivid)
            
        Returns:
            Image with enhanced color saturation
        """
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Enhance saturation channel (index 1)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
            
            # Convert back to uint8 and BGR
            hsv = hsv.astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            return result
        except Exception as e:
            self.get_logger().warning(f'Error enhancing saturation: {e}. Returning original image.')
            return img

    def _resize_nearest(self, img: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
        """
        Resize image using nearest neighbor interpolation.
        
        Args:
            img: Input image array
            new_w: Target width
            new_h: Target height
            
        Returns:
            Resized image array
        """
        h, w = img.shape[:2]
        
        # If dimensions match, return original
        if new_w == w and new_h == h:
            return img
        
        # Create coordinate maps for nearest neighbor
        x_indices = (np.arange(new_w) * w / new_w).astype(int)
        y_indices = (np.arange(new_h) * h / new_h).astype(int)
        
        # Clip to valid range
        x_indices = np.clip(x_indices, 0, w - 1)
        y_indices = np.clip(y_indices, 0, h - 1)
        
        # Apply indexing
        return img[y_indices[:, None], x_indices]


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = ImagePreprocessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
