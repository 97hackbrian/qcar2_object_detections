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
from sensor_msgs.msg import Image
import numpy as np


class ImagePreprocessorNode(Node):
    """
    ROS2 Node that preprocesses images by resizing with padding.
    
    Subscribes to an input image topic and publishes resized images
    with letterbox padding to maintain aspect ratio.
    """

    def __init__(self):
        super().__init__('image_preprocessor_node')
        
        # Declare parameters
        self.declare_parameter('input_image_topic', '/camera/color_image')
        self.declare_parameter('output_image_topic', '/image')
        self.declare_parameter('input_width', 640)
        self.declare_parameter('input_height', 480)
        self.declare_parameter('target_width', 640)
        self.declare_parameter('target_height', 640)
        self.declare_parameter('padding_color', [0, 0, 0])
        self.declare_parameter('input_encoding', 'bgr8')
        
        # Get parameters
        self.input_topic = self.get_parameter('input_image_topic').value
        self.output_topic = self.get_parameter('output_image_topic').value
        self.input_width = self.get_parameter('input_width').value
        self.input_height = self.get_parameter('input_height').value
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.padding_color = self.get_parameter('padding_color').value
        self.input_encoding = self.get_parameter('input_encoding').value
        
        # Calculate padding offsets for centered letterbox
        self._calculate_padding()
        
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
        
        self.get_logger().info(
            f'Image Preprocessor initialized:\n'
            f'  Input: {self.input_topic} ({self.input_width}x{self.input_height})\n'
            f'  Output: {self.output_topic} ({self.target_width}x{self.target_height})\n'
            f'  Padding: {self.padding_color}'
        )

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
