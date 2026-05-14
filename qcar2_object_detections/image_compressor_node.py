#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Compressor Node

Subscribes to an uncompressed Image topic and publishes a CompressedImage (JPEG).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageCompressorNode(Node):
    def __init__(self):
        super().__init__('image_compressor_node')
        
        self.bridge = CvBridge()
        
        self.declare_parameter('input_topic', '/yolov8_processed_image')
        self.declare_parameter('output_topic', '/yolov8_processed_image/compressed')
        self.declare_parameter('jpeg_quality', 80)
        
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            CompressedImage,
            self.output_topic,
            10
        )
        
        self.get_logger().info(
            f'Image Compressor initialized:\n'
            f'  Input: {self.input_topic}\n'
            f'  Output: {self.output_topic}\n'
            f'  Quality: {self.jpeg_quality}'
        )

    def image_callback(self, msg: Image):
        try:
            # Convert ROS Image to CV2
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Compress to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            success, encoded_image = cv2.imencode('.jpg', cv_img, encode_param)
            
            if success:
                # Create CompressedImage message
                comp_msg = CompressedImage()
                comp_msg.header = msg.header
                comp_msg.format = "jpeg"
                comp_msg.data = encoded_image.tobytes()
                
                # Publish
                self.publisher.publish(comp_msg)
            else:
                self.get_logger().error("Failed to compress image")
                
        except Exception as e:
            self.get_logger().error(f"Error compressing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageCompressorNode()
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
