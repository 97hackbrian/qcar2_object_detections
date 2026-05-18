#!/usr/bin/env python3
"""
Dynamic CLAHE Parameter Helper

Utility script to easily adjust CLAHE parameters at runtime without RQT.

Usage:
    ros2 run qcar2_object_detections dynamic_clahe_helper.py --clip-limit 4.0
    ros2 run qcar2_object_detections dynamic_clahe_helper.py --tile-size 32
    ros2 run qcar2_object_detections dynamic_clahe_helper.py --brightness -10
    ros2 run qcar2_object_detections dynamic_clahe_helper.py --enable true
"""

import argparse
import sys
import rclpy
from rclpy.node import Node


class CLAHEHelper(Node):
    def __init__(self, args):
        super().__init__('clahe_helper')
        self.args = args
        self.process_arguments()

    def process_arguments(self):
        """Process command line arguments and set parameters."""
        # Set parameters if provided
        if self.args.clip_limit is not None:
            value = float(self.args.clip_limit)
            if 1.0 <= value <= 8.0:
                self.set_parameters([
                    rclpy.parameter.Parameter(
                        'image_preprocessor_node.clahe_clip_limit',
                        rclpy.Parameter.Type.DOUBLE,
                        value
                    )
                ])
                self.get_logger().info(f"Set clip_limit to {value}")
            else:
                self.get_logger().error(f"clip_limit must be between 1.0 and 8.0")

        if self.args.tile_size is not None:
            value = int(self.args.tile_size)
            if value in [4, 8, 16, 32, 64]:
                self.set_parameters([
                    rclpy.parameter.Parameter(
                        'image_preprocessor_node.clahe_tile_size',
                        rclpy.Parameter.Type.INTEGER,
                        value
                    )
                ])
                self.get_logger().info(f"Set tile_size to {value}x{value}")
            else:
                self.get_logger().error(f"tile_size must be one of: 4, 8, 16, 32, 64")

        if self.args.brightness is not None:
            value = float(self.args.brightness)
            if -50.0 <= value <= 50.0:
                self.set_parameters([
                    rclpy.parameter.Parameter(
                        'image_preprocessor_node.brightness_adjustment',
                        rclpy.Parameter.Type.DOUBLE,
                        value
                    )
                ])
                self.get_logger().info(f"Set brightness_adjustment to {value}")
            else:
                self.get_logger().error(f"brightness_adjustment must be between -50 and 50")

        if self.args.enable is not None:
            value = self.args.enable.lower() in ['true', '1', 'yes', 'on']
            self.set_parameters([
                rclpy.parameter.Parameter(
                    'image_preprocessor_node.clahe_enabled',
                    rclpy.Parameter.Type.BOOL,
                    value
                )
            ])
            self.get_logger().info(f"Set clahe_enabled to {value}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Dynamic CLAHE parameter adjustment helper'
    )
    parser.add_argument(
        '--clip-limit', type=float, default=None,
        help='Set CLAHE clip limit (1.0-8.0, default: 3.5)'
    )
    parser.add_argument(
        '--tile-size', type=int, default=None,
        help='Set CLAHE tile size (4, 8, 16, 32, 64, default: 16)'
    )
    parser.add_argument(
        '--brightness', type=float, default=None,
        help='Set brightness adjustment (-50 to 50, default: 0)'
    )
    parser.add_argument(
        '--enable', type=str, default=None,
        help='Enable/disable CLAHE (true/false)'
    )

    parsed_args = parser.parse_args(args[1:] if args else None)

    rclpy.init(args=args)
    helper = CLAHEHelper(parsed_args)
    
    try:
        rclpy.spin_once(helper, timeout_sec=1.0)
    finally:
        helper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
