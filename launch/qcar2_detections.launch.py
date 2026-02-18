#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCar2 Object Detections Launch File

This launch file starts:
1. Image Preprocessor Node (resize with padding)
2. Isaac ROS YoloV8 TensorRT inference pipeline
3. Detection Filter Node (ROI filtering + traffic light color + zebra)

Usage:
    ros2 launch qcar2_object_detections qcar2_detections.launch.py

With custom parameters:
    ros2 launch qcar2_object_detections qcar2_detections.launch.py \
        input_image_topic:=/camera/color_image \
        model_file_path:=/tmp/yolov8s.onnx

Author: QCar2 Developer
License: MIT
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    """Generate launch description for QCar2 Object Detections."""
    
    # Package directories
    pkg_qcar2_detections = get_package_share_directory('qcar2_object_detections')
    
    # =========================================================================
    # LAUNCH ARGUMENTS
    # =========================================================================
    
    launch_args = [
        # =====================================================================
        # Preprocessor Node Arguments
        # =====================================================================
        DeclareLaunchArgument(
            'input_image_topic',
            default_value='/camera/color_image',
            description='Input image topic for preprocessing'
        ),
        DeclareLaunchArgument(
            'preprocessed_image_topic',
            default_value='/image',
            description='Output preprocessed image topic (input for YoloV8)'
        ),
        DeclareLaunchArgument(
            'input_width',
            default_value='640',
            description='Input image width'
        ),
        DeclareLaunchArgument(
            'input_height',
            default_value='480',
            description='Input image height'
        ),
        DeclareLaunchArgument(
            'target_width',
            default_value='640',
            description='Target image width for YoloV8'
        ),
        DeclareLaunchArgument(
            'target_height',
            default_value='640',
            description='Target image height for YoloV8'
        ),
        
        # =====================================================================
        # YoloV8 TensorRT Arguments
        # =====================================================================
        DeclareLaunchArgument(
            'model_file_path',
            default_value='/tmp/yolov8s.onnx',
            description='Path to YoloV8 ONNX model file'
        ),
        DeclareLaunchArgument(
            'engine_file_path',
            default_value='/tmp/yolov8s.plan',
            description='Path to TensorRT engine file'
        ),
        DeclareLaunchArgument(
            'input_binding_names',
            default_value="['images']",
            description='Input tensor binding names'
        ),
        DeclareLaunchArgument(
            'output_binding_names',
            default_value="['output0']",
            description='Output tensor binding names'
        ),
        DeclareLaunchArgument(
            'network_image_width',
            default_value='640',
            description='Network input width'
        ),
        DeclareLaunchArgument(
            'network_image_height',
            default_value='640',
            description='Network input height'
        ),
        DeclareLaunchArgument(
            'force_engine_update',
            default_value='False',
            description='Force TensorRT engine rebuild'
        ),
        DeclareLaunchArgument(
            'image_mean',
            default_value='[0.0, 0.0, 0.0]',
            description='Image normalization mean'
        ),
        DeclareLaunchArgument(
            'image_stddev',
            default_value='[1.0, 1.0, 1.0]',
            description='Image normalization stddev'
        ),
        DeclareLaunchArgument(
            'confidence_threshold',
            default_value='0.75',
            description='YoloV8 confidence threshold'
        ),
        DeclareLaunchArgument(
            'nms_threshold',
            default_value='0.45',
            description='YoloV8 NMS threshold'
        ),
        
        # =====================================================================
        # Detection Filter Node Arguments
        # =====================================================================
        DeclareLaunchArgument(
            'detections_input_topic',
            default_value='/detections_output',
            description='YoloV8 detections input topic'
        ),
        DeclareLaunchArgument(
            'person_output_topic',
            default_value='/detections/person',
            description='Person detection output topic'
        ),
        DeclareLaunchArgument(
            'traffic_light_output_topic',
            default_value='/detections/traffic_light',
            description='Traffic light detection output topic'
        ),
        DeclareLaunchArgument(
            'stop_sign_output_topic',
            default_value='/detections/stop_sign',
            description='Stop sign detection output topic'
        ),
        DeclareLaunchArgument(
            'zebra_output_topic',
            default_value='/detections/zebra_crossing',
            description='Zebra crossing detection output topic'
        ),
        DeclareLaunchArgument(
            'zebra_image_topic',
            default_value='/camera/csi_image_3',
            description='Image topic for zebra crossing detection'
        ),
        DeclareLaunchArgument(
            'min_confidence',
            default_value='0.5',
            description='Minimum confidence threshold for detections'
        ),
        DeclareLaunchArgument(
            'person_roi',
            default_value='[0, 0, 640, 640]',
            description='Person detection ROI [x1, y1, x2, y2]'
        ),
        DeclareLaunchArgument(
            'traffic_light_roi',
            default_value='[0, 0, 640, 640]',
            description='Traffic light detection ROI [x1, y1, x2, y2]'
        ),
        DeclareLaunchArgument(
            'stop_sign_roi',
            default_value='[0, 0, 640, 640]',
            description='Stop sign detection ROI [x1, y1, x2, y2]'
        ),
        DeclareLaunchArgument(
            'zebra_enabled',
            default_value='True',
            description='Enable zebra crossing detection'
        ),
    ]
    
    # =========================================================================
    # LAUNCH CONFIGURATIONS
    # =========================================================================
    
    # Preprocessor
    input_image_topic = LaunchConfiguration('input_image_topic')
    preprocessed_image_topic = LaunchConfiguration('preprocessed_image_topic')
    input_width = LaunchConfiguration('input_width')
    input_height = LaunchConfiguration('input_height')
    target_width = LaunchConfiguration('target_width')
    target_height = LaunchConfiguration('target_height')
    
    # YoloV8
    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    input_binding_names = LaunchConfiguration('input_binding_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    force_engine_update = LaunchConfiguration('force_engine_update')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')
    
    # Detection Filter
    detections_input_topic = LaunchConfiguration('detections_input_topic')
    person_output_topic = LaunchConfiguration('person_output_topic')
    traffic_light_output_topic = LaunchConfiguration('traffic_light_output_topic')
    stop_sign_output_topic = LaunchConfiguration('stop_sign_output_topic')
    zebra_output_topic = LaunchConfiguration('zebra_output_topic')
    zebra_image_topic = LaunchConfiguration('zebra_image_topic')
    min_confidence = LaunchConfiguration('min_confidence')
    person_roi = LaunchConfiguration('person_roi')
    traffic_light_roi = LaunchConfiguration('traffic_light_roi')
    stop_sign_roi = LaunchConfiguration('stop_sign_roi')
    zebra_enabled = LaunchConfiguration('zebra_enabled')
    
    # =========================================================================
    # NODES
    # =========================================================================
    
    # Image Preprocessor Node
    image_preprocessor_node = Node(
        package='qcar2_object_detections',
        executable='image_preprocessor_node.py',
        name='image_preprocessor_node',
        output='screen',
        parameters=[{
            'input_image_topic': input_image_topic,
            'output_image_topic': preprocessed_image_topic,
            'input_width': input_width,
            'input_height': input_height,
            'target_width': target_width,
            'target_height': target_height,
            'padding_color': [0, 0, 0],
        }]
    )
    
    # YoloV8 TensorRT Container (Composable Nodes)
    encoder_node = ComposableNode(
        name='dnn_image_encoder',
        package='isaac_ros_dnn_image_encoder',
        plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
        remappings=[
            ('image', preprocessed_image_topic),
            ('encoded_tensor', 'tensor_pub')
        ],
        parameters=[{
            'input_image_width': network_image_width,
            'input_image_height': network_image_height,
            'network_image_width': network_image_width,
            'network_image_height': network_image_height,
            'image_mean': image_mean,
            'image_stddev': image_stddev,
        }]
    )
    
    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': model_file_path,
            'engine_file_path': engine_file_path,
            'output_binding_names': output_binding_names,
            'output_tensor_names': ['output_tensor'],
            'input_tensor_names': ['input_tensor'],
            'input_binding_names': input_binding_names,
            'verbose': False,
            'force_engine_update': force_engine_update,
        }]
    )
    
    yolov8_decoder_node = ComposableNode(
        name='yolov8_decoder_node',
        package='isaac_ros_yolov8',
        plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
        parameters=[{
            'confidence_threshold': confidence_threshold,
            'nms_threshold': nms_threshold,
        }]
    )
    
    tensor_rt_container = ComposableNodeContainer(
        name='tensor_rt_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            encoder_node,
            tensor_rt_node,
            yolov8_decoder_node
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', 'INFO'],
        namespace=''
    )
    
    # Detection Filter Node
    detection_filter_node = Node(
        package='qcar2_object_detections',
        executable='detection_filter_node.py',
        name='detection_filter_node',
        output='screen',
        parameters=[{
            'image_topic': preprocessed_image_topic,
            'detections_input_topic': detections_input_topic,
            'zebra_image_topic': zebra_image_topic,
            'person_output_topic': person_output_topic,
            'traffic_light_output_topic': traffic_light_output_topic,
            'stop_sign_output_topic': stop_sign_output_topic,
            'zebra_output_topic': zebra_output_topic,
            'person_class_id': '0',
            'traffic_light_class_id': '9',
            'stop_sign_class_id': '11',
            'person_roi': person_roi,
            'traffic_light_roi': traffic_light_roi,
            'stop_sign_roi': stop_sign_roi,
            'min_confidence': min_confidence,
            'zebra_enabled': zebra_enabled,
            'zebra_roi_top': 0.80,
            'zebra_roi_bottom': 0.985,
            'zebra_roi_width': 0.40,
            'zebra_min_stripes': 4,
            'zebra_max_stripes': 7,
            'zebra_vote_threshold': 5,
            'zebra_vote_window': 7,
        }]
    )
    
    # =========================================================================
    # RETURN LAUNCH DESCRIPTION
    # =========================================================================
    
    return LaunchDescription(
        launch_args + [
            image_preprocessor_node,
            tensor_rt_container,
            detection_filter_node,
        ]
    )
