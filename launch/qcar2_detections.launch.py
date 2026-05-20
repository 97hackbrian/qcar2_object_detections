#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from ament_index_python.packages import get_package_share_directory

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_qcar2 = get_package_share_directory('qcar2_object_detections')

    # YAML por defecto dentro del paquete
    default_params_file = os.path.join(pkg_qcar2, 'config', 'detections_params.yaml')

    # ---------------------------
    # Launch arguments
    # ---------------------------
    launch_args = [
        DeclareLaunchArgument('params_file', default_value=default_params_file,
                              description='Full parameter YAML file'),

        DeclareLaunchArgument('preprocessed_image_topic', default_value='/image'),

        DeclareLaunchArgument('model_file_path', default_value='/tmp/yolov8s.onnx'),
        DeclareLaunchArgument('engine_file_path', default_value='/tmp/yolov8s.plan'),
        DeclareLaunchArgument('input_binding_names', default_value="['images']"),
        DeclareLaunchArgument('output_binding_names', default_value="['output0']"),
        DeclareLaunchArgument('network_image_width', default_value='640'),
        DeclareLaunchArgument('network_image_height', default_value='640'),
        DeclareLaunchArgument('input_image_width', default_value='640'),
        DeclareLaunchArgument('input_image_height', default_value='640'),
        DeclareLaunchArgument('force_engine_update', default_value='False'),
        DeclareLaunchArgument('image_mean', default_value='[0.0, 0.0, 0.0]'),
        DeclareLaunchArgument('image_stddev', default_value='[1.0, 1.0, 1.0]'),
        DeclareLaunchArgument('nms_threshold', default_value='0.01'),

    ]

    # ---------------------------
    # LaunchConfigurations
    # ---------------------------
    params_file = LaunchConfiguration('params_file')

    preprocessed_image_topic = LaunchConfiguration('preprocessed_image_topic')

    model_file_path = LaunchConfiguration('model_file_path')
    engine_file_path = LaunchConfiguration('engine_file_path')
    input_binding_names = LaunchConfiguration('input_binding_names')
    output_binding_names = LaunchConfiguration('output_binding_names')
    network_image_width = LaunchConfiguration('network_image_width')
    network_image_height = LaunchConfiguration('network_image_height')
    input_image_width = LaunchConfiguration('input_image_width')
    input_image_height = LaunchConfiguration('input_image_height')
    force_engine_update = LaunchConfiguration('force_engine_update')
    image_mean = LaunchConfiguration('image_mean')
    image_stddev = LaunchConfiguration('image_stddev')
    nms_threshold = LaunchConfiguration('nms_threshold')

    def launch_setup(context, *args, **kwargs):
        params_path = LaunchConfiguration('params_file').perform(context)
        with open(params_path, 'r', encoding='utf-8') as f:
            params_yaml = yaml.safe_load(f) or {}

        confidence_threshold_value = 0.75
        preproc_params = params_yaml.get('image_preprocessor_node', {}).get('ros__parameters', {})
        if 'yolo_confidence_threshold' in preproc_params:
            confidence_threshold_value = preproc_params.get('yolo_confidence_threshold', 0.75)
        else:
            confidence_threshold_value = params_yaml.get('yolo_confidence_threshold', 0.75)

        # ---------------------------
        # Nodes
        # ---------------------------
        image_preprocessor_node = Node(
            package='qcar2_object_detections',
            executable='image_preprocessor_node.py',
            name='image_preprocessor_node',
            output='screen',
            parameters=[params_file]
        )

        yolov8_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('isaac_ros_yolov8'),
                '/launch/isaac_ros_yolov8_visualize.launch.py'
            ]),
            launch_arguments={
                'model_file_path': model_file_path,
                'engine_file_path': engine_file_path,
                'input_binding_names': input_binding_names,
                'output_binding_names': output_binding_names,
                'network_image_width': network_image_width,
                'network_image_height': network_image_height,
                'input_image_width': input_image_width,
                'input_image_height': input_image_height,
                'image_name': preprocessed_image_topic,
                'force_engine_update': force_engine_update,
                'image_mean': image_mean,
                'image_stddev': image_stddev,
                'confidence_threshold': str(confidence_threshold_value),
                'nms_threshold': nms_threshold,
                'bounding_box_scale': '1.0',
                'setup_image_viewer': 'False',
            }.items()
        )

        detection_filter_node = Node(
            package='qcar2_object_detections',
            executable='detection_filter_node.py',
            name='detection_filter_node',
            output='screen',
            parameters=[params_file]
        )

        image_compressor_node = Node(
            package='qcar2_object_detections',
            executable='image_compressor_node.py',
            name='image_compressor_node',
            output='screen',
            parameters=[params_file]
        )

        return [
            image_preprocessor_node,
            yolov8_launch,
            detection_filter_node,
            image_compressor_node,
        ]

    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])