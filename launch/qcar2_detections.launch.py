#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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

        DeclareLaunchArgument('input_image_topic', default_value='/camera/color_image'),
        DeclareLaunchArgument('preprocessed_image_topic', default_value='/image'),
        DeclareLaunchArgument('input_width', default_value='640'),
        DeclareLaunchArgument('input_height', default_value='480'),
        DeclareLaunchArgument('target_width', default_value='640'),
        DeclareLaunchArgument('target_height', default_value='640'),

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
        DeclareLaunchArgument('confidence_threshold', default_value='0.50'),
        DeclareLaunchArgument('nms_threshold', default_value='0.45'),

        # Detections / filter arguments
        DeclareLaunchArgument('detections_input_topic', default_value='/detections_output'),
        DeclareLaunchArgument('zebra_image_topic', default_value='/camera/csi_image_3'),

        DeclareLaunchArgument('person_output_topic', default_value='/detections/person'),
        DeclareLaunchArgument('traffic_light_output_topic', default_value='/detections/traffic_light'),
        DeclareLaunchArgument('stop_sign_output_topic', default_value='/detections/stop_sign'),
        DeclareLaunchArgument('zebra_output_topic', default_value='/detections/zebra_crossing'),

        DeclareLaunchArgument('min_confidence', default_value='0.60'),
        DeclareLaunchArgument('zebra_enabled', default_value='True'),
    ]

    # ---------------------------
    # LaunchConfigurations
    # ---------------------------
    params_file = LaunchConfiguration('params_file')

    input_image_topic = LaunchConfiguration('input_image_topic')
    preprocessed_image_topic = LaunchConfiguration('preprocessed_image_topic')
    input_width = LaunchConfiguration('input_width')
    input_height = LaunchConfiguration('input_height')
    target_width = LaunchConfiguration('target_width')
    target_height = LaunchConfiguration('target_height')

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
    confidence_threshold = LaunchConfiguration('confidence_threshold')
    nms_threshold = LaunchConfiguration('nms_threshold')

    detections_input_topic = LaunchConfiguration('detections_input_topic')
    zebra_image_topic = LaunchConfiguration('zebra_image_topic')

    person_output_topic = LaunchConfiguration('person_output_topic')
    traffic_light_output_topic = LaunchConfiguration('traffic_light_output_topic')
    stop_sign_output_topic = LaunchConfiguration('stop_sign_output_topic')
    zebra_output_topic = LaunchConfiguration('zebra_output_topic')

    min_confidence = LaunchConfiguration('min_confidence')
    zebra_enabled = LaunchConfiguration('zebra_enabled')

    # ---------------------------
    # Nodes
    # ---------------------------
    image_preprocessor_node = Node(
        package='qcar2_object_detections',
        executable='image_preprocessor_node.py',
        name='image_preprocessor_node',
        output='screen',
        parameters=[
            params_file,
            {
                'input_image_topic': input_image_topic,
                'output_image_topic': preprocessed_image_topic,
                'input_width': input_width,
                'input_height': input_height,
                'target_width': target_width,
                'target_height': target_height,
                'padding_color': [0, 0, 0],
                'input_encoding': 'bgr8',
            }
        ]
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
            'confidence_threshold': confidence_threshold,
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
        parameters=[
            params_file,  #  carga TODO del YAML primero
            {
                # override mínimos / wiring
                'image_topic': preprocessed_image_topic,
                'detections_input_topic': detections_input_topic,
                'zebra_image_topic': zebra_image_topic,

                'person_output_topic': person_output_topic,
                'traffic_light_output_topic': traffic_light_output_topic,
                'stop_sign_output_topic': stop_sign_output_topic,
                'zebra_output_topic': zebra_output_topic,

                'min_confidence': min_confidence,
                'zebra_enabled': zebra_enabled,




                #  -------------- VISUALIZACION ------------
                # elegir with true or false
                'person_debug_view': False,
                'stop_sign_debug_view': False,
                'zebra_debug_view': True,
                'traffic_light_debug_view': True,



            }
        ]
    )

    return LaunchDescription(launch_args + [
        image_preprocessor_node,
        yolov8_launch,
        detection_filter_node,
    ])