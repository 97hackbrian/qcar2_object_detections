cmake_minimum_required(VERSION 3.8)
project(qcar2_object_detections)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rclpy REQUIRED)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PersonDetection.msg"
  "msg/TrafficLightDetection.msg"
  "msg/StopSignDetection.msg"
  "msg/ZebraCrossingDetection.msg"
  DEPENDENCIES std_msgs
)

# Export dependencies for downstream packages
ament_export_dependencies(rosidl_default_runtime)

# Install Python executables
install(PROGRAMS
  qcar2_object_detections/image_preprocessor_node.py
  qcar2_object_detections/detection_filter_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install resource files
install(FILES
  resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
