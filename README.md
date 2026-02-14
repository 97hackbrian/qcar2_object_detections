## Flujo de ejecución

# 1. Resize / padding de la cámara (640×480 → 640×640)
python3 rqt_fixer.py

# 2. Visualización de la cámara en RQT
ros2 run rqt_image_view rqt_image_view

# 3. Ejecutar YoloV8
1. cp /workspaces/isaac_ros-dev/yolov8s.onnx /tmp/yolov8s.onnx

2. source /workspaces/isaac_ros-dev/ros2/install/setup.bash

3. ros2 launch isaac_ros_yolov8 isaac_ros_yolov8_visualize.launch.py \
  model_file_path:=/tmp/yolov8s.onnx \
  engine_file_path:=/tmp/yolov8s.plan \
  input_binding_names:="['images']" \
  output_binding_names:="['output0']" \
  network_image_width:=640 \
  network_image_height:=640 \
  input_image_width:=640 \
  input_image_height:=640 \
  image_name:=/image \
  force_engine_update:=False \
  image_mean:="[0.0,0.0,0.0]" \
  image_stddev:="[1.0,1.0,1.0]" \
  confidence_threshold:=0.15 \
  nms_threshold:=0.45 \
  bounding_box_scale:=1.0 \
  setup_image_viewer:=False

Nota: si no aparece la detección después de 1–2 minutos, detén el launch y ejecútalo nuevamente

# 4. Ejecutar scripts adicionales
python3 color_rv.py

python3 persona.py

python3 s_cebra.py

