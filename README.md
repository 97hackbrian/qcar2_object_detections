# QCar2 Object Detections

Paquete ROS2 Humble para detección de objetos usando Isaac ROS 2.1 con YoloV8.

## Características

- **Preprocesamiento de imagen**: Resize con padding (letterbox) para compatibilidad con YoloV8
- **Detección de personas**: Filtrado por ROI con publicación de mayor confianza
- **Detección de semáforos**: Análisis de color HSV (rojo, verde, amarillo)
- **Detección de señales STOP**: Filtrado por ROI con publicación de mayor confianza
- **Detección de paso de cebra**: Procesamiento de imagen basado en patrones de franjas

## Estructura del Paquete

```
qcar2_object_detections/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── setup.cfg
├── .gitignore
├── README.md
├── launch/
│   └── qcar2_detections.launch.py
├── config/
│   └── detections_params.yaml
├── qcar2_object_detections/
│   ├── __init__.py
│   ├── image_preprocessor_node.py
│   └── detection_filter_node.py
├── msg/
│   ├── PersonDetection.msg
│   ├── TrafficLightDetection.msg
│   ├── StopSignDetection.msg
│   └── ZebraCrossingDetection.msg
└── resource/
    └── qcar2_object_detections
```

## Dependencias

- ROS2 Humble
- Isaac ROS 2.1
  - `isaac_ros_yolov8`
  - `isaac_ros_tensor_rt`
  - `isaac_ros_dnn_image_encoder`
- OpenCV
- NumPy
- cv_bridge

## Instalación

### 1. Clonar en workspace

```bash
cd ~/ros2_ws/src
# El paquete ya debe estar en qcar2_object_detections/
```

### 2. Compilar

```bash
cd ~/ros2_ws
colcon build --packages-select qcar2_object_detections
source install/setup.bash
```

### 3. Preparar modelo YoloV8

```bash
# Copiar modelo ONNX a /tmp (o ruta preferida)
cp /path/to/yolov8s.onnx /tmp/yolov8s.onnx
```

## Uso

### Ejecución Básica

```bash
source /workspaces/isaac_ros-dev/ros2/install/setup.bash

ros2 launch qcar2_object_detections qcar2_detections.launch.py
```

### Con Parámetros Personalizados

```bash
ros2 launch qcar2_object_detections qcar2_detections.launch.py \
  input_image_topic:=/camera/color_image \
  model_file_path:=/tmp/yolov8s.onnx \
  engine_file_path:=/tmp/yolov8s.plan \
  confidence_threshold:=0.75 \
  min_confidence:=0.5
```

### Con Archivo de Configuración

```bash
ros2 launch qcar2_object_detections qcar2_detections.launch.py \
  --params-file $(ros2 pkg prefix qcar2_object_detections)/share/qcar2_object_detections/config/detections_params.yaml
```

## Tópicos

### Entrada

| Tópico | Tipo | Descripción |
|--------|------|-------------|
| `/camera/color_image` | `sensor_msgs/Image` | Imagen de entrada (configurable) |
| `/camera/csi_image_3` | `sensor_msgs/Image` | Imagen para detección de cebra |

### Salida

| Tópico | Tipo | Descripción |
|--------|------|-------------|
| `/image` | `sensor_msgs/Image` | Imagen preprocesada |
| `/detections/person` | `PersonDetection` | Detección de persona |
| `/detections/traffic_light` | `TrafficLightDetection` | Detección de semáforo |
| `/detections/stop_sign` | `StopSignDetection` | Detección de señal STOP |
| `/detections/zebra_crossing` | `ZebraCrossingDetection` | Detección de paso de cebra |

## Mensajes Personalizados

### PersonDetection.msg

```
std_msgs/Header header
bool detected                 # Persona detectada en ROI
float32 confidence            # Confianza [0.0-1.0]
```

### TrafficLightDetection.msg

```
std_msgs/Header header
bool detected                 # Semáforo detectado en ROI
string state                  # "red", "green", "yellow", "unknown"
float32 confidence            # Confianza [0.0-1.0]
```

### StopSignDetection.msg

```
std_msgs/Header header
bool detected                 # Señal STOP detectada en ROI
float32 confidence            # Confianza [0.0-1.0]
```

### ZebraCrossingDetection.msg

```
std_msgs/Header header
bool detected                 # Paso de cebra detectado
int32 stripe_count            # Número estimado de franjas
```

## Parámetros Configurables

### Image Preprocessor Node

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `input_image_topic` | string | `/camera/color_image` | Tópico de imagen entrada |
| `output_image_topic` | string | `/image` | Tópico de imagen salida |
| `input_width` | int | 640 | Ancho imagen entrada |
| `input_height` | int | 480 | Alto imagen entrada |
| `target_width` | int | 640 | Ancho objetivo |
| `target_height` | int | 640 | Alto objetivo |
| `padding_color` | int[3] | [0,0,0] | Color de padding RGB |

### Detection Filter Node

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `person_roi` | int[4] | [0,0,640,640] | ROI persona [x1,y1,x2,y2] |
| `traffic_light_roi` | int[4] | [0,0,640,640] | ROI semáforo |
| `stop_sign_roi` | int[4] | [0,0,640,640] | ROI stop sign |
| `min_confidence` | float | 0.5 | Confianza mínima |
| `zebra_enabled` | bool | true | Habilitar detección cebra |
| `zebra_roi_top` | float | 0.80 | ROI cebra inicio (%) |
| `zebra_roi_bottom` | float | 0.985 | ROI cebra fin (%) |
| `zebra_roi_width` | float | 0.40 | ROI cebra ancho (%) |

## Visualización

### Con RQT Image View

```bash
ros2 run rqt_image_view rqt_image_view /image
```

### Con RViz2

Agregar displays para:
- Image: `/image`
- Topic: `/detections/person`, `/detections/traffic_light`, etc.

### Monitorear Tópicos

```bash
# Ver detecciones de persona
ros2 topic echo /detections/person

# Ver detecciones de semáforo
ros2 topic echo /detections/traffic_light

# Ver detecciones de stop sign
ros2 topic echo /detections/stop_sign

# Ver detecciones de cebra
ros2 topic echo /detections/zebra_crossing
```

## IDs de Clase COCO

| Clase | ID |
|-------|-----|
| person | 0 |
| traffic light | 9 |
| stop sign | 11 |

Referencia completa: [COCO Dataset Classes](https://docs.ultralytics.com/datasets/detect/coco/)

## Notas

- El motor TensorRT (`.plan`) se genera automáticamente la primera vez
- La primera ejecución puede tardar 1-2 minutos mientras se compila el modelo
- Si no aparecen detecciones, reintentar el launch
- No se usan ventanas GUI (cv2.imshow) - todo es mediante tópicos ROS2

## Licencia

MIT License

