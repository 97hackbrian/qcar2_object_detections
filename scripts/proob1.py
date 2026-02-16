import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from qcar2_interfaces.msg import MotorCommands
from cv_bridge import CvBridge

import cv2
import numpy as np
from ultralytics import YOLO
from time import time


class QCar2VisionSafetyTeleop(Node):
    """
    Un SOLO nodo que hace:
      1) Lee cámara (/camera/color_image) y detecta personas con YOLOv8.
      2) Lee LiDAR (/scan) y calcula distancia mínima frontal (aprox. mínimo válido).
      3) Permite teleop por teclado (W/A/S/D, X stop, +/- cambia sensibilidad).
      4) Aplica lógica de seguridad con 3 estados: GO / SLOW / STOP.
      5) Publica comandos a los motores en /qcar2_motor_speed_cmd (MotorCommands).
    """

    def __init__(self):
        super().__init__('qcar2_vision_safety_teleop')

        # --- ROS I/O ---
        self.bridge = CvBridge()
        self.sub_cam = self.create_subscription(Image, '/camera/color_image', self.image_cb, 10)
        self.sub_lidar = self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.pub_motor = self.create_publisher(MotorCommands, '/qcar2_motor_speed_cmd', 10)

        # --- YOLO ---
        self.model = YOLO('yolov8n.pt')

        # --- Estado sensores ---
        self.distancia_lidar = 0.0  # distancia mínima válida (m)

        # --- Teleop (teclado) ---
        self.throttle = 0.0
        self.steering = 0.0
        self.step = 0.08            # incremento por tecla
        self.soft_brake = 0.02      # retorno a 0 si no se presionan teclas
        self.soft_center = 0.01     # retorno del volante hacia 0 si no se presionan teclas
        self.max_throttle = 0.6
        self.max_steer = 0.5

        # --- Seguridad (tuning) ---
        self.yolo_conf_min = 0.50
        self.min_bbox_w = 30
        self.min_bbox_h = 120

        self.stop_dist = 0.50       # STOP si LiDAR < 0.5m
        self.slow_dist = 1.00       # SLOW si LiDAR < 1.0m

        # Zona de riesgo horizontal (persona "en mi carril" aprox)
        self.risk_left = 0.40       # 40% del ancho
        self.risk_right = 0.60      # 60% del ancho

        # Filtro temporal (evita parpadeo)
        self.frames_peligro = 0
        self.frames_libre = 0
        self.stop_hold_until = 0.0  # mantiene STOP por un tiempo mínimo

        # Estado de decisión
        self.estado = "GO"          # GO / SLOW / STOP

        self.get_logger().info(
            "Nodo único listo. Controles:\n"
            "  W/S: acelerar/frenar (throttle)\n"
            "  A/D: girar (steering)\n"
            "  X  : stop inmediato\n"
            "  +/-: cambia sensibilidad (step)\n"
            "Seguridad: GO/SLOW/STOP con YOLO + LiDAR (STOP con hold anti-parpadeo)."
        )

    # ----------------- Callbacks -----------------

    def lidar_cb(self, msg: LaserScan):
        # Mínimo válido: filtra ceros/infinito y valores muy pequeños
        rangos = np.array(msg.ranges, dtype=np.float32)
        validos = rangos[(rangos > 0.2) & (rangos < msg.range_max)]
        self.distancia_lidar = float(np.min(validos)) if validos.size > 0 else 0.0

    def image_cb(self, msg: Image):
        # 1) ROS Image -> OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if frame is None:
                return
        except Exception:
            return

        h, w = frame.shape[:2]
        now = time()

        # 2) YOLO: detecta si hay "persona peligrosa" (por confianza + tamaño + zona central)
        peligro_vision, best_box = self.detectar_persona_peligrosa(frame, w, h)

        # 3) LiDAR: define riesgo por distancia
        peligro_lidar_stop = (0.1 < self.distancia_lidar < self.stop_dist)
        peligro_lidar_slow = (self.stop_dist <= self.distancia_lidar < self.slow_dist)

        # 4) Filtro temporal + estado GO/SLOW/STOP
        self.actualizar_estado(peligro_vision, peligro_lidar_stop, peligro_lidar_slow, now)

        # 5) Teclado (una sola lectura por frame)
        self.leer_teclado_y_actualizar_teleop()

        # 6) Gate de seguridad: aplica límites/acciones según estado
        self.aplicar_seguridad_a_comandos()

        # 7) Publica comando a motores
        self.publicar_motor()

        # 8) Overlay + ventana (debug)
        self.dibujar_debug(frame, best_box, w, h)

    # ----------------- Percepción -----------------

    def detectar_persona_peligrosa(self, frame, w, h):
        """
        Retorna (peligro_vision, best_box).
        best_box = (x1,y1,x2,y2,conf) de la persona más relevante si existe.
        """
        results = self.model(frame, verbose=False)

        peligro_vision = False
        best_box = None
        best_score = -1.0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != 0:  # COCO: 0 = person
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw, bh = (x2 - x1), (y2 - y1)
                cx = (x1 + x2) / 2.0

                if conf < self.yolo_conf_min:
                    continue
                if bw < self.min_bbox_w or bh < self.min_bbox_h:
                    continue

                # Zona de riesgo: centro de la bbox en el centro de la imagen
                in_risk_zone = (self.risk_left * w) <= cx <= (self.risk_right * w)

                # Para elegir "la mejor" bbox: tamaño * confianza (simple)
                score = conf * (bw * bh)

                if score > best_score:
                    best_score = score
                    best_box = (x1, y1, x2, y2, conf, in_risk_zone)

                if in_risk_zone:
                    peligro_vision = True

        return peligro_vision, best_box

    # ----------------- Decisión -----------------

    def actualizar_estado(self, peligro_vision, peligro_lidar_stop, peligro_lidar_slow, now):
        """
        GO / SLOW / STOP:
          - STOP si LiDAR < stop_dist o persona peligrosa por N frames.
          - SLOW si LiDAR < slow_dist o persona peligrosa (sin llegar a STOP).
          - HOLD: cuando entra STOP, se mantiene un tiempo mínimo para evitar parpadeos.
        """
        peligro = peligro_vision or peligro_lidar_stop or peligro_lidar_slow

        if peligro:
            self.frames_peligro += 1
            self.frames_libre = 0
        else:
            self.frames_libre += 1
            self.frames_peligro = 0

        # HOLD: si todavía estamos dentro de la ventana de STOP, manten STOP
        if now < self.stop_hold_until:
            self.estado = "STOP"
            return

        # STOP fuerte por LiDAR o por visión persistente
        if peligro_lidar_stop or self.frames_peligro >= 3:
            self.estado = "STOP"
            self.stop_hold_until = now + 1.0  # mantiene STOP 1 segundo mínimo
            return

        # SLOW si visión o LiDAR cerca
        if peligro_vision or peligro_lidar_slow:
            self.estado = "SLOW"
        else:
            self.estado = "GO"

    # ----------------- Teclado / Teleop -----------------

    def leer_teclado_y_actualizar_teleop(self):
        """
        Lee una tecla por frame usando OpenCV:
        W/S: throttle +/- step
        A/D: steering +/- step
        X  : stop inmediato
        +/-: cambia step
        Si no se presiona nada: freno suave + centrado suave
        """
        key = cv2.waitKey(1) & 0xFF
        hubo_tecla = (key != 255)

        if key == ord('w'):
            self.throttle += self.step
        elif key == ord('s'):
            self.throttle -= self.step
        elif key == ord('a'):
            self.steering += self.step
        elif key == ord('d'):
            self.steering -= self.step
        elif key == ord('x'):
            self.throttle = 0.0
            self.steering = 0.0
        elif key in (ord('+'), ord('=')):
            self.step = min(self.step + 0.01, 0.20)
            self.get_logger().info(f"step = {self.step:.2f}")
        elif key in (ord('-'), ord('_')):
            self.step = max(self.step - 0.01, 0.01)
            self.get_logger().info(f"step = {self.step:.2f}")

        # si no presionas teclas, vuelve suavemente a 0 (más controlable)
        if not hubo_tecla:
            # throttle -> 0
            if self.throttle > 0.0:
                self.throttle = max(0.0, self.throttle - self.soft_brake)
            elif self.throttle < 0.0:
                self.throttle = min(0.0, self.throttle + self.soft_brake)

            # steering -> 0
            if self.steering > 0.0:
                self.steering = max(0.0, self.steering - self.soft_center)
            elif self.steering < 0.0:
                self.steering = min(0.0, self.steering + self.soft_center)

        # límites
        self.throttle = float(np.clip(self.throttle, -self.max_throttle, self.max_throttle))
        self.steering = float(np.clip(self.steering, -self.max_steer, self.max_steer))

    # ----------------- Safety gate -----------------

    def aplicar_seguridad_a_comandos(self):
        """
        Convierte el estado en acciones sobre los motores:
          - STOP: throttle = 0; steering se reduce hacia 0.
          - SLOW: limita throttle máximo.
          - GO  : teleop normal.
        """
        if self.estado == "STOP":
            self.throttle = 0.0
            self.steering *= 0.5  # centra un poco (suave)
        elif self.estado == "SLOW":
            max_slow = 0.15
            self.throttle = float(np.clip(self.throttle, -max_slow, max_slow))

    # ----------------- Actuación -----------------

    def publicar_motor(self):
        msg = MotorCommands()
        msg.motor_names = ["steering_angle", "motor_throttle"]
        msg.values = [float(self.steering), float(self.throttle)]
        self.pub_motor.publish(msg)

    def parar_motor(self):
        msg = MotorCommands()
        msg.motor_names = ["steering_angle", "motor_throttle"]
        msg.values = [0.0, 0.0]
        self.pub_motor.publish(msg)

    # ----------------- Debug UI -----------------

    def dibujar_debug(self, frame, best_box, w, h):
        # Dibuja zona de riesgo (vertical)
        xL = int(self.risk_left * w)
        xR = int(self.risk_right * w)
        cv2.line(frame, (xL, 0), (xL, h), (255, 255, 0), 2)
        cv2.line(frame, (xR, 0), (xR, h), (255, 255, 0), 2)

        # Si hay bbox, dibuja
        if best_box is not None:
            x1, y1, x2, y2, conf, in_zone = best_box
            color = (0, 0, 255) if in_zone else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"person {conf:.2f} zone={in_zone}",
                        (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Texto estado
        cv2.putText(frame,
                    f"STATE={self.estado} | lidar_min={self.distancia_lidar:.2f}m | thr={self.throttle:.2f} steer={self.steering:.2f} step={self.step:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("QCar2 Vision+Safety+Teleop", frame)


def main():
    rclpy.init()
    node = QCar2VisionSafetyTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.parar_motor()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
