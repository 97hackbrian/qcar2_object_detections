import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2
import numpy as np


# detect red and green

class TrafficLightDetector(Node):
    def __init__(self):
        super().__init__('traffic_light_detector')
        self.bridge = CvBridge()
        self.current_image = None
        
        # ID 9 suele ser 'traffic light' en modelos COCO
        self.TRAFFIC_LIGHT_CLASS_ID = "9" 

        self.img_sub = self.create_subscription(Image, '/image', self.image_callback, 10)
        self.det_sub = self.create_subscription(Detection2DArray, '/detections_output', self.detection_callback, 10)
        self.get_logger().info('--- Nodo detector iniciado')

    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def detection_callback(self, msg):
        if self.current_image is None: return
        
        for det in msg.detections:
            try:
                # Filtrar solo semáforos
                if str(det.results[0].hypothesis.class_id) != self.TRAFFIC_LIGHT_CLASS_ID:
                    continue

                # Geometría del Bounding Box
                cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
                sx, sy = det.bbox.size_x, det.bbox.size_y
                
                x1, y1 = int(cx - sx/2), int(cy - sy/2)
                x2, y2 = int(cx + sx/2), int(cy + sy/2)

                h_img, w_img, _ = self.current_image.shape
                roi = self.current_image[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]
                
                if roi.size == 0: continue
                
                estado = self.analyze_no_sections(roi)
                self.get_logger().info(f'SEMÁFORO DETECTADO: {estado}')

            except Exception as e:
                self.get_logger().error(f'Error en procesamiento: {e}')

    def analyze_no_sections(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # DEFINICIÓN DE UMBRALES 
        lower_red = np.array([0, 160, 160])
        upper_red = np.array([10, 255, 255])
        
        lower_yellow_kill = np.array([20, 30, 240])
        upper_yellow_kill = np.array([45, 160, 255])
        
        lower_green = np.array([46, 140, 140])
        upper_green = np.array([90, 255, 255])

        # EXCLUSIÓN 
        mask_yellow = cv2.inRange(hsv, lower_yellow_kill, upper_yellow_kill)
        
        # roi_clean, amarillo convertido a negro
        roi_clean = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(mask_yellow))

        # DETECCIÓN DE ROJO Y VERDE 
        mask_r = cv2.inRange(roi_clean, lower_red, upper_red)
        mask_g = cv2.inRange(roi_clean, lower_green, upper_green)

        r_px = cv2.countNonZero(mask_r)
        g_px = cv2.countNonZero(mask_g)

        umbral_sensibilidad = 8

        # LÓGICA DE DECISIÓN 
        if r_px > umbral_sensibilidad and r_px > g_px:
            return "\033[91m● ROJO\033[0m"
        elif g_px > umbral_sensibilidad and g_px > r_px:
            return "\033[92m● VERDE\033[0m"
        
        # Si el amarillo fue dominante o no hay luz clara, se omite
        return "AMARILLO O APAGADO (OMITIDO)"

def main():
    rclpy.init()
    node = TrafficLightDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
