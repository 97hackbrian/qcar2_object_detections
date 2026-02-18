# TrafficLightDetection.msg
# Detection message for traffic light class with color state

std_msgs/Header header
bool detected                 # Traffic light detected within ROI
string state                  # Color state: "red", "green", "yellow", "unknown"
float32 confidence            # Confidence [0.0-1.0] of highest score detection
