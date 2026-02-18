# PersonDetection.msg
# Detection message for person class

std_msgs/Header header
bool detected                 # Person detected within ROI
float32 confidence            # Confidence [0.0-1.0] of highest score detection
