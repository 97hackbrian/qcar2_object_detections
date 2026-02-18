# StopSignDetection.msg
# Detection message for stop sign class

std_msgs/Header header
bool detected                 # Stop sign detected within ROI
float32 confidence            # Confidence [0.0-1.0] of highest score detection
