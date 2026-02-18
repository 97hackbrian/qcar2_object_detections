# ZebraCrossingDetection.msg
# Detection message for zebra crossing (crosswalk) using image processing

std_msgs/Header header
bool detected                 # Zebra crossing detected
int32 stripe_count            # Estimated number of stripes detected
