import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "video")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Model Settings
MODEL_NAME = "yolov8n-seg.pt"  # Switching to Nano for faster testing/download
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# COCO Classes (YOLOv8 default)
# 18: sheep (often works for goats)
# 19: cow
# 21: bear
# 22: zebra
# 23: giraffe
TARGET_CLASSES = [18, 19] # Detecting Sheep and Cow as proxies for Goat

# Measurement Calibrations
# This is a placeholder. In a real scenario, we need a reference object or depth.
# For now, we assume a standard distance where 100 pixels approx = 10 cm (Example)
# This SHOULD be calibrated per camera/setup.
PIXELS_PER_CM = 10.0 
