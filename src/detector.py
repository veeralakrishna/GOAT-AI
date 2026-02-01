from ultralytics import YOLO
import cv2
import numpy as np

class GoatDetector:
    def __init__(self, model_path="yolov8x-seg.pt", conf_thres=0.5, target_classes=None):
        """
        Initialize the YOLOv8 Segmentation Detector.
        """
        print(f"Loading model: {model_path}...")
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.target_classes = target_classes 
        # Default to sheep(18) if none provided. 
        if self.target_classes is None:
            self.target_classes = [18] 

    def detect(self, frame):
        """
        Perform detection and segmentation on a frame.
        Returns:
            results: YOLO results object
        """
        # Run inference
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            classes=self.target_classes,
            verbose=False,
            retina_masks=True # High quality masks
        )
        return results[0]
