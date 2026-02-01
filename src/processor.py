import cv2
import time
import os
import numpy as np
from src.detector import GoatDetector
from src.measurements import BiometricEstimator
import config

class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        
        # Initialize components
        self.detector = GoatDetector(
            model_path=config.MODEL_NAME, 
            conf_thres=config.CONFIDENCE_THRESHOLD,
            target_classes=config.TARGET_CLASSES
        )
        self.estimator = BiometricEstimator(pixels_per_cm=config.PIXELS_PER_CM)
        
    def process(self):
        print(f"Opening video: {self.input_path}")
        cap = cv2.VideoCapture(self.input_path)
        
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Output writer
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 1. Detection
            results = self.detector.detect(frame)
            
            # 2. Visualization & Measurement
            annotated_frame = frame.copy()
            
            if results.masks:
                # We have segmentations
                masks = results.masks.xy # List of polygon arrays
                
                for i, mask in enumerate(masks):
                    if len(mask) == 0: continue
                    
                    # Convert to integer countour
                    contour = np.array(mask, dtype=np.int32)
                    
                    # 3. Measurement
                    metrics = self.estimator.estimate_dimensions(contour)
                    
                    # 4. Draw
                    # Draw mask contour
                    cv2.drawContours(annotated_frame, [contour], -1, (0, 255, 0), 2)
                    
                    # Draw bounding box (rotated)
                    box = metrics['rect_points']
                    cv2.drawContours(annotated_frame, [box], 0, (0, 0, 255), 2)
                    
                    # Draw Text
                    label_pos = box[1] # Use one of the corners
                    text = f"L: {metrics['length_cm']}cm | H: {metrics['width_cm']}cm"
                    cv2.putText(annotated_frame, text, (label_pos[0], label_pos[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                
            # Write frame
            out.write(annotated_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")

        # Cleanup
        cap.release()
        out.release()
        end_time = time.time()
        duration = end_time - start_time
        print(f"Processing complete. Saved to {self.output_path}")
        print(f"Total time: {duration:.2f}s, FPS: {frame_count/duration:.2f}")

