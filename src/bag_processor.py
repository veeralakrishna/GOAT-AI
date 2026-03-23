import cv2
import time
import os
import numpy as np
from src.detector import GoatDetector
from src.measurements import BiometricEstimator
import config
from pyorbbecsdk import *

class BagProcessor:
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
        print(f"Opening bag file: {self.input_path}")
        try:
            playback = PlaybackDevice(self.input_path)
            pipeline = Pipeline(playback)
            cfg = Config()
            
            # Enable streams
            cfg.enable_stream(OBSensorType.COLOR_SENSOR)
            cfg.enable_stream(OBSensorType.DEPTH_SENSOR)
            
            # Use AlignFilter for Software Alignment (Depth to Color) 
            align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
            
            # Start pipeline
            pipeline.start(cfg)
        except Exception as e:
            print(f"Failed to open bag file {self.input_path}: {e}")
            return
            
        color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(0)
        intrinsics = color_profile.get_intrinsic()
        self.estimator.fx = intrinsics.fx
        self.estimator.fy = intrinsics.fy
        print(f"Camera Intrinsics - fx: {intrinsics.fx:.2f}, fy: {intrinsics.fy:.2f}")

        # Need width and height for video writer
        width = color_profile.get_width()
        height = color_profile.get_height()
        fps = color_profile.get_fps()
        
        # Output writer
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                frames = pipeline.wait_for_frames(100)
                if not frames:
                    continue
                
                # Align depth to color
                frames = align_filter.process(frames)
                if frames is None:
                    continue
                
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert color frame
                color_format = color_frame.get_format()
                c_height = color_frame.get_height()
                c_width = color_frame.get_width()
                color_data = np.asanyarray(color_frame.get_data())
                
                if color_format == OBFormat.MJPG:
                    color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
                elif color_format == OBFormat.RGB:
                    color_image = color_data.reshape((c_height, c_width, 3))
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                elif color_format == OBFormat.BGR:
                    color_image = color_data.reshape((c_height, c_width, 3))
                else:
                    print(f"Unsupported color format {color_format}")
                    continue

                # Get Depth image
                d_height = depth_frame.get_height()
                d_width = depth_frame.get_width()
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_image = depth_data.reshape((d_height, d_width))
                
                frame_count += 1
                
                # 1. Detection
                results = self.detector.detect(color_image)
                annotated_frame = color_image.copy()
                
                if results.masks:
                    masks = results.masks.xy # List of polygon arrays
                    for i, mask in enumerate(masks):
                        if len(mask) == 0: continue
                        
                        contour = np.array(mask, dtype=np.int32)
                        
                        # Find median depth within the segmentation mask
                        mask_img = np.zeros_like(depth_image, dtype=np.uint8)
                        cv2.fillPoly(mask_img, [contour], 255)
                        # Mask depth valid values
                        valid_depths = depth_image[(mask_img == 255) & (depth_image > 0)]
                        
                        median_depth_mm = np.median(valid_depths) if len(valid_depths) > 0 else 0
                        median_depth_cm = median_depth_mm / 10.0 if median_depth_mm > 0 else None
                        
                        # 3. Measurement
                        metrics = self.estimator.estimate_dimensions(contour, depth_cm=median_depth_cm)
                        
                        # 4. Draw
                        cv2.drawContours(annotated_frame, [contour], -1, (0, 255, 0), 2)
                        box = metrics['rect_points']
                        cv2.drawContours(annotated_frame, [box], 0, (0, 0, 255), 2)
                        
                        # Draw Text
                        label_pos = box[1] # Use one of the corners
                        text = f"L:{metrics['length_cm']}cm H:{metrics['width_cm']}cm C:{metrics['chest_girth_cm']}cm"
                        if median_depth_cm:
                            text += f" Z:{median_depth_cm:.1f}cm"
                        cv2.putText(annotated_frame, text, (label_pos[0], label_pos[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                # Write frame
                out.write(annotated_frame)
                
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count} frames...")
                    
            except Exception as e:
                # wait_for_frames timeout or EOF
                if str(e).find("timeout") != -1:
                    continue 
                print(f"Playback ended or hit exception: {e}")
                break

        # Cleanup
        try:
            pipeline.stop()
        except Exception:
            pass
        out.release()
        end_time = time.time()
        duration = end_time - start_time
        fps = frame_count / duration if duration > 0 else 0
        print(f"Processing complete. Saved to {self.output_path}")
        print(f"Total time: {duration:.2f}s, FPS: {fps:.2f}")
