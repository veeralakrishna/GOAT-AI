import cv2
import numpy as np

class BiometricEstimator:
    def __init__(self, pixels_per_cm=10.0):
        self.pixels_per_cm = pixels_per_cm

    def estimate_dimensions(self, mask_contour):
        """
        Estimate dimensions from a contour using a rotated bounding box.
        
        Args:
            mask_contour: numpy array of points (the contour)
            
        Returns:
            dict: {
                "length_cm": float,
                "width_cm": float, # Often proxy for girth/height depending on angle
                "area_cm2": float,
                "rect_points": np.array (box corners for drawing)
            }
        """
        # Get rotated rectangle
        rect = cv2.minAreaRect(mask_contour)
        (center), (width, height), angle = rect
        
        # Determine which is length vs width (Length usually > Width for a standing goat side-on)
        dim1 = width / self.pixels_per_cm
        dim2 = height / self.pixels_per_cm
        
        length_cm = max(dim1, dim2)
        width_cm = min(dim1, dim2)
        
        # Calculate Area
        area_pixels = cv2.contourArea(mask_contour)
        area_cm2 = area_pixels / (self.pixels_per_cm ** 2)
        
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        return {
            "length_cm": round(length_cm, 2),
            "width_cm": round(width_cm, 2),
            "area_cm2": round(area_cm2, 2),
            "rect_points": box,
            "center": center
        }
