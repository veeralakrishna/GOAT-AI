import cv2
import numpy as np

class BiometricEstimator:
    def __init__(self, pixels_per_cm=10.0, fx=None, fy=None):
        self.pixels_per_cm = pixels_per_cm
        self.fx = fx
        self.fy = fy

    def estimate_dimensions(self, mask_contour, depth_cm=None):
        """
        Estimate dimensions from a contour using a rotated bounding box.
        
        Args:
            mask_contour: numpy array of points (the contour)
            depth_cm: optional median depth distance to the object in cm
            
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
        
        if depth_cm is not None and self.fx is not None and self.fy is not None:
            # Use intrinsic parameters and true depth
            # Distance in real world X = (Width_pixels * Z) / fx
            dim1 = (width * depth_cm) / self.fx
            dim2 = (height * depth_cm) / self.fy
            
            # Area estimation (approximated based on bounding box conversion ratio)
            ratio_x = depth_cm / self.fx
            ratio_y = depth_cm / self.fy
            area_pixels = cv2.contourArea(mask_contour)
            area_cm2 = area_pixels * (ratio_x * ratio_y)
        else:
            # Fallback to pure 2D constant pixel ratio
            dim1 = width / self.pixels_per_cm
            dim2 = height / self.pixels_per_cm
            
            area_pixels = cv2.contourArea(mask_contour)
            area_cm2 = area_pixels / (self.pixels_per_cm ** 2)
        
        length_cm = max(dim1, dim2)
        width_cm = min(dim1, dim2)
        
        # Estimate Chest Girth
        # The bounding box includes legs, neck, and head. 
        # Empirical ratio: True chest depth is approx 27% of the full standing height for these goats.
        chest_diameter_cm = width_cm * 0.27
        chest_girth_cm = chest_diameter_cm * np.pi
        
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        return {
            "length_cm": round(length_cm, 2),
            "width_cm": round(width_cm, 2),
            "chest_girth_cm": round(chest_girth_cm, 2),
            "area_cm2": round(area_cm2, 2),
            "rect_points": box,
            "center": center
        }
