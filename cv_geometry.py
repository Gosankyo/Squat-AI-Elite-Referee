import numpy as np

class VisionMetrics:
    """
    Pixel-to-metric conversion using shoulder width calibration.
    Enables real-world distance measurements from video coordinates.
    """
    
    def __init__(self, reference_width_cm=45.0):
        # Reference shoulder width (biacromial distance in cm)
        self.ref_cm = reference_width_cm
        # Calibration state: cm per pixel ratio (None until calibrated)
        self.cm_per_pixel = None
        
    def calibrate_scale(self, keypoints):
        """
        Calibrates pixel-to-cm conversion using shoulder width.
        Should be called with athlete in upright position.
        """
        # Extract shoulder keypoints from YOLO (indices 5 and 6)
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        
        # Calculate shoulder width in pixels using Euclidean distance
        pixel_width = np.linalg.norm(left_shoulder - right_shoulder)
        
        # Validate measurement and update conversion factor
        if pixel_width > 10:
            self.cm_per_pixel = self.ref_cm / pixel_width
            return True
        return False
        
    def pixels_to_cm(self, pixel_distance):
        """Converts pixel distance to real-world centimeters."""
        if self.cm_per_pixel is not None:
            return round(pixel_distance * self.cm_per_pixel, 1)
        return 0.0

# Test module calibration
if __name__ == "__main__":
    v_metrics = VisionMetrics()
    
    # Simulate YOLO keypoints: shoulders separated by 200 pixels
    simulated_keypoints = np.zeros((17, 3))
    simulated_keypoints[5] = [100, 200, 0.9]  # Left shoulder
    simulated_keypoints[6] = [300, 200, 0.9]  # Right shoulder
    
    if v_metrics.calibrate_scale(simulated_keypoints):
        print("[CALIBRATION] Vision metrics calibration successful")
        print(f"  Detected shoulder width: 200 pixels")
        print(f"  Reference width: 45.0 cm")
        print(f"  Conversion factor: {round(v_metrics.cm_per_pixel, 4)} cm/pixel")
        
        # Test pixel-to-cm conversion
        depth_cm = v_metrics.pixels_to_cm(120)
        print(f"  Descent of 120 pixels: {depth_cm} cm")