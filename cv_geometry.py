import numpy as np

class VisionMetrics:
    """
    Modulo de Geometria de Camara para Squat AI.
    Convierte medidas de pixeles en pantalla a centimetros reales (Metric Vision).
    Cumple con los requisitos de Geometria y Calibracion de Computer Vision.
    """
    
    def __init__(self, reference_width_cm=45.0):
        # Asumimos una anchura media de hombros estandar (45cm biacromial)
        self.ref_cm = reference_width_cm
        self.cm_per_pixel = None # Factor de conversion
        
    def calibrate_scale(self, keypoints):
        """
        Calcula el factor cm/pixel basandose en la anchura de hombros actual.
        Llamar a esta funcion cuando el atleta este de pie y recto.
        """
        # Keypoints YOLO: 5 (hombro izq), 6 (hombro der)
        h_izq = keypoints[5][:2] # Tomamos solo (x, y)
        h_der = keypoints[6][:2]
        
        # Calculamos la distancia euclidea en pixeles (Ancho de hombros)
        pixel_width = np.linalg.norm(h_izq - h_der)
        
        if pixel_width > 10: # Evitar division por cero o errores de YOLO
            self.cm_per_pixel = self.ref_cm / pixel_width
            return True
        return False
        
    def pixels_to_cm(self, pixel_distance):
        """Convierte una distancia en pixeles a centimetros reales."""
        if self.cm_per_pixel is not None:
            return round(pixel_distance * self.cm_per_pixel, 1)
        return 0.0

# --- Prueba del Modulo ---
if __name__ == "__main__":
    v_metrics = VisionMetrics()
    
    # Simulamos keypoints YOLO (h_izq en 100,200 y h_der en 300,200) -> 200px
    simulated_kp = np.zeros((17, 3))
    simulated_kp[5] = [100, 200, 0.9] # Hombro izq
    simulated_kp[6] = [300, 200, 0.9] # Hombro der
    
    if v_metrics.calibrate_scale(simulated_kp):
        print("\n--- INFORME DE CALIBRACION OPTICA ---")
        print(f"Ancho de hombros detectado: 200 pixeles.")
        print(f"Referencia real: 45.0 cm.")
        print(f"Factor de conversion: {round(v_metrics.cm_per_pixel, 4)} cm/pixel.")
        
        # Probamos a convertir la profundidad de 120px
        prof_cm = v_metrics.pixels_to_cm(120)
        print(f"Una bajada de 120 pixeles equivale a: {prof_cm} cm reales.")