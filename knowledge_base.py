class BiomechanicsExpertSystem:
    """
    Sistema Basado en Conocimiento (Knowledge-Based System) para Squat AI.
    Actua como un Motor de Inferencia Forward-Chaining que aplica reglas
    de literatura cientifica (Schoenfeld, Escamilla) a los hechos geometricos.
    """
    
    def __init__(self):
        # La "Base de Conocimiento": Reglas y umbrales definidos por expertos
        self.rules = {
            "max_torso_angle": 65.5, # Schoenfeld (2010): Limite de tension por cizalla
            "knee_valgus_threshold": 10.0, # Escamilla (2001): Grados de tolerancia hacia adentro
            "hip_depth_multiplier": 1.1 # Bryanton (2012): Reclutamiento de cadena posterior
        }
        
    def infer_diagnosis(self, facts):
        """
        Motor de Inferencia: Recibe un diccionario de 'hechos' (facts) y
        aplica las reglas para generar un diagnostico.
        """
        diagnosis = []
        is_safe = True
        
        # Hecho 1: Inclinacion del Torso
        if "torso_angle" in facts:
            if facts["torso_angle"] > self.rules["max_torso_angle"]:
                diagnosis.append("ADVERTENCIA (Schoenfeld): Inclinacion excesiva del torso. Riesgo de cizalla espinal.")
                is_safe = False
                
        # Hecho 2: Estabilidad de la Rodilla (Valgo)
        if "knee_valgus_angle" in facts:
            if facts["knee_valgus_angle"] > self.rules["knee_valgus_threshold"]:
                diagnosis.append("ADVERTENCIA (Escamilla): Valgo de rodilla detectado. Riesgo para ligamento cruzado (ACL).")
                is_safe = False
                
        # Hecho 3: Analisis de Profundidad Relativa
        if "depth_achieved" in facts and not facts["depth_achieved"]:
            diagnosis.append("AVISO TECNICO (Bryanton): Profundidad insuficiente para maximo reclutamiento de cadena posterior.")
            is_safe = False

        if not diagnosis:
            diagnosis.append("Diagnostico Biomecanico: Ejecucion Segura y Competitiva.")

        return {
            "is_safe": is_safe,
            "feedback": diagnosis
        }

# --- Prueba del Motor de Inferencia ---
if __name__ == "__main__":
    expert = BiomechanicsExpertSystem()
    
    simulated_facts = {
        "torso_angle": 68.0, 
        "knee_valgus_angle": 5.0, 
        "depth_achieved": True
    }
    
    result = expert.infer_diagnosis(simulated_facts)
    print("\n--- INFORME DEL SISTEMA EXPERTO ---")
    for msg in result["feedback"]:
        print("-", msg)