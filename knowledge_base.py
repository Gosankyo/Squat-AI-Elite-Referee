class BiomechanicsExpertSystem:
    """
    Rule-based inference engine for biomechanical validation.
    Applies expert-defined thresholds to keypoint data.
    References: Schoenfeld (2010), Escamilla (2001), Bryanton (2012).
    """
    
    def __init__(self):
        # Expert-defined thresholds for biomechanical validation
        self.rules = {
            "max_torso_angle": 65.5,  # Max forward lean before shear stress risk
            "knee_valgus_threshold": 10.0,  # Max inward knee deviation (degrees)
            "hip_depth_multiplier": 1.1  # Depth achievement coefficient
        }
        
    def infer_diagnosis(self, facts):
        """
        Inference engine: evaluates facts against expert rules.
        Returns safety assessment with diagnostic feedback.
        """
        diagnosis = []
        is_safe = True
        
        # Evaluate torso angle against Schoenfeld threshold
        if "torso_angle" in facts:
            if facts["torso_angle"] > self.rules["max_torso_angle"]:
                diagnosis.append("WARNING (Schoenfeld): Excessive torso lean increases spinal shear stress.")
                is_safe = False
                
        # Evaluate knee stability against Escamilla threshold
        if "knee_valgus_angle" in facts:
            if facts["knee_valgus_angle"] > self.rules["knee_valgus_threshold"]:
                diagnosis.append("WARNING (Escamilla): Knee valgus detected - assess ACL loading.")
                is_safe = False
                
        # Validate squat depth achievement
        if "depth_achieved" in facts and not facts["depth_achieved"]:
            diagnosis.append("TECHNICAL ALERT (Bryanton): Insufficient depth achieved for optimal posterior chain recruitment.")
            is_safe = False

        if not diagnosis:
            diagnosis.append("PASS: Biomechanically sound execution pattern detected.")

        return {
            "is_safe": is_safe,
            "feedback": diagnosis
        }

# Test inference engine
if __name__ == "__main__":
    expert = BiomechanicsExpertSystem()
    
    # Simulate biomechanical fact set
    simulated_facts = {
        "torso_angle": 68.0, 
        "knee_valgus_angle": 5.0, 
        "depth_achieved": True
    }
    
    result = expert.infer_diagnosis(simulated_facts)
    print("[EXPERT] Biomechanical Inference Report")
    for msg in result["feedback"]:
        print(f"  - {msg}")