import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3
import threading
from voice_trigger import NLPVoiceAssistant
from knowledge_base import BiomechanicsExpertSystem
from cv_geometry import VisionMetrics

print("[SYSTEM] Initializing biomechanical analysis engine...")

# --- 1. TEXT-TO-SPEECH MODULE ---
def coach_habla(texto):
    """Output voice feedback in background thread."""
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(texto)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run_tts, daemon=True).start()

# --- 2. MODEL INITIALIZATION ---
print("[MODELS] Loading YOLO pose detector...")
yolo_pose = YOLO('yolov8n-pose.pt')
print("[MODELS] Loading LSTM classifier...")
model_ia = load_model('modelo_squat_biomecanico.h5')

print("[SYSTEM] Initializing expert system...")
expert_system = BiomechanicsExpertSystem()

print("[SYSTEM] Initializing vision metrics module...")
v_metrics = VisionMetrics(reference_width_cm=45.0)

# --- 3. VOICE COMMAND INTERFACE ---
assistant = NLPVoiceAssistant()
print("\n" + "-" * 50)
print("[VOICE] Waiting for voice trigger...")
print("[VOICE] Say: 'start analysis' to begin")
print("-" * 50)

if not assistant.wait_for_command(wake_word="start analysis"):
    print("[SYSTEM] Shutdown requested.")
    exit()

print("[VOICE] Command accepted. Initializing camera feed...")
coach_habla("System initialized.")

# --- 4. CAMERA SETUP ---
cap = cv2.VideoCapture(0)

# State variables
sequence = []
rep_count = 0
state = "IDLE"
best_confidence = 0.0
initial_height = None
lockout_frames = 0
last_verdict = "Waiting for squat..."
hud_color = (255, 255, 255)
descent_distance_cm = 0.0

# --- 5. MAIN PROCESSING LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    res = yolo_pose.predict(frame, conf=0.5, verbose=False)
    frame_final = res[0].plot() 
    
    # Check if there are detections and keypoints
    if res[0].boxes and res[0].keypoints and len(res[0].keypoints.data) > 0:
        det = res[0].keypoints.data.cpu().numpy()
        boxes = res[0].boxes.xywh.cpu().numpy() # [x_center, y_center, width, height]
        
        # [FIX] Select the athlete closest to the center of the camera
        frame_center_x = frame.shape[1] / 2
        best_idx = np.argmin([abs(box[0] - frame_center_x) for box in boxes])
        
        atleta_kp = det[best_idx]
        
        h_x = (atleta_kp[5][0] + atleta_kp[6][0]) / 2
        h_y = (atleta_kp[5][1] + atleta_kp[6][1]) / 2
        c_x = (atleta_kp[11][0] + atleta_kp[12][0]) / 2
        c_y = (atleta_kp[11][1] + atleta_kp[12][1]) / 2
        
        if initial_height is None:
            initial_height = h_y
            continue
            
        if state == "IDLE":
            # Continuous calibration when standing
            v_metrics.calibrate_scale(atleta_kp)
            descent_distance_cm = 0.0
            
            if h_y < initial_height:
                initial_height = h_y
            elif h_y > initial_height and (h_y - initial_height) < 30:
                initial_height = int((initial_height * 0.9) + (h_y * 0.1))

        # Use the specific index for the centered athlete
        kp_data = res[0].keypoints.data[best_idx].cpu().numpy().flatten()
        
        if len(kp_data) == 51:
            # Extract and normalize to hip center
            hip_left = kp_data[11*3 : 11*3+3]
            hip_right = kp_data[12*3 : 12*3+3]
            hip_center = (hip_left + hip_right) / 2
            kp_normalized = kp_data.copy()
            for i in range(17):
                kp_normalized[i*3 : i*3+3] -= hip_center
            sequence.append(kp_normalized)
        
        if len(sequence) > 30:
            sequence.pop(0)

        # --- STATE MACHINE ---
        if lockout_frames > 0:
            lockout_frames -= 1
        else:
            height_delta = h_y - initial_height
            # Update real-world descent distance
            if height_delta > 0:
                descent_distance_cm = v_metrics.pixels_to_cm(height_delta)
            
            # Squat initiation trigger
            if height_delta > 120 and state == "IDLE":
                state = "DESCENDING"
                best_confidence = 0.0
                hud_color = (100, 149, 237)  # Cornflower blue
                last_verdict = "Analyzing rep..."
            
            if state == "DESCENDING":
                # Get prediction from LSTM
                seq_array = np.array(sequence)
                pad_len = 30 - len(seq_array)
                if pad_len > 0:
                    seq_array = np.pad(seq_array, ((0, pad_len), (0, 0)), 'constant')
                
                pred = model_ia.predict(np.expand_dims(seq_array, axis=0), verbose=0)[0]
                if pred[0] > best_confidence:
                    best_confidence = pred[0]
            
            # Squat completion
            if height_delta < 40 and state == "DESCENDING":
                rep_count += 1
                lockout_frames = 45
                state = "IDLE"
                
                # --- [FIX] STRICT VERDICT HIERARCHY ---
                depth_valid = True if best_confidence > 0.50 else False
                
                dx = abs(h_x - c_x)
                dy = abs(h_y - c_y)
                angulo_torso = np.degrees(np.arctan2(dx, dy)) if dy != 0 else 0

                # Passed as fact to the expert system (NameError fixed)
                facts = {
                    "depth_achieved": depth_valid,
                    "torso_angle": angulo_torso
                }

                diagnostico = expert_system.infer_diagnosis(facts)

                if not depth_valid:
                    last_verdict = "FAIL - High Squat (Depth)"
                    hud_color = (0, 165, 255)  # Orange for depth failure
                    coach_habla("Squat was too high. Go deeper.")
                elif not diagnostico["is_safe"]:
                    last_verdict = "FAIL - Form Error"
                    hud_color = (0, 0, 255)  # Red for safety failure
                    coach_habla(diagnostico["feedback"][0])
                else:
                    last_verdict = f"PASS ({round(best_confidence*100)}%)"
                    hud_color = (0, 255, 0)  # Green
                    coach_habla("Valid rep.")

    # --- HUD RENDERING ---
    cv2.putText(frame_final, f"Reps: {rep_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(frame_final, f"State: {state}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame_final, f"Descent: {descent_distance_cm}cm", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame_final, last_verdict, (20, frame_final.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, hud_color, 3)
    
    cv2.imshow("Squat AI - Live Analysis", frame_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()