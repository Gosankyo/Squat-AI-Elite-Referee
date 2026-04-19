import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3
import threading
from voice_trigger import NLPVoiceAssistant
from knowledge_base import BiomechanicsExpertSystem
from cv_geometry import VisionMetrics # <-- INTEGRACIÓN COMPUTER VISION

print("🚀 Iniciando Motor Biomecánico Multidisciplinar...")

# --- 1. CONFIGURACIÓN DE VOZ (TTS) ---
def coach_habla(texto):
    """Ejecuta la voz en un hilo separado para no congelar la cámara"""
    def run_tts():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(texto)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run_tts, daemon=True).start()

# --- 2. CARGA DE MODELOS E IA CLÁSICA ---
print("🧠 Cargando Redes Neuronales (YOLO & LSTM)...")
yolo_pose = YOLO('yolov8n-pose.pt')
model_ia = load_model('modelo_squat_biomecanico.h5')

print("🧑‍⚕️ Cargando Sistema Experto Biomecánico...")
expert_system = BiomechanicsExpertSystem()

print("📏 Cargando Módulo de Geometría Óptica...")
v_metrics = VisionMetrics(reference_width_cm=45.0) # Hombros estándar

# --- 3. MÓDULO NLP: ACTIVACIÓN POR VOZ ---
assistant = NLPVoiceAssistant()
print("\n" + "="*50)
print("🎙️ SISTEMA ESPERANDO COMANDO DE VOZ")
print("="*50)
print("Dí claramente: 'start analysis' para comenzar...")

if not assistant.wait_for_command(wake_word="start analysis"):
    print("Saliendo...")
    exit()

print("¡Comando reconocido! Encendiendo cámara...")
coach_habla("Sistema iniciado. Calibrando telemetría óptica.")

# --- 4. CONFIGURACIÓN DE CÁMARA ---
cap = cv2.VideoCapture(0)

# Variables de control
secuencia = []
contador = 0
estado = "ESPERANDO"
mejor_conf = 0.0
y_ini = None
frames_bloqueo = 0
ultimo_veredicto = "Haz una sentadilla"
color_hud = (255, 255, 255)
dist_cm = 0.0 # Variable para guardar los centímetros bajados

# --- 5. BUCLE EN TIEMPO REAL ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    res = yolo_pose.predict(frame, conf=0.5, verbose=False)
    frame_final = res[0].plot() 
    
    if res[0].keypoints and len(res[0].keypoints.data) > 0:
        det = res[0].keypoints.data.cpu().numpy()
        atleta_kp = det[0]
        
        h_x = (atleta_kp[5][0] + atleta_kp[6][0]) / 2
        h_y = (atleta_kp[5][1] + atleta_kp[6][1]) / 2
        c_x = (atleta_kp[11][0] + atleta_kp[12][0]) / 2
        c_y = (atleta_kp[11][1] + atleta_kp[12][1]) / 2
        
        if y_ini is None: 
            y_ini = h_y
            continue
            
        if estado == "ESPERANDO":
            # [QUICK WIN CV] Calibración métrica continua mientras estás de pie
            v_metrics.calibrate_scale(atleta_kp)
            dist_cm = 0.0 # Reseteamos la distancia al esperar
            
            if h_y < y_ini:
                y_ini = h_y
            elif h_y > y_ini and (h_y - y_ini) < 30:
                y_ini = int((y_ini * 0.9) + (h_y * 0.1))

        kp_crudo = res[0].keypoints.data[0].cpu().numpy().flatten()
        
        if len(kp_crudo) == 51:
            hip_l = kp_crudo[11*3 : 11*3+3]
            hip_r = kp_crudo[12*3 : 12*3+3]
            centro_cadera = (hip_l + hip_r) / 2
            kp_norm = kp_crudo.copy()
            for i in range(17):
                kp_norm[i*3 : i*3+3] -= centro_cadera
            secuencia.append(kp_norm)
        
        if len(secuencia) > 30: 
            secuencia.pop(0)

        # --- MÁQUINA DE ESTADOS ---
        if frames_bloqueo > 0: 
            frames_bloqueo -= 1
        else:
            dist = h_y - y_ini
            # Calculamos la distancia en CM reales usando el calibrador
            if dist > 0:
                dist_cm = v_metrics.pixels_to_cm(dist)
            
            if dist > 120 and estado == "ESPERANDO": 
                estado = "BAJANDO"
                mejor_conf = 0.0
                color_hud = (0, 165, 255)
                ultimo_veredicto = "Analizando Profundidad..."
            
            if estado == "BAJANDO":
                seq_array = np.array(secuencia)
                pad_len = 30 - len(seq_array)
                if pad_len > 0:
                    seq_array = np.pad(seq_array, ((0, pad_len), (0, 0)), 'constant')
                
                pred = model_ia.predict(np.expand_dims(seq_array, axis=0), verbose=0)[0]
                if pred[0] > mejor_conf: mejor_conf = pred[0]
            
            if dist < 40 and estado == "BAJANDO":
                contador += 1
                frames_bloqueo = 45 
                estado = "ESPERANDO"
                
                # --- LA DECISIÓN FINAL ---
                profundidad_ok = True if mejor_conf > 0.50 else False
                
                dx = abs(h_x - c_x)
                dy = abs(h_y - c_y)
                angulo_torso = np.degrees(np.arctan2(dx, dy)) if dy != 0 else 0

                facts = {
                    "depth_achieved": profundidad_ok,
                    "torso_angle": angulo_torso
                }

                diagnostico = expert_system.infer_diagnosis(facts)

                if diagnostico["is_safe"]:
                    ultimo_veredicto = f"VALIDA ({round(mejor_conf*100)}%)"
                    color_hud = (0, 255, 0)
                    coach_habla("Válida.")
                else: 
                    ultimo_veredicto = "NULA / AVISO"
                    color_hud = (0, 0, 255)
                    coach_habla(diagnostico["feedback"][0])

    # --- HUD VISUAL ---
    cv2.putText(frame_final, f"REPS: {contador}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(frame_final, estado, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # Mostramos los centímetros bajados en color amarillo cian
    cv2.putText(frame_final, f"DESCENSO: {dist_cm} cm", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame_final, ultimo_veredicto, (20, frame_final.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_hud, 3)
    
    cv2.imshow("🏋️‍♂️ Squat AI - Visión Métrica", frame_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()