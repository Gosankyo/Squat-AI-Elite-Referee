import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import pyttsx3
import threading

print("🚀 Iniciando Motor Biomecánico en Vivo...")

# --- 1. CONFIGURACIÓN DE VOZ (TTS) ---
def coach_habla(texto):
    """Ejecuta la voz en un hilo separado para no congelar la cámara"""
    def run_tts():
        try:
            # Inicializamos el motor de voz
            engine = pyttsx3.init()
            engine.setProperty('rate', 160) # Velocidad al hablar
            engine.say(texto)
            engine.runAndWait()
        except:
            pass
    threading.Thread(target=run_tts, daemon=True).start()

# --- 2. CARGA DE MODELOS ---
print("🧠 Cargando Inteligencia Artificial...")
yolo_pose = YOLO('yolov8n-pose.pt')
model_ia = load_model('modelo_squat_biomecanico.h5')
print("✅ IA Cargada. Encendiendo cámara...")

# --- 3. CONFIGURACIÓN DE WEBCAM ---
cap = cv2.VideoCapture(0) # 0 es la cámara por defecto de tu PC

# Variables de control
secuencia = []
contador = 0
estado = "ESPERANDO"
mejor_conf = 0.0
y_ini = None
frames_bloqueo = 0
ultimo_veredicto = "Haz una sentadilla"
color_hud = (255, 255, 255)

coach_habla("Sistema preparado. Vamos a por esa sentadilla.")

# --- 4. BUCLE EN TIEMPO REAL ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Espejo visual (opcional, para verte como en un espejo)
    # frame = cv2.flip(frame, 1) 
    
    # Detección rápida de YOLO
    res = yolo_pose.predict(frame, conf=0.5, verbose=False)
    frame_final = res[0].plot() # Dibujamos el esqueleto
    
    if res[0].keypoints and len(res[0].keypoints.data) > 0:
        det = res[0].keypoints.data.cpu().numpy()
        boxes = res[0].boxes.data.cpu().numpy()
        
        # Seleccionamos a la persona principal
        atleta_kp = det[0]
        
        # Tracker altura (Media de los hombros y cadera)
        h_y = (atleta_kp[5][1] + atleta_kp[6][1]) / 2
        cadera_y = (atleta_kp[11][1] + atleta_kp[12][1]) / 2
        
        if y_ini is None: 
            y_ini = h_y
            continue
            
        # Extracción de puntos normalizados para la IA (Centrado en Cadera)
        kp_crudo = res[0].keypoints.data[0].cpu().numpy().flatten()
        
        if len(kp_crudo) == 51:
            hip_l = kp_crudo[11*3 : 11*3+3]
            hip_r = kp_crudo[12*3 : 12*3+3]
            centro_cadera = (hip_l + hip_r) / 2
            
            kp_norm = kp_crudo.copy()
            for i in range(17):
                kp_norm[i*3 : i*3+3] -= centro_cadera
            
            secuencia.append(kp_norm)
        
        # Mantenemos los últimos 30 fotogramas
        if len(secuencia) > 30: 
            secuencia.pop(0)

        # --- MÁQUINA DE ESTADOS Y PREDICCIÓN ---
        if frames_bloqueo > 0: 
            frames_bloqueo -= 1
        else:
            dist = h_y - y_ini
            
            # Detecta que empieza a bajar
            if dist > 50 and estado == "ESPERANDO": 
                estado = "BAJANDO"
                mejor_conf = 0.0
                color_hud = (0, 165, 255) # Naranja
                ultimo_veredicto = "Analizando..."
            
            if estado == "BAJANDO":
                # Predicción continua
                seq_array = np.array(secuencia)
                pad_len = 30 - len(seq_array)
                if pad_len > 0:
                    seq_array = np.pad(seq_array, ((0, pad_len), (0, 0)), 'constant')
                
                pred = model_ia.predict(np.expand_dims(seq_array, axis=0), verbose=0)[0]
                if pred[0] > mejor_conf: mejor_conf = pred[0]
            
            # Detecta que ha subido (Termina la repetición)
            if dist < 30 and estado == "BAJANDO":
                contador += 1
                frames_bloqueo = 45 # Bloqueo para evitar dobles conteos
                estado = "ESPERANDO"
                
                # LA DECISIÓN FINAL Y LA VOZ
                if mejor_conf > 0.50:
                    ultimo_veredicto = f"VALIDA ({round(mejor_conf*100)}%)"
                    color_hud = (0, 255, 0) # Verde
                    coach_habla("¡Válida! Tres luces blancas.")
                else: 
                    ultimo_veredicto = f"NULA ({round((1-mejor_conf)*100)}%)"
                    color_hud = (0, 0, 255) # Rojo
                    coach_habla("¡Nula! No has roto el paralelo.")

    # --- INTERFAZ VISUAL EN PANTALLA (HUD) ---
    cv2.putText(frame_final, f"REPS: {contador}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(frame_final, estado, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame_final, ultimo_veredicto, (20, frame_final.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color_hud, 4)
    
    cv2.imshow("🏋️‍♂️ SQUAT AI - Live Demo", frame_final)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()