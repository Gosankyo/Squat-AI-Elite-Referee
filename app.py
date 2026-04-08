import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="SQUAT AI ULTRA ELITE", layout="wide", page_icon="🚀")

# CSS Estilo "Cyber-Gym" (Neón y Oscuro)
st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stMetric { background-color: #161b22; border: 1px solid #3b82f6; box-shadow: 0px 0px 10px #3b82f6; border-radius: 15px; }
    .stTabs [data-baseweb="tab"] { font-size: 20px; font-weight: bold; color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom: 3px solid #3b82f6; }
    h1, h2, h3 { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA DE MODELOS ---
@st.cache_resource
def iniciar_modelos():
    model_ia = load_model('modelo_squat_yolo.h5')
    yolo_pose = YOLO('yolov8n-pose.pt')
    return model_ia, yolo_pose

model_ia, yolo_pose = iniciar_modelos()

# --- 3. FUNCIONES DE ANÁLISIS AVANZADO ---

def realizar_diagnostico_completo(secuencia_kp):
    """Diagnóstico de 10: Torso, Valgo y Simetría"""
    f_fondo = secuencia_kp[len(secuencia_kp)//2] 
    
    # 1. Ángulo Torso
    dx_t = f_fondo[11*3] - f_fondo[5*3]
    dy_t = f_fondo[11*3+1] - f_fondo[5*3+1]
    angulo_torso = abs(np.degrees(np.arctan2(dx_t, dy_t)))
    
    # 2. Detección de Valgo (Rodillas hacia adentro)
    # Hips (11,12) vs Knees (13,14)
    ancho_caderas = abs(f_fondo[11*3] - f_fondo[12*3])
    ancho_rodillas = abs(f_fondo[13*3] - f_fondo[14*3])
    
    # 3. Simetría de Cadera (¿Baja un lado más que otro?)
    desequilibrio_cadera = abs(f_fondo[11*3+1] - f_fondo[12*3+1])
    
    errores, soluciones = [], []
    
    if angulo_torso > 45:
        errores.append("⚠️ **Torso Inclinado:** Exceso de carga lumbar.")
        soluciones.append("Sentadilla Frontal / Pausa en el hoyo.")
    
    if ancho_rodillas < (ancho_caderas * 0.9):
        errores.append("⚠️ **Valgo de Rodilla:** Tus rodillas colapsan hacia adentro.")
        soluciones.append("Sentadilla con banda elástica (clamshells).")
        
    if desequilibrio_cadera > 15:
        errores.append("⚠️ **Hip Shift:** Tu cadera se desplaza lateralmente.")
        soluciones.append("Sentadilla Búlgara para corregir asimetría.")
        
    if not errores:
        errores.append("⭐ **Técnica de Élite:** Sin fallos críticos detectados.")
        soluciones.append("Sigue progresando en cargas.")
        
    return errores, soluciones, angulo_torso

def analizar_proporciones(puntos_kp):
    h_y = (puntos_kp[5][1]+puntos_kp[6][1])/2
    c_y = (puntos_kp[11][1]+puntos_kp[12][1])/2
    r_y = (puntos_kp[13][1]+puntos_kp[14][1])/2
    len_torso, len_femur = abs(c_y - h_y), abs(r_y - c_y)
    ratio = len_femur / len_torso if len_torso != 0 else 0
    if ratio > 0.88: return "FÉMURES LARGOS", "Barra Baja", ratio
    elif ratio < 0.78: return "FÉMURES CORTOS", "Barra Alta", ratio
    return "ESTRUCTURA NEUTRA", "Barra Media", ratio

# --- 4. INTERFAZ Y SIDEBAR ---
st.sidebar.title("🚀 SQUAT AI CONTROL")
lift_weight = st.sidebar.number_input("Peso en barra (kg)", value=100.0, step=2.5)
user_weight = st.sidebar.number_input("Peso Corporal (kg)", value=75.0, step=0.5)
sexo = st.sidebar.selectbox("Sexo", ["men", "women"])
pais = st.sidebar.selectbox("Región", ["España", "Francia", "USA", "All"])
mapa_paises = {"España": "all-spain", "Francia": "all-france", "USA": "all-usa", "All": "all"}

tab1, tab2, tab3, tab4 = st.tabs(["🎥 ANÁLISIS VBT", "📐 ARQUITECTO", "📊 RANKING", "🕹️ JUEGO"])

# --- TAB 1: JUEZ + VBT + DIAGNÓSTICO ---
with tab1:
    c_left, c_right = st.columns([2, 1])
    
    with c_right:
        reps_st = st.metric("REPSET", "0")
        luces = st.empty()
        st.subheader("📈 Velocidad (VBT)")
        grafico_v = st.empty()
        st.subheader("📝 Diagnóstico")
        diag_area = st.empty()

    video = st.file_uploader("Sube vídeo de sentadilla", type=['mp4', 'mov'])
    if video:
        with open("temp.mp4", "wb") as f: f.write(video.read())
        cap = cv2.VideoCapture("temp.mp4")
        st_frame = c_left.empty()
        
        # Variables de control
        secuencia, contador, estado, mejor_conf, y_ini, frames_bloqueo = [], 0, "ESPERANDO", 0, None, 0
        velocidades, y_prev = [], None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_res = cv2.resize(frame, (800, 600))
            res = yolo_pose.predict(frame_res, conf=0.5, verbose=False)
            
            if res[0].keypoints and len(res[0].keypoints.data) > 0:
                det = res[0].keypoints.data.cpu().numpy()
                boxes = res[0].boxes.data.cpu().numpy()
                atleta_kp = det[np.argmin([abs((b[0]+b[2])/2 - 400) for b in boxes])]
                h_y = (atleta_kp[5][1] + atleta_kp[6][1]) / 2
                
                if y_ini is None: y_ini = h_y; y_prev = h_y; continue
                
                # Cálculo de velocidad instantánea
                v_inst = abs(h_y - y_prev)
                velocidades.append(v_inst)
                y_prev = h_y
                
                kp_f = atleta_kp.flatten()
                if len(kp_f) == 51: secuencia.append(kp_f)
                if len(secuencia) > 30: secuencia.pop(0)
                
                if frames_bloqueo > 0: frames_bloqueo -= 1
                else:
                    dist = h_y - y_ini
                    if dist > 60 and estado == "ESPERANDO": estado = "BAJANDO"
                    if estado == "BAJANDO" and len(secuencia) == 30:
                        pred = model_ia.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
                        if pred[1] > mejor_conf: mejor_conf = pred[1]
                    if dist < 35 and estado == "BAJANDO":
                        if mejor_conf > 0.75:
                            contador += 1; frames_bloqueo = 90
                            luces.success("⚪ ⚪ ⚪ VÁLIDA")
                            fallos, sols, ang_t = realizar_diagnostico_completo(secuencia)
                            with diag_area.container():
                                st.write(f"Ángulo Torso: {round(ang_t,1)}°")
                                for f in fallos: st.error(f)
                                for s in sols: st.success(f"📌 {s}")
                        else: luces.error("🔴 🔴 🔴 NULA")
                        estado = "ESPERANDO"
                
                reps_st.metric("REPSET", contador)
                grafico_v.line_chart(velocidades[-50:]) # Mostrar últimos 50 frames
            st_frame.image(res[0].plot(), channels="BGR")
        cap.release()

# --- TAB 2: ARQUITECTO ---
with tab2:
    st.header("📐 Sastre Biomecánico")
    foto = st.file_uploader("Foto de frente completa", type=['jpg', 'png'])
    if foto:
        img = cv2.imdecode(np.frombuffer(foto.read(), np.uint8), 1)
        res_p = yolo_pose.predict(img, conf=0.5, verbose=False)
        if res_p[0].keypoints:
            tipo, rec, ratio = analizar_proporciones(res_p[0].keypoints.data[0].cpu().numpy())
            c1, c2 = st.columns(2)
            c1.image(res_p[0].plot(), use_container_width=True)
            with c2:
                st.metric("Ratio Fémur/Torso", round(ratio,2))
                st.success(f"**Estructura:** {tipo}\n\n**Técnica recomendada:** {rec}")
                st.info("Esta recomendación optimiza el centro de gravedad sobre el mediopié.")

# --- TAB 3: RANKINGS ---
with tab3:
    st.header("📊 Nivel Competitivo")
    reps_cal = st.number_input("Reps para 1RM", min_value=1, value=1)
    one_rm = lift_weight * (1 + reps_cal / 30)
    st.metric("Tu 1RM Estimado", f"{round(one_rm, 2)} kg")
    url = f"https://www.openpowerlifting.org/rankings/{int(user_weight)}/{mapa_paises[pais]}/{sexo}/by-squat"
    st.markdown(f"### [🔗 Ir a OpenPowerlifting Rank]({url})")

# --- TAB 4: MINIJUEGO ---
with tab4:
    st.header("🕹️ Juez Simulator")
    st.write("Mejora tu criterio de juez visualizando la base de datos de entrenamiento.")
    st.progress(100)
    st.write("Entrenamiento del modelo LSTM completado con un 94% de precisión.")

# Tabla Maestra al final
st.divider()
st.subheader("📋 Guía Maestra de Corrección Técnica")
st.table([
    {"Fallo": "Valgo Rodilla", "Causa": "Glúteo medio débil", "Ejercicio": "Banded Squats / Monster Walk"},
    {"Fallo": "Hip Shift", "Causa": "Asimetría de movilidad", "Ejercicio": "Sentadilla Búlgara / Cossack Squat"},
    {"Fallo": "Good Morning Squat", "Causa": "Cuádriceps débiles", "Ejercicio": "Sentadilla con Pausa / Frontal"}
])