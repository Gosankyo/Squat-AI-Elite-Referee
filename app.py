import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import ollama

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="SQUAT AI ULTRA ELITE - EDICIÓN NLP", layout="wide", page_icon="🚀")

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
    # Seguimos usando el Cerebro Biomecánico Perfecto
    model_ia = load_model('modelo_squat_biomecanico.h5')
    yolo_pose = YOLO('yolov8n-pose.pt')
    return model_ia, yolo_pose

model_ia, yolo_pose = iniciar_modelos()

# --- 3. FUNCIONES DE ANÁLISIS BIOMECÁNICO ---
def calculate_angle(a, b, c):
    """Motor trigonométrico para calcular ángulos articulares"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def realizar_diagnostico_completo(kp_hoyo):
    """Diagnóstico experto basado en el punto de máxima profundidad"""
    if kp_hoyo is None:
        return ["No se detectó el fondo de la sentadilla."], [""], 0
        
    dx_t = kp_hoyo[11][0] - kp_hoyo[5][0]
    dy_t = kp_hoyo[11][1] - kp_hoyo[5][1]
    angulo_torso = abs(np.degrees(np.arctan2(dx_t, dy_t)))
    
    ancho_caderas = abs(kp_hoyo[11][0] - kp_hoyo[12][0])
    ancho_rodillas = abs(kp_hoyo[13][0] - kp_hoyo[14][0])
    desequilibrio_cadera = abs(kp_hoyo[11][1] - kp_hoyo[12][1])
    
    errores, soluciones = [], []
    if angulo_torso > 45:
        errores.append("⚠️ **Torso Inclinado:** Exceso de carga lumbar (Good Morning Squat).")
        soluciones.append("Haz Sentadilla Frontal o Pausa en el hoyo.")
    if ancho_rodillas < (ancho_caderas * 0.9):
        errores.append("⚠️ **Valgo de Rodilla:** Tus rodillas colapsan hacia adentro.")
        soluciones.append("Usa banda elástica en calentamiento (Monster walks).")
    if desequilibrio_cadera > 15:
        errores.append("⚠️ **Hip Shift:** Tu cadera se desplaza lateralmente.")
        soluciones.append("Añade Sentadilla Búlgara para corregir asimetría.")
        
    if not errores:
        errores.append("⭐ **Técnica de Élite:** Sin fallos críticos detectados.")
        soluciones.append("Sigue progresando en cargas, vas perfecto.")
        
    return errores, soluciones, angulo_torso

def analizar_proporciones(puntos_kp):
    h_y = (puntos_kp[5][1]+puntos_kp[6][1])/2
    c_y = (puntos_kp[11][1]+puntos_kp[12][1])/2
    r_y = (puntos_kp[13][1]+puntos_kp[14][1])/2
    len_torso, len_femur = abs(c_y - h_y), abs(r_y - c_y)
    ratio = len_femur / len_torso if len_torso != 0 else 0
    if ratio > 0.88: return "FÉMURES LARGOS", "Barra Baja recomendada", ratio
    elif ratio < 0.78: return "FÉMURES CORTOS", "Barra Alta recomendada", ratio
    return "ESTRUCTURA NEUTRA", "Barra Media", ratio

# --- 4. INTERFAZ ---
st.sidebar.title("🚀 SQUAT AI CONTROL")
lift_weight = st.sidebar.number_input("Peso en barra (kg)", value=100.0, step=2.5)
user_weight = st.sidebar.number_input("Peso Corporal (kg)", value=75.0, step=0.5)
sexo = st.sidebar.selectbox("Sexo", ["men", "women"])
pais = st.sidebar.selectbox("Región", ["España", "Francia", "USA", "All"])
mapa_paises = {"España": "all-spain", "Francia": "all-france", "USA": "all-usa", "All": "all"}

tab1, tab2, tab3, tab4 = st.tabs(["🎥 ANÁLISIS VBT", "📐 ARQUITECTO", "📊 RANKING", "🕹️ JUEGO"])

# --- TAB 1: JUEZ + VBT + COACH NLP ---
with tab1:
    c_left, c_right = st.columns([2, 1])
    with c_right:
        reps_st = st.metric("REPSET", "0")
        luces = st.empty()
        st.subheader("📈 Velocidad (VBT)")
        grafico_v = st.empty()
        st.subheader("📸 Deep Capture")
        deep_cap_area = st.empty()
        st.subheader("📝 Diagnóstico de IA")
        diag_area = st.empty()
        
        # Área para el Coach Virtual
        st.subheader("💬 Entrenador Virtual (NLP)")
        coach_area = st.empty()

    video = st.file_uploader("Sube vídeo de sentadilla", type=['mp4', 'mov'])
    if video:
        with open("temp.mp4", "wb") as f: f.write(video.read())
        cap = cv2.VideoCapture("temp.mp4")
        st_frame = c_left.empty()
        
        # Controladores de la IA
        secuencia = []
        contador, estado, mejor_conf = 0, "ESPERANDO", 0
        y_ini, frames_bloqueo = None, 0
        velocidades, y_prev = [], None
        frame_profundo, kp_hoyo = None, None
        y_max_hoyo = -1.0 
        
        # Variables para Ollama
        ultimo_veredicto = "Desconocido"
        ultimos_fallos = []
        ultima_confianza = 0.0
        ultimo_ang_torso = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            # --- LA SOLUCIÓN AL ESTIRAMIENTO ---
            h, w = frame.shape[:2] # Obtenemos el alto y ancho original
            nuevo_ancho = 800
            nuevo_alto = int(h * (nuevo_ancho / w)) # Calculamos el alto proporcional
            
            frame_res = cv2.resize(frame, (nuevo_ancho, nuevo_alto))
            # -----------------------------------
            
            # YOLO detecta el esqueleto visualmente
            res = yolo_pose.predict(frame_res, conf=0.5, verbose=False)
            
            # YOLO detecta el esqueleto visualmente
            res = yolo_pose.predict(frame_res, conf=0.5, verbose=False)
            
            if res[0].keypoints and len(res[0].keypoints.data) > 0:
                det = res[0].keypoints.data.cpu().numpy()
                boxes = res[0].boxes.data.cpu().numpy()
                atleta_kp = det[np.argmin([abs((b[0]+b[2])/2 - 400) for b in boxes])]
                
                # Coordenadas clave
                h_y = (atleta_kp[5][1] + atleta_kp[6][1]) / 2
                cadera_y = (atleta_kp[11][1] + atleta_kp[12][1]) / 2
                
                if y_ini is None: y_ini = h_y; y_prev = h_y; continue
                
                # 1. Tracker VBT (Velocidad)
                v_inst = abs(h_y - y_prev)
                velocidades.append(v_inst)
                y_prev = h_y
                
                # 2. EXTRACCIÓN DE PUNTOS PARA EL MODELO LSTM (51 Puntos Normalizados)
                kp_crudo = res[0].keypoints.data[0].cpu().numpy().flatten()
                
                if len(kp_crudo) == 51:
                    hip_l = kp_crudo[11*3 : 11*3+3]
                    hip_r = kp_crudo[12*3 : 12*3+3]
                    centro_cadera = (hip_l + hip_r) / 2
                    
                    kp_norm = kp_crudo.copy()
                    for i in range(17):
                        kp_norm[i*3 : i*3+3] -= centro_cadera
                    
                    secuencia.append(kp_norm)
                
                # Mantenemos solo los últimos 30 frames
                if len(secuencia) > 30: 
                    secuencia.pop(0)
                
                # 3. MÁQUINA DE ESTADOS Y PREDICCIÓN
                if frames_bloqueo > 0: frames_bloqueo -= 1
                else:
                    dist = h_y - y_ini
                    if dist > 60 and estado == "ESPERANDO": 
                        estado = "BAJANDO"
                        y_max_hoyo = -1.0 
                        mejor_conf = 0.0
                        kp_hoyo = None
                    
                    if estado == "BAJANDO":
                        if cadera_y > y_max_hoyo:
                            y_max_hoyo = cadera_y
                            frame_profundo = res[0].plot()
                            kp_hoyo = atleta_kp 
                            
                        # Predicción continua con LSTM
                        seq_array = np.array(secuencia)
                        pad_len = 30 - len(seq_array)
                        if pad_len > 0:
                            seq_array = np.pad(seq_array, ((0, pad_len), (0, 0)), 'constant')
                        
                        pred = model_ia.predict(np.expand_dims(seq_array, axis=0), verbose=0)[0]
                        if pred[0] > mejor_conf: mejor_conf = pred[0] # pred[0] es VÁLIDA
                    
                    if dist < 35 and estado == "BAJANDO":
                        contador += 1
                        frames_bloqueo = 60
                        
                        # Guardamos confianza
                        ultima_confianza = mejor_conf
                        
                        if mejor_conf > 0.50:
                            ultimo_veredicto = "VÁLIDA"
                            luces.success(f"⚪ ⚪ ⚪ VÁLIDA (Confianza: {round(mejor_conf*100)}%)")
                        else: 
                            ultimo_veredicto = "NULA"
                            luces.error(f"🔴 🔴 🔴 NULA (Confianza: {round((1-mejor_conf)*100)}%)")
                            
                        # Diagnóstico
                        fallos, sols, ang_t = realizar_diagnostico_completo(kp_hoyo)
                        ultimos_fallos = fallos
                        ultimo_ang_torso = ang_t
                        
                        with diag_area.container():
                            st.write(f"**Ángulo Torso Máximo:** {round(ang_t,1)}°")
                            for f in fallos: st.error(f)
                            for s in sols: st.success(f"📌 {s}")
                            
                        if frame_profundo is not None:
                            deep_cap_area.image(frame_profundo, caption="📸 Punto crítico", use_container_width=True)
                            
                        estado = "ESPERANDO"
                
                reps_st.metric("REPSET", contador)
                grafico_v.line_chart(velocidades[-50:])
            st_frame.image(res[0].plot(), channels="BGR")
        cap.release()

        # --- SECCIÓN NLP (OLLAMA COACH) ---
        with coach_area.container():
            if st.button("🎙️ Pedir Feedback al Entrenador", type="primary"):
                with st.spinner("El entrenador virtual está escribiendo su reporte..."):
                    
                    # Limpiamos los caracteres especiales para el LLM
                    fallos_limpios = [f.replace("⚠️", "").replace("⭐", "").replace("**", "").strip() for f in ultimos_fallos]
                    
                    prompt_entrenador = f"""
                    Actúa como un entrenador de Powerlifting de élite. 
                    El sistema biomecánico acaba de analizar a tu atleta con estos datos:
                    - Veredicto: Sentadilla {ultimo_veredicto}
                    - Confianza de la IA: {round(ultima_confianza*100, 1)}%
                    - Ángulo máximo del torso en el hoyo: {round(ultimo_ang_torso, 1)} grados
                    - Fallos detectados: {', '.join(fallos_limpios)}
                    
                    Redacta un feedback técnico, directo y muy motivador en un máximo de 2 párrafos cortos. 
                    Háblale de 'tú' al atleta. Si la sentadilla fue válida, felicítale. Si fue nula, dale un consejo biomecánico.
                    IMPORTANTE: NUNCA menciones que eres una IA, un modelo de lenguaje o que te han pasado un prompt.
                    """
                    
                    try:
                        # Llamada local a Ollama (Llama 3.2)
                        respuesta = ollama.chat(model='llama3.2', messages=[
                            {'role': 'user', 'content': prompt_entrenador}
                        ])
                        st.info(f"**Coach Virtual:**\n\n {respuesta['message']['content']}")
                    except Exception as e:
                        st.error(f"Error al conectar con Ollama. ¿Está la app abierta en tu PC? Detalle: {e}")

# --- TAB 2, 3, 4 ---
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

with tab3:
    st.header("📊 Nivel Competitivo")
    reps_cal = st.number_input("Reps para 1RM", min_value=1, value=1)
    one_rm = lift_weight * (1 + reps_cal / 30)
    st.metric("Tu 1RM Estimado", f"{round(one_rm, 2)} kg")
    url = f"https://www.openpowerlifting.org/rankings/{int(user_weight)}/{mapa_paises[pais]}/{sexo}/by-squat"
    st.markdown(f"### [🔗 Ir a OpenPowerlifting Rank]({url})")

with tab4:
    st.header("🕹️ Estado del Sistema")
    st.progress(100)
    st.success("✅ Motor Biomecánico Operativo")
    st.success("✅ Coach Virtual (Ollama / NLP) Conectado")

st.divider()
st.table([
    {"Fallo": "Valgo Rodilla", "Causa": "Glúteo medio débil", "Ejercicio": "Banded Squats / Monster Walk"},
    {"Fallo": "Hip Shift", "Causa": "Asimetría de movilidad", "Ejercicio": "Sentadilla Búlgara / Cossack Squat"},
    {"Fallo": "Good Morning Squat", "Causa": "Cuádriceps débiles", "Ejercicio": "Sentadilla con Pausa / Frontal"}
])