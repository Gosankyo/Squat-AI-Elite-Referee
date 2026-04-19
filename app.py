import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from tensorflow.keras.models import load_model
import ollama

# Configuration for Streamlit interface
st.set_page_config(page_title="Squat AI Biomechanical Analysis", layout="wide", page_icon="🚀")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e0e0e0; }
    .stMetric { background-color: #161b22; border: 1px solid #3b82f6; box-shadow: 0px 0px 10px #3b82f6; border-radius: 15px; padding: 10px; }
    .stTabs [data-baseweb="tab"] { font-size: 20px; font-weight: bold; color: #8b949e; }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom: 3px solid #3b82f6; }
    h1, h2, h3 { color: #58a6ff; }
    </style>
    """, unsafe_allow_html=True)

# Load machine learning models
@st.cache_resource
def load_models():
    """Load YOLO pose estimation and LSTM biomechanical classification models."""
    lstm_model = load_model('modelo_squat_biomecanico.h5')
    yolo_model = YOLO('yolov8n-pose.pt')
    return lstm_model, yolo_model

model_lstm, model_yolo = load_models()

# Biomechanical analysis functions
def perform_full_diagnosis(keypoints_at_depth):
    """Biomechanical diagnosis at maximum depth position."""
    if keypoints_at_depth is None:
        return ["Bottom position not detected."], [""], 0, False
        
    # --- [FIX] INVARIANT ANGLE: Calculate from center of shoulders to center of hips ---
    shoulder_center_x = (keypoints_at_depth[5][0] + keypoints_at_depth[6][0]) / 2.0
    shoulder_center_y = (keypoints_at_depth[5][1] + keypoints_at_depth[6][1]) / 2.0
    hip_center_x = (keypoints_at_depth[11][0] + keypoints_at_depth[12][0]) / 2.0
    hip_center_y = (keypoints_at_depth[11][1] + keypoints_at_depth[12][1]) / 2.0

    dx_torso = hip_center_x - shoulder_center_x
    dy_torso = hip_center_y - shoulder_center_y
    torso_angle = abs(np.degrees(np.arctan2(dx_torso, dy_torso)))
    
    # Measure hip and knee width for valgus detection
    hip_width = abs(keypoints_at_depth[11][0] - keypoints_at_depth[12][0])
    knee_width = abs(keypoints_at_depth[13][0] - keypoints_at_depth[14][0])
    hip_shift = abs(keypoints_at_depth[11][1] - keypoints_at_depth[12][1])
    
    errors, solutions = [], []
    is_safe = True
    
    if torso_angle > 45:
        errors.append("[FORM] Excessive torso lean detected - spinal load risk.")
        solutions.append("Consider Front Squat or pause squats for form correction.")
        is_safe = False
    if knee_width < (hip_width * 0.9):
        errors.append("[FORM] Knee valgus detected - inward collapse pattern.")
        solutions.append("Implement elastic band training and single-leg work.")
        is_safe = False
    if hip_shift > 15:
        errors.append("[FORM] Lateral hip shift detected - asymmetrical loading.")
        solutions.append("Add unilateral exercises to correct imbalance.")
        is_safe = False
        
    if not errors:
        errors.append("[FORM] Excellent technique - no critical form deviations.")
        solutions.append("Continue progressive overload with current form pattern.")
        
    return errors, solutions, torso_angle, is_safe

def analyze_proportions(keypoints):
    """Analyze athlete anthropometry to recommend squat variant."""
    shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
    hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
    torso_length, femur_length = abs(hip_y - shoulder_y), abs(knee_y - hip_y)
    ratio = femur_length / torso_length if torso_length != 0 else 0
    if ratio > 0.88: 
        return "Long Femurs", "Low Bar Squat recommended", ratio
    elif ratio < 0.78: 
        return "Short Femurs", "High Bar Squat recommended", ratio
    return "Neutral Structure", "Moderate Bar Position", ratio

# Sidebar configuration
st.sidebar.title("System Configuration")
lift_weight = st.sidebar.number_input("Weight on bar (kg)", value=100.0, step=2.5)
user_weight = st.sidebar.number_input("Body Weight (kg)", value=75.0, step=0.5)
gender = st.sidebar.selectbox("Gender", ["men", "women"])
region = st.sidebar.selectbox("Region", ["Spain", "France", "USA", "All"])
country_mapping = {"Spain": "all-spain", "France": "all-france", "USA": "all-usa", "All": "all"}

tab1, tab2, tab3, tab4 = st.tabs(["Biomechanical Referee", "Anatomical Architect", "Competitive Ranking", "System Status"])

# Tab 1: Real-time video analysis with biomechanical referee
with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_right:
        rep_counter = st.metric("Completed Reps", "0")
        status_lights = st.empty()
        
        st.subheader("Biomechanical Scorecard")
        diagnosis_area = st.empty()
        
        st.subheader("Depth Geometry")
        depth_capture_area = st.empty()
        
        st.subheader("Virtual Coach")
        coach_area = st.empty()

    with col_left:
        st.info("Upload a side-angle squat video to begin the AI analysis workflow.")
        video = st.file_uploader("Upload squat video", type=['mp4', 'mov'], label_visibility="collapsed")
        frame_display = st.empty()

    if video:
        with open("temp.mp4", "wb") as f: 
            f.write(video.read())
        video_capture = cv2.VideoCapture("temp.mp4")
        
        # Video processing state variables
        keypoint_sequence = []
        rep_count, state, best_confidence = 0, "IDLE", 0
        y_initial, frame_lock = None, 0
        frame_at_depth, keypoints_at_depth = None, None
        y_max_depth = -1.0 
        
        # NLP coaching variables
        last_verdict = "Unknown"
        last_errors = []
        last_confidence = 0.0
        last_torso_angle = 0.0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret: 
                break
            
            # Resize frame for processing
            frame_height, frame_width = frame.shape[:2]
            new_width = 800
            new_height = int(frame_height * (new_width / frame_width))
            frame_resized = cv2.resize(frame, (new_width, new_height))
            
            # YOLO pose detection
            detection = model_yolo.predict(frame_resized, conf=0.5, verbose=False)
            
            if detection[0].boxes and detection[0].keypoints and len(detection[0].keypoints.data) > 0:
                keypoints_raw = detection[0].keypoints.data.cpu().numpy()
                boxes = detection[0].boxes.data.cpu().numpy()
                
                best_idx = np.argmin([abs((b[0]+b[2])/2 - (new_width/2)) for b in boxes])
                athlete_keypoints = keypoints_raw[best_idx]
                
                # --- [FIX] VISUAL TARGET MARKER ---
                annotated_frame = detection[0].plot()
                try:
                    # Draw a target reticle over the nose of the evaluated athlete
                    nose_x, nose_y = int(athlete_keypoints[0][0]), int(athlete_keypoints[0][1])
                    cv2.circle(annotated_frame, (nose_x, nose_y), 8, (0, 255, 0), -1) 
                    cv2.circle(annotated_frame, (nose_x, nose_y), 18, (0, 255, 255), 2)
                except Exception:
                    pass
                
                # Extract key body positions
                shoulder_y = (athlete_keypoints[5][1] + athlete_keypoints[6][1]) / 2
                hip_y = (athlete_keypoints[11][1] + athlete_keypoints[12][1]) / 2
                
                if y_initial is None: 
                    y_initial = shoulder_y
                    continue
                
                keypoints_raw_flat = detection[0].keypoints.data[best_idx].cpu().numpy().flatten()
                
                if len(keypoints_raw_flat) == 51:
                    hip_left = keypoints_raw_flat[11*3 : 11*3+3]
                    hip_right = keypoints_raw_flat[12*3 : 12*3+3]
                    hip_center = (hip_left + hip_right) / 2
                    
                    keypoints_normalized = keypoints_raw_flat.copy()
                    for i in range(17):
                        keypoints_normalized[i*3 : i*3+3] -= hip_center
                    
                    keypoint_sequence.append(keypoints_normalized)
                
                if len(keypoint_sequence) > 30: 
                    keypoint_sequence.pop(0)
                
                # State machine
                if frame_lock > 0: 
                    frame_lock -= 1
                else:
                    descent = shoulder_y - y_initial
                    if descent > 60 and state == "IDLE": 
                        state = "DESCENDING"
                        y_max_depth = -1.0 
                        best_confidence = 0.0
                        keypoints_at_depth = None
                        status_lights.info("Analyzing descent...")
                    
                    if state == "DESCENDING":
                        if hip_y > y_max_depth:
                            y_max_depth = hip_y
                            frame_at_depth = annotated_frame.copy() # Use the frame with the target marker
                            keypoints_at_depth = athlete_keypoints 
                            
                        sequence_array = np.array(keypoint_sequence)
                        pad_length = 30 - len(sequence_array)
                        if pad_length > 0:
                            sequence_array = np.pad(sequence_array, ((0, pad_length), (0, 0)), 'constant')
                        
                        prediction = model_lstm.predict(np.expand_dims(sequence_array, axis=0), verbose=0)[0]
                        if prediction[0] > best_confidence: 
                            best_confidence = prediction[0]
                    
                    if descent < 35 and state == "DESCENDING":
                        rep_count += 1
                        frame_lock = 60
                        
                        depth_achieved = best_confidence > 0.50
                        errors, solutions, torso_angle, is_safe = perform_full_diagnosis(keypoints_at_depth)
                        
                        last_confidence = best_confidence
                        last_torso_angle = torso_angle
                        last_errors = [e for e in errors if "Excellent" not in e]
                        
                        if not depth_achieved:
                            last_verdict = "INVALID - High Squat"
                            status_lights.warning(f"[FAIL] High Squat")
                        elif not is_safe:
                            last_verdict = "INVALID - Form Error"
                            status_lights.error(f"[FAIL] Form Deviation")
                        else:
                            last_verdict = "VALID"
                            status_lights.success(f"[PASS] Valid Rep")
                        
                        with diagnosis_area.container():
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Depth", "PASS" if depth_achieved else "FAIL")
                            c2.metric("Torso", f"{round(torso_angle)} degrees")
                            c3.metric("Safety", "SAFE" if is_safe else "RISK")
                            
                            st.markdown("---")
                            for error in errors: 
                                if "Excellent" in error:
                                    st.success(error)
                                else:
                                    st.error(error)
                            for solution in solutions: 
                                st.info(f"💡 {solution}")
                            
                        if frame_at_depth is not None and keypoints_at_depth is not None:
                            try:
                                # --- [FIX] INVARIANT OVERLAY: Draw based on center of body ---
                                s_x = int((keypoints_at_depth[5][0] + keypoints_at_depth[6][0]) / 2)
                                s_y = int((keypoints_at_depth[5][1] + keypoints_at_depth[6][1]) / 2)
                                h_x = int((keypoints_at_depth[11][0] + keypoints_at_depth[12][0]) / 2)
                                h_y = int((keypoints_at_depth[11][1] + keypoints_at_depth[12][1]) / 2)
                                
                                p_shoulder = (s_x, s_y)
                                p_hip = (h_x, h_y)
                                
                                cv2.line(frame_at_depth, p_shoulder, p_hip, (0, 255, 255), 4) # Yellow back line
                                cv2.line(frame_at_depth, p_hip, (p_hip[0], p_hip[1] - 100), (0, 0, 255), 2) # Red vertical reference
                                
                                cv2.putText(frame_at_depth, f"Torso Angle: {round(torso_angle)}", 
                                            (p_hip[0] + 20, p_hip[1] - 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            except Exception:
                                pass 
                                
                            depth_capture_area.image(frame_at_depth, caption="Kinematic Overlay at Maximum Depth", channels="BGR", use_container_width=True)
                            
                        state = "IDLE"
                
                rep_counter.metric("Completed Reps", rep_count)
                
                # Display the frame with the target marker
                frame_display.image(annotated_frame, channels="BGR")
                
            else:
                status_lights.warning("⚠️ Athlete lost. Ensure full body is visible.")
                if 'frame_resized' in locals():
                    frame_display.image(frame_resized, channels="BGR")

        video_capture.release()

        # NLP coaching section
        with coach_area.container():
            st.markdown("---")
            st.markdown("### NLP Coach Feedback")
            
            model_choice = st.selectbox(
                "Select Language Model:", 
                ["llama3.2", "qwen2.5"] 
            )
            
            if st.button(f"Request Feedback ({model_choice})", type="primary", use_container_width=True):
                with st.spinner(f"{model_choice} generating coaching report..."):
                    
                    error_text = ', '.join(last_errors) if last_errors else "None. Perfect execution."
                    
                    coach_prompt = f"""
                    Act as an elite Powerlifting coach providing direct technical feedback.
                    Athlete squat analysis results:
                    - Verdict: {last_verdict}
                    - Depth Confidence: {round(last_confidence*100, 1)}%
                    - Torso angle at depth: {round(last_torso_angle, 1)} degrees
                    - Form deviations detected: {error_text}
                    
                    Provide direct, technical coaching feedback in 2-3 sentences maximum.
                    Address the athlete directly. If valid, congratulate them. If invalid, provide specific biomechanical corrections.
                    Never mention you are an AI or language model.
                    """
                    
                    try:
                        response = ollama.chat(model=model_choice, messages=[
                            {'role': 'user', 'content': coach_prompt}
                        ])
                        st.info(f"**Coach Feedback ({model_choice}):**\n\n {response['message']['content']}")
                    except Exception as e:
                        st.error(f"Connection error. Ensure Ollama is running with: ollama pull {model_choice}")

# Tab 2: Anthropometric analysis
with tab2:
    st.header("Anthropometric Analysis")
    photo = st.file_uploader("Upload front-facing photo", type=['jpg', 'png'])
    if photo:
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), 1)
        
        img_height, img_width = img.shape[:2]
        img_width_new = 800
        img_height_new = int(img_height * (img_width_new / img_width))
        img_resized = cv2.resize(img, (img_width_new, img_height_new))
        
        detection_photo = model_yolo.predict(img_resized, conf=0.5, verbose=False)
        
        if detection_photo[0].boxes and detection_photo[0].keypoints and len(detection_photo[0].keypoints.data) > 0:
            
            # --- [FIX] CENTER ATHLETE IN PHOTO ---
            keypoints_raw_photo = detection_photo[0].keypoints.data.cpu().numpy()
            boxes_photo = detection_photo[0].boxes.data.cpu().numpy()
            
            best_idx_photo = np.argmin([abs((b[0]+b[2])/2 - (img_width_new/2)) for b in boxes_photo])
            athlete_kps_photo = keypoints_raw_photo[best_idx_photo]
            
            structure_type, recommendation, femur_ratio = analyze_proportions(athlete_kps_photo)
            
            # Draw target marker on photo
            annotated_photo = detection_photo[0].plot()
            try:
                p_nose_x, p_nose_y = int(athlete_kps_photo[0][0]), int(athlete_kps_photo[0][1])
                cv2.circle(annotated_photo, (p_nose_x, p_nose_y), 8, (0, 255, 0), -1)
                cv2.circle(annotated_photo, (p_nose_x, p_nose_y), 18, (0, 255, 255), 2)
            except Exception:
                pass
            
            col1, col2 = st.columns(2)
            col1.image(annotated_photo, channels="BGR", use_container_width=True)
            
            with col2:
                st.metric("Femur/Torso Ratio", round(femur_ratio, 2))
                st.success(f"**Anthropometric Type:** {structure_type}\n\n**Recommended Technique:** {recommendation}")
                
                st.markdown("---")
                if st.button("Generate Analysis (NLP)", type="primary"):
                    with st.spinner("Analyzing anthropometry..."):
                        
                        analysis_prompt = f"""
                        Act as an expert biomechanical analyst specializing in Powerlifting anthropometry.
                        Athlete measurement data from pose analysis:
                        - Femur/Torso Ratio: {round(femur_ratio, 2)}
                        - Structure Classification: {structure_type}
                        - Recommended Variant: {recommendation}
                        
                        Provide brief technical analysis (2-3 sentences) addressing the athlete directly.
                        Explain what their anthropometry means for squat performance, why the recommended variant suits them, and one specific advantage.
                        Never mention you are an AI or language model.
                        """
                        
                        try:
                            response = ollama.chat(model='llama3.2', messages=[
                                {'role': 'user', 'content': analysis_prompt}
                            ])
                            st.info(f"**Analysis Report:**\n\n {response['message']['content']}")
                        except Exception as e:
                            st.error("Connection error. Ensure Ollama is running.")

# Tab 3: Competitive benchmarking
with tab3:
    st.header("Competitive Benchmarking")
    estimated_reps = st.number_input("Reps performed", min_value=1, value=1)
    estimated_1rm = lift_weight * (1 + estimated_reps / 30)
    st.metric("Estimated 1RM", f"{round(estimated_1rm, 2)} kg")
    ranking_url = f"https://www.openpowerlifting.org/rankings/{int(user_weight)}/{country_mapping[region]}/{gender}/by-squat"
    st.markdown(f"### [View OpenPowerlifting Rankings]({ranking_url})")

# Tab 4: System diagnostics
with tab4:
    st.header("System Status")
    st.progress(100)
    st.success("[ONLINE] Biomechanical inference engine operational (YOLOv8 + LSTM)")
    st.success("[ONLINE] Natural language coaching module active (Ollama with local LLM)")
    
st.divider()

with st.expander("View Biomechanical Correction Matrix (Knowledge Base)"):
    st.table([
        {"Form Deviation": "Knee Valgus", "Root Cause": "Weak glute medius", "Corrective Exercise": "Banded squats, monster walks"},
        {"Form Deviation": "Hip Shift", "Root Cause": "Mobility asymmetry", "Corrective Exercise": "Bulgarian squat, Cossack squat"},
        {"Form Deviation": "Forward Lean", "Root Cause": "Weak quadriceps", "Corrective Exercise": "Pause squats, front squats"}
    ])