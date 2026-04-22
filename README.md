# Final Project: AI-Powered Biomechanical Squat Analysis

## Description

This project implements an advanced real-time biomechanical analysis system for squat validation using artificial intelligence, computer vision, and voice processing. The system detects form errors, provides immediate feedback, and recommends corrections based on biomechanical principles.

## Key Features

### 🚀 Real-Time Analysis
- Body pose detection using YOLOv8
- Biomechanical classification with LSTM model
- Analysis of angles and body proportions
- Automatic squat technique validation

### 🎯 Functionalities
- **Biomechanical Referee**: Live analysis with visual and audio feedback
- **Anatomical Architect**: Personalized recommendations based on body proportions
- **Competitive Ranking**: Comparison with powerlifting standards
- **Voice Interface**: Natural voice command control
- **Genetic Optimization**: Automatic model parameter improvement
  Note: The genetic algorithm module is used during model development and is not required for running the application.

### 📊 Biomechanical Analysis
- Knee valgus detection
- Torso lean measurement
- Symmetry and stability analysis
- Squat variant recommendations (high bar, low bar)

## System Requirements

- Python 3.10 (recommended)
- Compatible webcam
- Microphone for voice commands
- GPU recommended for better performance

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd proyecto-final-4
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models:**
   - `yolov8n-pose.pt` (included in repository)
   - `modelo_squat_biomecanico.h5` (included in repository)

4. **Install Ollama (optional for advanced analysis):**
   ```bash
   # Follow instructions at https://ollama.ai/
   ollama pull llama2
   ```

## Usage

### Live Demo
```bash
python demo_live.py
```
- Say "start analysis" to begin
- The system will analyze your squat form in real-time
- Receive voice and visual feedback
- ⚠️ Requires microphone and internet connection for voice commands.

### Web Application
```bash
streamlit run app.py
```
- Complete web interface with detailed analysis
- Weight, gender, and region configuration
- Multiple tabs for different analyses

### Model Training
Open `2_Entrenamiento_IA.ipynb` in Jupyter Notebook to:
- Re-train the LSTM model
- Optimize hyperparameters with genetic algorithms
- Validate system performance

## Project Structure

```
proyecto-final-4/
├── app.py                    # Streamlit web application
├── demo_live.py             # Real-time demo with voice
├── 2_Entrenamiento_IA.ipynb # Training notebook
├── ga_optimizer.py          # Genetic algorithm optimizer
├── knowledge_base.py        # Biomechanical expert system
├── cv_geometry.py           # Computer vision metrics
├── voice_trigger.py         # NLP voice assistant
├── modelo_squat_biomecanico.h5 # Trained LSTM model
├── yolov8n-pose.pt         # YOLO pose model
├── archive/                 # Training data
│   ├── Squat_Data/         # Valid/invalid squat dataset
│   └── Longest_Sequence.npy # Training sequences
└── README.md               # This file
```

## Technologies Used

- **Computer Vision**: OpenCV, YOLOv8
- **Artificial Intelligence**: TensorFlow/Keras, LSTM
- **Voice Processing**: pyttsx3, Custom NLP
- **Web Interface**: Streamlit
- **Optimization**: Genetic Algorithms
- **Expert System**: Custom biomechanical logic

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

For questions or suggestions, please open an issue in the repository.

---

**Note**: This system is an assistive tool and does not replace professional supervision from a qualified trainer. Always consult with a specialist before modifying your training technique.