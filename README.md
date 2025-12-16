## 🎯 Project Overview

This project implements an **AI-driven crime detection system** specifically designed for airport security environments. The system leverages **deep learning techniques** combining **Convolutional Neural Networks (CNNs)** and **Long Short-Term Memory (LSTM)** networks for spatial-temporal anomaly detection, integrated with **YOLO (You Only Look Once)** for real-time object detection.

### Key Features:
- **Real-time Anomaly Detection**: Identifies suspicious behaviors (unattended luggage, running, loitering)
- **Object Detection**: YOLO-based detection of persons, luggage, and potential threats
- **Hybrid CNN-LSTM Architecture**: Combines spatial and temporal analysis
- **Interactive Dashboard**: Streamlit-based monitoring interface
- **Multi-source Input**: Webcam, video files, and CCTV integration

## 🏗️ System Architecture

### Technical Stack:
- **Backend**: Python 3.8+, TensorFlow 2.11+, PyTorch
- **Computer Vision**: OpenCV, YOLOv8
- **Deep Learning**: CNN (VGG16/ResNet50) + LSTM
- **Dashboard**: Streamlit, Streamlit-WebRTC
- **Data Processing**: NumPy, Pandas, Albumentations

### Architecture Diagram:
```
┌─────────────────────────────────────────────────────────────┐
│                    Video Input Sources                       │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│     │  Webcam  │ │ CCTV     │ │ Video    │ │ Test     │    │
│     │          │ │ Streams  │ │ Files    │ │ Videos   │    │
│     └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│                   │                                        │
│                   ▼                                        │
│         ┌─────────────────────────┐                        │
│         │   Frame Preprocessing   │                        │
│         │  • Resize (224×224)     │                        │
│         │  • Normalization        │                        │
│         │  • Augmentation         │                        │
│         └─────────────────────────┘                        │
│                   │                                        │
│         ┌─────────────────────────┐ ┌───────────────────┐ │
│         │    YOLO Detection       │ │   CNN-LSTM Model  │ │
│         │  • Object Detection     │ │  • Feature Extrac │ │
│         │  • Bounding Boxes       │ │  • Temporal Analy │ │
│         │  • Threat Classification│ │  • Anomaly Score  │ │
│         └─────────────────────────┘ └───────────────────┘ │
│                   │                     │                  │
│                   └─────────────────────┘                  │
│                               │                            │
│                   ┌─────────────────────┐                  │
│                   │   Decision Fusion   │                  │
│                   │  • Alert Generation │                  │
│                   │  • Confidence Score │                  │
│                   └─────────────────────┘                  │
│                               │                            │
│         ┌─────────────────────────────────────────┐        │
│         │          Output & Visualization         │        │
│         │  • Annotated Video      • Dashboard     │        │
│         │  • Log Files            • Real-time     │        │
│         │  • Alert Notifications    Alerts        │        │
│         └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ Prerequisites

### Hardware Requirements:
- **Minimum**: CPU with AVX support, 8GB RAM, 10GB free disk space
- **Recommended**: NVIDIA GPU (4GB+ VRAM), 16GB RAM, 50GB SSD
- **Optimal**: NVIDIA RTX 3060+ (for real-time processing), 32GB RAM

### Software Requirements:
- **Operating System**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.2+ (for GPU acceleration)
- **cuDNN**: 8.1+ (for GPU acceleration)

## 📥 Installation Guide

### For Windows Users:

#### 1. Install Python and Dependencies
```powershell
# Download the latest version of Python from python.org
# During installation, check "Add Python to PATH"

# Open Command Prompt as Administrator
python --version  # Should show Python 3.12 or a version above

# Install Git (if not installed)
# Download from https://git-scm.com/download/win
```

#### 2. Clone the Repository
```powershell
# Open PowerShell or Command Prompt
cd C:\Users\YourName\Projects
git clone https://github.com/Aradoledge/airport-crime-detection-AEI
cd airport-crime-detection-AEI
```

#### 3. Create Virtual Environment
```powershell
# Create virtual environment
python -m venv myenv

# Activate virtual environment
.\myenv\Scripts\activate

# Your prompt should change to show (myenv)
```

#### 4. Install System Dependencies
```powershell
# Install Visual C++ Build Tools (for some packages)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install CMake (for dlib/face_recognition)
# Download from: https://cmake.org/download/
```

#### 5. Install Python Packages
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# If you have CUDA-enabled GPU, install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Install PyTorch with CUDA (visit https://pytorch.org/get-started/locally/)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### For Linux Users (Ubuntu/Debian):

#### 1. System Updates and Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3.9 python3.9-venv python3-pip git cmake
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# For GPU support (NVIDIA)
sudo apt install -y nvidia-cuda-toolkit nvidia-driver-525
```

#### 2. Clone and Setup
```bash
# Clone repository
cd ~/Projects
git clone https://github.com/Aradoledge/airport-crime-detection-AEI
cd airport-crime-detection-AEI

# Create virtual environment
python3.9 -m venv myenv
source myenv/bin/activate
```

#### 3. Install Python Packages
```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio
```

### For macOS Users:

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and dependencies
brew install python@3.9 git cmake
brew install libomp

# Clone and setup
git clone https://github.com/Aradoledge/airport-crime-detection-AEI
cd airport-crime-detection-AEI
python3.9 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Option 1: Using Real UCF-Crime Dataset (Recommended)

#### 1. Download UCF-Crime Dataset
```bash
# Download from: https://www.dropbox.com/scl/fo/2aczdnx37hxvcfdo4rq4q/AOjRokSTaiKxXmgUyqdcI6k?rlkey=5bg7mxxbq46t7aujfch46dlvz&e=3&dl=0

# Extract to correct directory
mkdir -p data/raw/ucf_crime
# Copy downloaded videos to: data/raw/ucf_crime/
```

#### 2. Prepare Dataset Structure
```bash
# Expected structure:
data/raw/ucf_crime/
├── Normal/
│   └── Normal_Videos_xxx.mp4
└── Anomaly/
    ├── Abuse/
    │   └── Abusexxx_x264.mp4
    ├── Arrest/
    │   └── Arrestxxx_x264.mp4
    └── ... (other anomaly categories)
```

#### 3. Extract Frames
```bash
# Extract sample dataset (for testing)
python main.py extract --sample --sample_size 20

# Extract full dataset (will take time)
python main.py extract
```

### Option 2: Generate Fake Dataset (For Testing)

```bash
# Generate synthetic dataset
python main.py generate-fake --num_normal 10 --num_anomaly 10

# Extract frames from fake dataset
python main.py extract --fake
```

### Dataset Statistics
```bash
# Check dataset statistics
python main.py stats

# Expected output:
# Normal frames: XXXX
# Anomaly frames: XXXX
# Total frames: XXXX
```

## 🚀 Model Training

### 1. Configuration Setup
Edit `config/settings.py` to adjust parameters:
```python
# Training parameters
SEQUENCE_LENGTH = 20      # Number of frames per sequence
IMG_HEIGHT = 224          # Image height
IMG_WIDTH = 224           # Image width
BATCH_SIZE = 8            # Batch size (adjust based on GPU memory)
NUM_EPOCHS = 50           # Number of training epochs
LEARNING_RATE = 0.001     # Learning rate

# Model architecture
CNN_BACKBONE = "vgg16"    # Options: "vgg16", "resnet50", "inception_v3"
LSTM_UNITS = 128          # Number of LSTM units
DROPOUT_RATE = 0.5        # Dropout rate
```

### 2. Start Training Process
```bash
# Basic training
python main.py train --epochs 20 --batch_size 8

# Advanced training with monitoring
python src/train.py --epochs 50 --batch_size 16 --sequence_length 25

# Training with specific backbone
# Edit config/settings.py to change CNN_BACKBONE
```

### 3. Monitor Training Progress
```bash
# TensorBoard monitoring
tensorboard --logdir logs/

# Access at: http://localhost:6006

# Monitor training logs
tail -f training.log
```

### 4. Training Output
```bash
# Trained models are saved in:
models/trained/
├── airport_anomaly_detector_best.h5    # Best model
├── airport_anomaly_detector_final.h5   # Final model
└── checkpoint/                         # Checkpoints

# Training logs in:
logs/
└── airport_anomaly_detector_YYYYMMDD_HHMMSS/
    ├── train/                          # Training metrics
    └── validation/                     # Validation metrics
```

### 5. Training Performance Optimization
```bash
# For GPU training verification
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If GPU is detected, TensorFlow will automatically use it
# For mixed precision training (faster, less memory):
# Add to config/settings.py:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

## 🧪 Testing & Evaluation

### 1. Model Evaluation
```bash
# Evaluate trained model
python evaluate.py

# Expected output includes:
# - Accuracy, Precision, Recall, F1-Score
# - Confusion Matrix
# - ROC Curve
```

### 2. Real-time Detection Testing

#### With Webcam:
```bash
python main.py detect --source webcam
# Press 'q' to quit, 'p' to pause
```

#### With Video File:
```bash
python main.py detect --source "data/raw/ucf_crime/Normal/Normal_Videos_015_x264.mp4"

# With output recording
python main.py detect --source "input_video.mp4" --output "output_detection.avi"
```

<!-- #### Batch Testing:
```bash
# Test on multiple videos
python test_detection.py

# Performance benchmark
python benchmark.py
``` -->

### 3. Detection Controls
During detection, use these keyboard controls:
- **Q**: Quit application
- **P**: Pause/Resume
- **S**: Save current frame
- **+**: Increase confidence threshold
- **-**: Decrease confidence threshold

## 📊 Streamlit Dashboard

### 1. Local Dashboard Setup
```bash
# Start the dashboard
python main.py dashboard

# Or directly
streamlit run app.py

# Dashboard will open at: http://localhost:8501
```

### 2. Dashboard Features

#### Main Interface:
- **Live Camera Feed**: Real-time video processing
- **Detection Panel**: Object and anomaly detection results
- **Alert System**: Real-time security alerts
- **Statistics Dashboard**: Performance metrics
- **Historical Logs**: Past detection records

#### Configuration Panel:
```python
# Accessible from sidebar:
- Confidence Threshold: 0.1 - 1.0
- Anomaly Threshold: 0.1 - 1.0
- Enable Audio Alerts: True/False
- Model Selection: Choose between trained models
```

### 3. WebRTC Integration
For real-time browser-based camera access:
```bash
# Ensure WebRTC dependencies are installed
pip install streamlit-webrtc aiortc

# Start with WebRTC support
streamlit run app.py
```

<!-- ### 4. Dashboard Testing
```bash
# Test without camera (mock data)
python test_dashboard.py

# Test with sample video
streamlit run app.py --server.address=0.0.0.0 --server.port=8501
``` -->

## 🚀 Deployment Guide

### 1. Streamlit Cloud Deployment

#### Prepare for Deployment:
```bash
# Create requirements.txt for deployment
pip freeze > requirements.txt

# Create .streamlit/config.toml
mkdir -p .streamlit
cat > .streamlit/config.toml << EOF
[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
EOF
```

#### Deploy to Streamlit Cloud:
1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repository
4. Set main file path to `app.py`
5. Click "Deploy"





## 🧠 Methodology

### 1. Hybrid CNN-LSTM Architecture

#### Spatial Feature Extraction (CNN):
```python
# Using pre-trained VGG16 as feature extractor
base_model = VGG16(weights='imagenet', include_top=False)
# Feature maps are extracted from each frame
# Output: 512-dimensional feature vector per frame
```

#### Temporal Sequence Analysis (LSTM):
```python
# LSTM processes sequence of CNN features
lstm_layer = LSTM(128, return_sequences=True)
# Captures temporal dependencies between frames
# Identifies behavioral patterns over time
```

### 2. Anomaly Detection Algorithm

#### Sequence Processing:
```
Input: Video Sequence (20 frames)
Step 1: Frame extraction and preprocessing
Step 2: CNN feature extraction per frame
Step 3: LSTM sequence analysis
Step 4: Anomaly score calculation
Step 5: Threshold comparison
Output: Normal (0) / Anomaly (1) classification
```

#### Mathematical Formulation:
```
Let S = {f₁, f₂, ..., fₙ} be a sequence of frames
Let φ be the CNN feature extractor
Let L be the LSTM model

Features: X = [φ(f₁), φ(f₂), ..., φ(fₙ)]
Temporal Analysis: H = L(X)
Anomaly Score: y = σ(W·H + b)
Decision: anomaly if y > threshold
```

### 3. YOLO Integration

#### Object Detection Pipeline:
```
1. Frame input from video source
2. YOLO object detection (persons, luggage, etc.)
3. Bounding box extraction and classification
4. Feature extraction from detected regions
5. Integration with CNN-LSTM anomaly scores
```

### 4. Multi-modal Fusion

#### Decision Fusion Strategy:
```python
# Combine YOLO and CNN-LSTM outputs
final_score = α * yolo_confidence + β * anomaly_score
# Where α + β = 1, tuned based on validation
```

## 📈 Performance Metrics

### 1. Evaluation Metrics
```python
# Standard metrics used:
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
- ROC-AUC: Area under ROC curve
- Confusion Matrix: Visual classification performance
```

### 2. Expected Performance
```
With UCF-Crime Dataset:
- Accuracy: 85-92%
- Precision: 87-90%
- Recall: 83-88%
- F1-Score: 85-89%
- Inference Time: 30-50ms per frame (GPU)
```



## 🔧 Troubleshooting

### Common Issues and Solutions:

#### 1. CUDA/CuDNN Errors
```bash
# Check CUDA installation
nvidia-smi

# Verify TensorFlow GPU access
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reinstall with specific versions
pip install tensorflow==2.11.0
pip install nvidia-cudnn-cu11==8.6.0.163
```

#### 2. Memory Issues
```python
# Reduce batch size in config/settings.py
BATCH_SIZE = 4  # Instead of 8

# Use mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

#### 3. Webcam Access Issues
```bash
# Linux: Check permissions
sudo usermod -a -G video $USER

# Windows: Update camera drivers
# Check Device Manager → Imaging devices

# Test with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

#### 4. Streamlit Deployment Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Check port availability
netstat -ano | findstr :8501  # Windows
lsof -i :8501                 # Linux/Mac

# Run on different port
streamlit run app.py --server.port=8502
```


## 📚 References

### Tools and Libraries:
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)


**Note**: This system is designed for research and educational purposes. Always comply with local laws and regulations regarding surveillance and privacy when deploying in real-world scenarios.
