import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

FRAMES_DIR = PROCESSED_DATA_DIR / "frames"
SEQUENCES_DIR = PROCESSED_DATA_DIR / "sequences"

# Model parameters - Optimized for real dataset
SEQUENCE_LENGTH = 16  # Slightly reduced for real videos
IMG_HEIGHT = 160  # Reduced for faster processing
IMG_WIDTH = 160  # Reduced for faster processing
BATCH_SIZE = 4  # Increase to 8 for better GPU utilization
NUM_EPOCHS = 20  # More epochs for real data

# CNN-LSTM Model parameters
CNN_BACKBONE = "vgg16"  # Good balance of accuracy and speed
LSTM_UNITS = 64  # Reduced for faster training
DROPOUT_RATE = 0.4  # Slight dropout to prevent overfitting
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

# YOLO parameters
YOLO_MODEL = "yolov8n.pt"  # nano version for speed
YOLO_CONFIDENCE = 0.5

# UCF-Crime specific settings
UCF_CRIME_CLASSES = {
    "normal": ["Normal"],
    "anomaly": [
        "Abuse",
        "Arrest",
        "Arson",
        "Assault",
        "Burglary",
        "Explosion",
        "Fighting",
        "RoadAccidents",
        "Robbery",
        "Shooting",
        "Shoplifting",
        "Stealing",
        "Vandalism",
    ],
}

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    LABELS_DIR,
    FRAMES_DIR,
    SEQUENCES_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Class names
CLASS_NAMES = ["normal", "anomaly"]
