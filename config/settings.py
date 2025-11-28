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

# Model parameters
SEQUENCE_LENGTH = 20
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 8
NUM_EPOCHS = 50

# CNN-LSTM Model parameters
CNN_BACKBONE = "vgg16"  # Options: "vgg16", "resnet50", "inception_v3"
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001

# YOLO parameters
YOLO_MODEL = "yolov8n.pt"  # nano version for speed
YOLO_CONFIDENCE = 0.5

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LABELS_DIR, 
                  FRAMES_DIR, SEQUENCES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Class names
CLASS_NAMES = ["normal", "anomaly"]