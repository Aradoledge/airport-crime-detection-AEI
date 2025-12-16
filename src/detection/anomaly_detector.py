import tensorflow as tf
import numpy as np
import cv2
from collections import deque
import config.settings as settings


class AnomalyDetector:
    def __init__(self, model_path=None):
        self.model = None
        self.sequence_length = settings.SEQUENCE_LENGTH
        self.img_size = (settings.IMG_HEIGHT, settings.IMG_WIDTH)
        self.frame_buffer = deque(maxlen=self.sequence_length)

        if model_path:
            self.load_model(model_path)
        else:
            # Try to load from default path
            default_path = (
                settings.MODELS_DIR / "trained" / "airport_anomaly_detector_final.h5"
            )
            if default_path.exists():
                self.load_model(default_path)

    def load_model(self, model_path):
        """Load a pre-trained CNN-LSTM model"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Anomaly detection model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading anomaly detection model: {e}")
            print("Running in YOLO-only mode (no behavioral anomaly detection)")
            self.model = None

    def preprocess_frame(self, frame):
        """Preprocess a single frame for the model"""
        # Resize and normalize
        frame = cv2.resize(frame, self.img_size)
        frame = frame.astype(np.float32) / 255.0
        return frame

    def detect_anomaly(self, frame):
        """Detect anomaly in the current frame using sequence analysis"""
        if self.model is None:
            return 0.0  # Return 0 if model not loaded

        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)

        # Add to buffer
        self.frame_buffer.append(processed_frame)

        # Check if we have enough frames for a sequence
        if len(self.frame_buffer) < self.sequence_length:
            return 0.0

        # Create sequence
        sequence = np.array(self.frame_buffer)
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension

        try:
            # Predict
            prediction = self.model.predict(sequence, verbose=0)
            anomaly_score = float(
                prediction[0][0]
            )  # Assuming binary classification [0,1]
            return anomaly_score
        except Exception as e:
            print(f"Error in anomaly prediction: {e}")
            return 0.0

    def reset_buffer(self):
        """Reset the frame buffer"""
        self.frame_buffer.clear()

    def get_buffer_status(self):
        """Get the current buffer status"""
        return len(self.frame_buffer), self.sequence_length
