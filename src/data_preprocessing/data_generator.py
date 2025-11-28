import numpy as np
import tensorflow as tf
from keras.utils import Sequence
import cv2
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import config.settings as settings


class SequenceDataGenerator(Sequence):
    """
    Data Generator for creating sequences of frames for CNN-LSTM model
    """

    def __init__(
        self,
        data_dir,
        sequence_length=20,
        batch_size=8,
        img_size=(224, 224),
        shuffle=True,
        validation_split=0.2,
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

        # Collect all frame files and their labels
        self.frame_paths = []
        self.labels = []

        # Load frame paths and labels
        self._load_data()

        # Create sequences
        self.sequences, self.sequence_labels = self._create_sequences()

        # Split data
        self.train_sequences, self.val_sequences, self.train_labels, self.val_labels = (
            train_test_split(
                self.sequences,
                self.sequence_labels,
                test_size=validation_split,
                random_state=42,
                stratify=self.sequence_labels,
            )
        )

        # For training generator
        self.current_sequences = self.train_sequences
        self.current_labels = self.train_labels
        self.indices = np.arange(len(self.current_sequences))

        self.on_epoch_end()

    def _load_data(self):
        """Load frame paths and their corresponding labels"""
        print("Loading frame data...")

        # Process anomaly frames
        anomaly_dir = self.data_dir / "anomaly"
        if anomaly_dir.exists():
            for frame_file in anomaly_dir.rglob("*.jpg"):
                self.frame_paths.append(frame_file)
                self.labels.append(1)  # 1 for anomaly

        # Process normal frames
        normal_dir = self.data_dir / "normal"
        if normal_dir.exists():
            for frame_file in normal_dir.rglob("*.jpg"):
                self.frame_paths.append(frame_file)
                self.labels.append(0)  # 0 for normal

        print(f"Loaded {len(self.frame_paths)} frames")
        print(f"Anomaly frames: {sum(self.labels)}")
        print(f"Normal frames: {len(self.labels) - sum(self.labels)}")

    def _create_sequences(self):
        """Create sequences from individual frames"""
        sequences = []
        sequence_labels = []

        # Group frames by video
        video_frames = {}
        for frame_path, label in zip(self.frame_paths, self.labels):
            # Extract video name from frame path
            # Assuming path structure: frames/dataset/video_name/frame_xxxxxx.jpg
            video_name = frame_path.parent.name
            if video_name not in video_frames:
                video_frames[video_name] = {"paths": [], "labels": []}

            video_frames[video_name]["paths"].append(frame_path)
            video_frames[video_name]["labels"].append(label)

        # Create sequences for each video
        for video_name, data in video_frames.items():
            paths = sorted(data["paths"])
            labels = data["labels"]

            # Create sequences of specified length
            for i in range(
                0, len(paths) - self.sequence_length + 1, self.sequence_length
            ):
                sequence_paths = paths[i : i + self.sequence_length]
                sequence_label = max(
                    labels[i : i + self.sequence_length]
                )  # If any frame is anomaly, sequence is anomaly

                sequences.append(sequence_paths)
                sequence_labels.append(sequence_label)

        print(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        return sequences, sequence_labels

    def __len__(self):
        """Number of batches per epoch"""
        return len(self.current_sequences) // self.batch_size

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_indices = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        X = np.zeros(
            (
                self.batch_size,
                self.sequence_length,
                self.img_size[0],
                self.img_size[1],
                3,
            )
        )
        y = np.zeros((self.batch_size, 1))

        for i, batch_idx in enumerate(batch_indices):
            sequence_paths = self.current_sequences[batch_idx]
            sequence_frames = []

            for frame_path in sequence_paths:
                # Load and preprocess frame
                frame = cv2.imread(str(frame_path))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.img_size)
                frame = frame.astype(np.float32) / 255.0  # Normalize to [0,1]
                sequence_frames.append(frame)

            X[i] = np.array(sequence_frames)
            y[i] = self.current_labels[batch_idx]

        return X, y

    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def create_data_generators(self):
        """Create separate generators for train and validation"""
        train_generator = SequenceDataGenerator(
            self.data_dir,
            self.sequence_length,
            self.batch_size,
            self.img_size,
            shuffle=True,
        )

        # Set validation data
        val_generator = SequenceDataGenerator(
            self.data_dir,
            self.sequence_length,
            self.batch_size,
            self.img_size,
            shuffle=False,
        )
        val_generator.current_sequences = self.val_sequences
        val_generator.current_labels = self.val_labels
        val_generator.indices = np.arange(len(val_generator.current_sequences))

        return train_generator, val_generator, None  # Third return for test if needed


# Utility function to test the data generator
def test_data_generator():
    """Test the data generator"""
    generator = SequenceDataGenerator(
        settings.FRAMES_DIR, sequence_length=10, batch_size=4, img_size=(224, 224)
    )

    print(f"Number of batches: {len(generator)}")

    # Get one batch
    X_batch, y_batch = generator[0]
    print(f"Batch shape: {X_batch.shape}")
    print(f"Labels shape: {y_batch.shape}")
    print(f"Sample labels: {y_batch.flatten()}")

    return generator


if __name__ == "__main__":
    test_data_generator()
