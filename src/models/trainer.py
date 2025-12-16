import tensorflow as tf
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)
import numpy as np
import os
from datetime import datetime
import config.settings as settings


class ModelTrainer:
    def __init__(self, model, model_name="cnn_lstm"):
        self.model = model
        self.model_name = model_name
        self.history = None

    def setup_callbacks(self):
        """
        Setup training callbacks
        """
        # Create model directory
        model_dir = settings.MODELS_DIR / "trained"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create log directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = settings.LOGS_DIR / f"{self.model_name}_{timestamp}"

        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=str(model_dir / f"{self.model_name}_best.h5"),
                monitor="val_loss",
                save_best_only=True,
                mode="min",
                verbose=1,
            ),
            # Early stopping
            EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
            ),
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-7, verbose=1
            ),
            # TensorBoard
            TensorBoard(
                log_dir=str(log_dir),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
            ),
        ]

        return callbacks

    def train(
        self,
        train_generator,
        val_generator,
        steps_per_epoch,
        validation_steps,
        epochs=50,
    ):
        """
        Train the model
        """
        callbacks = self.setup_callbacks()

        print("Starting model training...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        print(f"Epochs: {epochs}")

        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1,
        )

        return self.history

    def save_model(self, filename=None):
        """
        Save the trained model
        """
        if filename is None:
            filename = f"{self.model_name}_final.h5"

        model_path = settings.MODELS_DIR / "trained" / filename
        self.model.save(str(model_path))
        print(f"Model saved to {model_path}")

    def plot_training_history(self):
        """
        Plot training history (to be implemented)
        """
        if self.history is None:
            print("No training history available")
            return
        pass
