import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
from src.models.trainer import ModelTrainer
from src.data_preprocessing.data_generator import SequenceDataGenerator
import config.settings as settings
from model import create_cnn_lstm_model
import argparse


def train_model(epochs=None, batch_size=None, sequence_length=None):
    """Train the CNN-LSTM model with given parameters"""

    # Use provided parameters or defaults from settings
    epochs = epochs or settings.NUM_EPOCHS
    batch_size = batch_size or settings.BATCH_SIZE
    sequence_length = sequence_length or settings.SEQUENCE_LENGTH

    print("Initializing Data Generators...")

    # Initialize data generators
    try:
        data_gen = SequenceDataGenerator(
            data_dir=settings.FRAMES_DIR,
            sequence_length=sequence_length,
            batch_size=batch_size,
            img_size=(settings.IMG_HEIGHT, settings.IMG_WIDTH),
        )

        # Create train/validation generators
        train_gen, val_gen, test_gen = data_gen.create_data_generators()

        print(f"Training sequences: {len(data_gen.train_sequences)}")
        print(f"Validation sequences: {len(data_gen.val_sequences)}")

        # Check if we have enough data
        if len(data_gen.train_sequences) == 0:
            print("ERROR: No training sequences found!")
            print(
                "Please make sure you have frames in data/processed/frames/normal and data/processed/frames/anomaly"
            )
            return

    except Exception as e:
        print(f"Error initializing data generators: {e}")
        return

    print("Creating CNN-LSTM Model...")

    try:
        # Create and compile model
        model = create_cnn_lstm_model()

        print("Model created successfully!")
        model.summary()

    except Exception as e:
        print(f"Error creating model: {e}")
        return

    print("Initializing Trainer...")

    try:
        # Initialize trainer
        trainer = ModelTrainer(model, "airport_anomaly_detector")

        # Calculate steps
        steps_per_epoch = max(1, len(data_gen.train_sequences) // batch_size)
        validation_steps = max(1, len(data_gen.val_sequences) // batch_size)

        print(f"Training configuration:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Validation steps: {validation_steps}")
        print(f"  Training sequences: {len(data_gen.train_sequences)}")
        print(f"  Validation sequences: {len(data_gen.val_sequences)}")

        # Train model
        print("Starting training...")
        history = trainer.train(
            train_gen, val_gen, steps_per_epoch, validation_steps, epochs=epochs
        )

        # Save final model
        trainer.save_model()

        print("Training completed successfully!")

        # Plot training history if available
        try:
            trainer.plot_training_history()
        except:
            print("Note: Could not plot training history")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function when called directly"""
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM model for anomaly detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=settings.NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=settings.BATCH_SIZE, help="Batch size"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=settings.SEQUENCE_LENGTH,
        help="Sequence length",
    )

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
    )


# Function to be called from main.py
def train_main(args=None):
    """Entry point for training from main.py"""
    if args is None:
        # Called directly, use defaults
        train_model()
    else:
        # Called from main.py with arguments
        train_model(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
