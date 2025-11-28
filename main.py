#!/usr/bin/env python3
"""
Main entry point for the Airport Crime Detection System
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def generate_dataset(args):
    """Generate fake dataset"""
    print("Generating fake dataset...")
    from src.utils.generate_fake_dataset import FakeDatasetGenerator

    generator = FakeDatasetGenerator()
    generator.generate_dataset(num_normal=args.num_normal, num_anomaly=args.num_anomaly)


def extract_frames(args):
    """Extract frames from videos"""
    print("Extracting frames from videos...")
    from src.data_preprocessing.frame_extractor import extract_frames_from_fake_dataset

    extract_frames_from_fake_dataset()


def train_model(args):
    """Train the CNN-LSTM model"""
    print("Training CNN-LSTM model...")
    from src.train import train_main

    # Call the training function with arguments
    train_main(args)


def start_dashboard(args):
    """Start the Streamlit dashboard"""
    import subprocess
    import sys

    print("Starting Streamlit dashboard...")

    # Try the main dashboard first, fallback to simple version
    try:
        # Test if webrtc imports work
        from streamlit_webrtc import webrtc_streamer

        print("Using WebRTC dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"WebRTC dashboard failed: {e}")
        print("Falling back to simple dashboard...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "simple_dashboard.py"]
        )


def start_detection(args):
    """Start real-time detection"""
    print(f"Starting detection with source: {args.source}")

    # Import and run detection directly
    from detect import RealTimeDetector

    detector = RealTimeDetector()

    if args.source == "webcam":
        print("Starting webcam detection...")
        detector.process_webcam()
    else:
        print(f"Processing video file: {args.source}")
        detector.process_video(args.source, args.output)


def main():
    parser = argparse.ArgumentParser(description="AI Airport Crime Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate dataset command
    gen_parser = subparsers.add_parser("generate", help="Generate fake dataset")
    gen_parser.add_argument(
        "--num_normal", type=int, default=8, help="Number of normal videos"
    )
    gen_parser.add_argument(
        "--num_anomaly", type=int, default=8, help="Number of anomaly videos"
    )

    # Extract frames command
    subparsers.add_parser("extract", help="Extract frames from videos")

    # Train model command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs"
    )
    train_parser.add_argument("--batch_size", type=int, default=None, help="Batch size")

    # Dashboard command
    subparsers.add_parser("dashboard", help="Start Streamlit dashboard")

    # Detection command
    detect_parser = subparsers.add_parser("detect", help="Start real-time detection")
    detect_parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help='Video source: "webcam" or path to video file',
    )
    detect_parser.add_argument(
        "--output", type=str, default=None, help="Output video path (optional)"
    )

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(args)
    elif args.command == "extract":
        extract_frames(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "dashboard":
        start_dashboard(args)
    elif args.command == "detect":
        start_detection(args)
    else:
        print("Please specify a command. Use --help for available commands.")
        print("Available commands: generate, extract, train, dashboard, detect")


if __name__ == "__main__":
    main()
