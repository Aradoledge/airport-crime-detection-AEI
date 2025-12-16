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


def generate_fake_dataset(args):
    """Generate fake dataset (for testing only)"""
    print("Generating fake dataset...")
    from src.utils.generate_fake_dataset import FakeDatasetGenerator

    generator = FakeDatasetGenerator()
    generator.generate_dataset(num_normal=args.num_normal, num_anomaly=args.num_anomaly)


def extract_frames(args):
    """Extract frames from videos"""
    print("Extracting frames from videos...")

    if args.sample:
        # Extract sample from real dataset
        from src.data_preprocessing.frame_extractor import FrameExtractor
        import config.settings as settings

        extractor = FrameExtractor(fps=5)
        ucf_crime_dir = settings.RAW_DATA_DIR / "ucf_crime"
        if ucf_crime_dir.exists():
            extractor.process_sample_dataset(
                ucf_crime_dir, sample_size=args.sample_size
            )
        else:
            print("❌ Real dataset not found. Use --fake flag or download the dataset.")
    elif args.fake:
        # Extract from fake dataset
        from src.data_preprocessing.frame_extractor import (
            extract_frames_from_fake_dataset,
        )

        extract_frames_from_fake_dataset()
    else:
        # Extract from real dataset
        from src.data_preprocessing.frame_extractor import (
            extract_frames_from_real_dataset,
        )

        extract_frames_from_real_dataset()


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

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except Exception as e:
        print(f"Dashboard failed: {e}")


def start_detection(args):
    """Start real-time detection"""
    print(f"Starting detection with source: {args.source}")

    # Try OpenVINO accelerated detector first
    try:
        from openvino_accelerator import OpenVINOAcceleratedDetector

        print("🚀 Using OpenVINO accelerated detector...")
        detector = OpenVINOAcceleratedDetector()

        if args.source == "webcam":
            print("Starting webcam detection with OpenVINO acceleration...")
            detector.process_webcam_openvino()
        else:
            print(f"Processing video file with OpenVINO: {args.source}")
            detector.process_video_openvino(args.source, args.output)

    except ImportError as e:
        print(f"⚠️ OpenVINO not available: {e}")
        print("🔄 Falling back to standard detector...")

        # Fallback to standard detector
        from detect import RealTimeDetector

        detector = RealTimeDetector()

        if args.source == "webcam":
            print("Starting webcam detection (standard mode)...")
            detector.process_webcam()
        else:
            print(f"Processing video file: {args.source}")
            detector.process_video(args.source, args.output)

    except Exception as e:
        print(f"❌ Error starting detection: {e}")
        print("Trying standard detector as last resort...")

        # Last resort: standard detector
        try:
            from detect import RealTimeDetector

            detector = RealTimeDetector()

            if args.source == "webcam":
                detector.process_webcam()
            else:
                detector.process_video(args.source, args.output)
        except Exception as e2:
            print(f"❌ All detection methods failed: {e2}")


def dataset_stats(args):
    """Show dataset statistics"""
    print("=== Dataset Statistics ===")

    import config.settings as settings

    # Check frames
    normal_frames = list((settings.FRAMES_DIR / "normal").glob("*.jpg"))
    anomaly_frames = list((settings.FRAMES_DIR / "anomaly").glob("*.jpg"))

    print(f"Normal frames: {len(normal_frames)}")
    print(f"Anomaly frames: {len(anomaly_frames)}")
    print(f"Total frames: {len(normal_frames) + len(anomaly_frames)}")

    # Check raw videos
    ucf_dir = settings.RAW_DATA_DIR / "ucf_crime"
    if ucf_dir.exists():
        normal_videos = list((ucf_dir / "Normal").glob("*.mp4"))
        anomaly_dirs = [d for d in (ucf_dir / "Anomaly").iterdir() if d.is_dir()]
        anomaly_videos = []
        for d in anomaly_dirs:
            anomaly_videos.extend(list(d.glob("*.mp4")))

        print(f"\nRaw UCF-Crime Dataset:")
        print(f"  Normal videos: {len(normal_videos)}")
        print(f"  Anomaly videos: {len(anomaly_videos)}")
        print(f"  Total videos: {len(normal_videos) + len(anomaly_videos)}")

        if anomaly_dirs:
            print(f"  Anomaly categories: {[d.name for d in anomaly_dirs]}")


def convert_to_openvino(args):
    """Convert YOLO model to OpenVINO format"""
    print("🔄 Converting YOLO model to OpenVINO format...")

    try:
        from ultralytics import YOLO
        import config.settings as settings

        # Load YOLO model
        model = YOLO(settings.YOLO_MODEL)

        # Export to OpenVINO format
        model.export(
            format="openvino",
            imgsz=[640, 640],
            half=False,  # Use FP32 for compatibility
            device="cpu",  # Export for CPU
        )

        print("✅ Conversion successful!")
        print(
            f"OpenVINO model saved to: {settings.YOLO_MODEL.replace('.pt', '_openvino_model/')}"
        )

    except Exception as e:
        print(f"❌ Conversion failed: {e}")


def test_hardware(args):
    """Test hardware acceleration options"""
    print("🖥️ Testing hardware acceleration...")

    try:
        from test_hardware import test_hardware_acceleration

        test_hardware_acceleration()
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        print("Make sure test_hardware.py exists in the project root.")


def main():
    parser = argparse.ArgumentParser(description="AI Airport Crime Detection System")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generate fake dataset command
    gen_parser = subparsers.add_parser(
        "generate-fake", help="Generate fake dataset (for testing)"
    )
    gen_parser.add_argument(
        "--num_normal", type=int, default=8, help="Number of normal videos"
    )
    gen_parser.add_argument(
        "--num_anomaly", type=int, default=8, help="Number of anomaly videos"
    )

    # Extract frames command
    extract_parser = subparsers.add_parser("extract", help="Extract frames from videos")
    extract_parser.add_argument(
        "--fake", action="store_true", help="Extract from fake dataset"
    )
    extract_parser.add_argument(
        "--sample", action="store_true", help="Extract sample from real dataset"
    )
    extract_parser.add_argument(
        "--sample_size", type=int, default=20, help="Sample size"
    )

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
    detect_parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "openvino", "auto"],
        default="auto",
        help="Detection mode: standard (YOLO only), openvino (OpenVINO accelerated), auto (choose best)",
    )

    # Dataset stats command
    subparsers.add_parser("stats", help="Show dataset statistics")

    # Convert to OpenVINO command
    subparsers.add_parser(
        "convert-openvino", help="Convert YOLO model to OpenVINO format"
    )

    # Test hardware command
    subparsers.add_parser("test-hardware", help="Test hardware acceleration options")

    args = parser.parse_args()

    if args.command == "generate-fake":
        generate_fake_dataset(args)
    elif args.command == "extract":
        extract_frames(args)
    elif args.command == "train":
        train_model(args)
    elif args.command == "dashboard":
        start_dashboard(args)
    elif args.command == "detect":
        # Handle different detection modes
        if hasattr(args, "mode"):
            if args.mode == "openvino":
                # Force OpenVINO mode
                try:
                    from openvino_accelerator import OpenVINOAcceleratedDetector

                    detector = OpenVINOAcceleratedDetector()
                    if args.source == "webcam":
                        detector.process_webcam_openvino()
                    else:
                        detector.process_video_openvino(args.source, args.output)
                except Exception as e:
                    print(f"❌ OpenVINO detection failed: {e}")
                    print("Falling back to standard mode...")
                    start_detection(args)
            elif args.mode == "standard":
                # Force standard mode
                from detect import RealTimeDetector

                detector = RealTimeDetector()
                if args.source == "webcam":
                    detector.process_webcam()
                else:
                    detector.process_video(args.source, args.output)
            else:  # auto mode
                start_detection(args)
        else:
            start_detection(args)
    elif args.command == "stats":
        dataset_stats(args)
    elif args.command == "convert-openvino":
        convert_to_openvino(args)
    elif args.command == "test-hardware":
        test_hardware(args)
    else:
        print("Please specify a command. Use --help for available commands.")
        print("\nAvailable commands:")
        print("  generate-fake    Generate fake dataset for testing")
        print("  extract          Extract frames from videos")
        print("  train            Train the CNN-LSTM model")
        print("  dashboard        Start Streamlit dashboard")
        print("  detect           Start real-time detection")
        print("  stats            Show dataset statistics")
        print("  convert-openvino Convert YOLO model to OpenVINO format")
        print("  test-hardware    Test hardware acceleration options")
        print("\nDetection modes:")
        print("  --mode standard  Use standard YOLO detection")
        print("  --mode openvino  Use OpenVINO accelerated detection")
        print("  --mode auto      Automatically choose best option (default)")


if __name__ == "__main__":
    main()
