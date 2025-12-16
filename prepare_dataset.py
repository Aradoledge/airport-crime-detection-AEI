#!/usr/bin/env python3
"""
Prepare the UCF-Crime dataset for training
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def prepare_real_dataset():
    """Prepare real UCF-Crime dataset for training"""
    print("=== Preparing UCF-Crime Dataset ===")

    import config.settings as settings
    from src.data_preprocessing.frame_extractor import FrameExtractor

    # Check if dataset exists
    ucf_dir = settings.RAW_DATA_DIR / "ucf_crime"
    if not ucf_dir.exists():
        print("❌ UCF-Crime dataset not found!")
        print("Please download it from: https://www.crcv.ucf.edu/projects/real-world/")
        print("And extract to: data/raw/ucf_crime/")
        return False

    # Check structure
    normal_dir = ucf_dir / "Normal"
    anomaly_dir = ucf_dir / "Anomaly"

    if not normal_dir.exists() or not anomaly_dir.exists():
        print("❌ Dataset structure incorrect!")
        print("Expected: data/raw/ucf_crime/Normal/ and data/raw/ucf_crime/Anomaly/")
        return False

    # Count videos
    normal_videos = list(normal_dir.glob("*.mp4"))
    anomaly_dirs = [d for d in anomaly_dir.iterdir() if d.is_dir()]
    anomaly_videos = []
    for d in anomaly_dirs:
        anomaly_videos.extend(list(d.glob("*.mp4")))

    print(f"Found:")
    print(f"  Normal videos: {len(normal_videos)}")
    print(f"  Anomaly videos: {len(anomaly_videos)}")
    print(f"  Anomaly categories: {[d.name for d in anomaly_dirs]}")

    # Ask user for action
    print("\nOptions:")
    print("1. Extract sample dataset (20 videos - for testing)")
    print("2. Extract full dataset (will take time and space)")
    print("3. Check existing frames")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        # Extract sample
        extractor = FrameExtractor(fps=5)
        extractor.process_sample_dataset(ucf_dir, sample_size=20)
        print("\n✅ Sample dataset extracted!")

    elif choice == "2":
        # Extract full dataset
        print("\n⚠ WARNING: This will extract frames from ALL videos.")
        print("This will take a long time and require significant disk space.")
        confirm = input("Continue? (yes/no): ").strip().lower()

        if confirm == "yes":
            extractor = FrameExtractor(fps=5)
            extractor.process_ucf_crime_dataset(ucf_dir)
            print("\n✅ Full dataset extracted!")
        else:
            print("Operation cancelled.")

    elif choice == "3":
        # Check existing frames
        normal_frames = list((settings.FRAMES_DIR / "normal").glob("*.jpg"))
        anomaly_frames = list((settings.FRAMES_DIR / "anomaly").glob("*.jpg"))

        print(f"\nExisting frames:")
        print(f"  Normal: {len(normal_frames)}")
        print(f"  Anomaly: {len(anomaly_frames)}")
        print(f"  Total: {len(normal_frames) + len(anomaly_frames)}")

    return True


if __name__ == "__main__":
    prepare_real_dataset()
