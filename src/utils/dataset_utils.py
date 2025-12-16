import os
import cv2
import numpy as np
from pathlib import Path
import config.settings as settings


def create_sample_dataset():
    """
    Create a small sample dataset for testing if real datasets are not available
    """
    print("Creating sample dataset for testing...")

    sample_dir = settings.FRAMES_DIR / "sample_videos"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Create some sample frames (in real scenario, you'd extract from actual videos)
    for i in range(100):
        # Create sample "normal" frames
        normal_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(sample_dir / f"normal_frame_{i:04d}.jpg"), normal_frame)

        # Create sample "anomaly" frames (with different patterns)
        if i % 20 == 0:  # Every 20th frame is "anomalous"
            anomaly_frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
            # Add some visual pattern to distinguish
            cv2.rectangle(anomaly_frame, (100, 100), (200, 200), (0, 0, 255), -1)
            cv2.imwrite(str(sample_dir / f"anomaly_frame_{i:04d}.jpg"), anomaly_frame)

    print(f"Sample dataset created at {sample_dir}")


def validate_dataset_structure():
    """
    Validate that the dataset structure is correct
    """
    required_dirs = [
        settings.RAW_DATA_DIR / "ucf_crime",
        settings.FRAMES_DIR / "anomaly",
        settings.FRAMES_DIR / "normal",
        settings.SEQUENCES_DIR,
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"Warning: Directory {dir_path} does not exist")
            return False

    print("Dataset structure is valid")
    return True


def get_dataset_stats():
    """
    Get statistics about the available dataset
    """
    stats = {}

    # Count frames
    anomaly_frames = list(settings.FRAMES_DIR.rglob("anomaly/*.jpg"))
    normal_frames = list(settings.FRAMES_DIR.rglob("normal/*.jpg"))

    stats["anomaly_frames"] = len(anomaly_frames)
    stats["normal_frames"] = len(normal_frames)
    stats["total_frames"] = stats["anomaly_frames"] + stats["normal_frames"]

    # Count videos in raw data
    ucf_videos = list((settings.RAW_DATA_DIR / "ucf_crime").glob("*.mp4"))
    virat_videos = list((settings.RAW_DATA_DIR / "virat").glob("*.mp4"))

    stats["ucf_videos"] = len(ucf_videos)
    stats["virat_videos"] = len(virat_videos)

    return stats


if __name__ == "__main__":
    # Create sample dataset if no real data exists
    if not validate_dataset_structure():
        print("Creating sample dataset for testing...")
        create_sample_dataset()

    stats = get_dataset_stats()
    print("Dataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
