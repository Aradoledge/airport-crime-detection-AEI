import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm
import re

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.settings as settings


class FrameExtractor:
    def __init__(self, fps=5):
        self.fps = fps

    def extract_frames_from_video(
        self, video_path, output_dir, video_name, label="normal"
    ):
        """
        Extract frames from a video file and organize by label
        """
        # Create output directories for labels
        normal_dir = output_dir / "normal"
        anomaly_dir = output_dir / "anomaly"
        normal_dir.mkdir(parents=True, exist_ok=True)
        anomaly_dir.mkdir(parents=True, exist_ok=True)

        # Choose output directory based on label
        if label == "normal":
            final_output_dir = normal_dir
        else:
            final_output_dir = anomaly_dir

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return 0

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate frame interval based on desired FPS
        if original_fps > 0:
            frame_interval = max(1, int(original_fps / self.fps))
        else:
            frame_interval = 1

        print(f"Extracting frames from {video_path.name} (Label: {label})")

        frame_count = 0
        saved_count = 0

        with tqdm(total=total_frames, desc=f"Processing {video_path.name}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Save frame at specified interval
                if frame_count % frame_interval == 0:
                    frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
                    frame_path = final_output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame)
                    saved_count += 1

                frame_count += 1
                pbar.update(1)

        cap.release()
        print(f"✓ Extracted {saved_count} frames from {video_name}")
        return saved_count

    def process_ucf_crime_dataset(self, input_dir):
        """
        Process UCF-Crime dataset with proper labeling
        """
        input_path = Path(input_dir)

        # Track statistics
        stats = {
            "normal_videos": 0,
            "anomaly_videos": 0,
            "normal_frames": 0,
            "anomaly_frames": 0,
        }

        print("=== Processing UCF-Crime Dataset ===")

        # Process Normal videos
        normal_dir = input_path / "Normal"
        if normal_dir.exists():
            normal_videos = list(normal_dir.glob("*.mp4"))
            print(f"Found {len(normal_videos)} normal videos")

            for video_file in tqdm(normal_videos, desc="Normal videos"):
                video_name = video_file.stem
                frames_extracted = self.extract_frames_from_video(
                    video_file, settings.FRAMES_DIR, video_name, label="normal"
                )
                stats["normal_videos"] += 1
                stats["normal_frames"] += frames_extracted
        else:
            print("⚠ Normal directory not found")

        # Process Anomaly videos
        anomaly_dir = input_path / "Anomaly"
        if anomaly_dir.exists():
            anomaly_categories = [d for d in anomaly_dir.iterdir() if d.is_dir()]
            print(f"Found {len(anomaly_categories)} anomaly categories")

            for category in anomaly_categories:
                category_name = category.name
                anomaly_videos = list(category.glob("*.mp4"))
                print(f"  Processing {category_name}: {len(anomaly_videos)} videos")

                for video_file in tqdm(anomaly_videos, desc=f"{category_name}"):
                    video_name = f"{category_name}_{video_file.stem}"
                    frames_extracted = self.extract_frames_from_video(
                        video_file, settings.FRAMES_DIR, video_name, label="anomaly"
                    )
                    stats["anomaly_videos"] += 1
                    stats["anomaly_frames"] += frames_extracted
        else:
            print("⚠ Anomaly directory not found")

        # Print summary
        print("\n" + "=" * 50)
        print("EXTRACTION SUMMARY:")
        print("=" * 50)
        print(f"Normal Videos: {stats['normal_videos']}")
        print(f"Anomaly Videos: {stats['anomaly_videos']}")
        print(f"Total Videos: {stats['normal_videos'] + stats['anomaly_videos']}")
        print(f"Normal Frames: {stats['normal_frames']}")
        print(f"Anomaly Frames: {stats['anomaly_frames']}")
        print(f"Total Frames: {stats['normal_frames'] + stats['anomaly_frames']}")
        print("=" * 50)

        return stats

    def process_sample_dataset(self, input_dir, sample_size=10):
        """
        Process a small sample of the dataset for testing
        """
        input_path = Path(input_dir)

        print("=== Processing Sample Dataset ===")

        # Process a few normal videos
        normal_dir = input_path / "Normal"
        if normal_dir.exists():
            normal_videos = list(normal_dir.glob("*.mp4"))[: sample_size // 2]
            print(f"Processing {len(normal_videos)} normal videos")

            for video_file in normal_videos:
                video_name = video_file.stem
                self.extract_frames_from_video(
                    video_file, settings.FRAMES_DIR, video_name, label="normal"
                )

        # Process a few anomaly videos
        anomaly_dir = input_path / "Anomaly"
        if anomaly_dir.exists():
            # Get first anomaly category
            anomaly_categories = [d for d in anomaly_dir.iterdir() if d.is_dir()]
            if anomaly_categories:
                category = anomaly_categories[0]
                anomaly_videos = list(category.glob("*.mp4"))[: sample_size // 2]
                print(f"Processing {len(anomaly_videos)} {category.name} videos")

                for video_file in anomaly_videos:
                    video_name = f"{category.name}_{video_file.stem}"
                    self.extract_frames_from_video(
                        video_file, settings.FRAMES_DIR, video_name, label="anomaly"
                    )


def extract_frames_from_real_dataset():
    """Extract frames from the real UCF-Crime dataset"""
    extractor = FrameExtractor(fps=5)

    # Process UCF-Crime dataset
    ucf_crime_dir = settings.RAW_DATA_DIR / "ucf_crime"
    if ucf_crime_dir.exists():
        # For testing, use sample dataset first
        extractor.process_sample_dataset(ucf_crime_dir, sample_size=20)

        # For full dataset (will take time!)
        # extractor.process_ucf_crime_dataset(ucf_crime_dir)
    else:
        print(f"❌ UCF-Crime directory not found: {ucf_crime_dir}")
        print(
            "Please download the dataset from: https://www.crcv.ucf.edu/projects/real-world/"
        )


if __name__ == "__main__":
    extract_frames_from_real_dataset()
