import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm

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

        print(f"Extracting frames from {video_path} (Label: {label})")

        frame_count = 0
        saved_count = 0

        with tqdm(total=total_frames) as pbar:
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
        print(f"Extracted {saved_count} frames from {video_name}")
        return saved_count

    def process_dataset(self, input_dir, dataset_name):
        """
        Process all videos in a dataset directory and auto-detect labels from filenames
        """
        input_path = Path(input_dir)
        video_files = list(input_path.glob("*.avi")) + list(input_path.glob("*.mp4"))

        if not video_files:
            print(f"No video files found in {input_dir}")
            return

        print(f"Found {len(video_files)} videos in {dataset_name}")

        for video_file in video_files:
            video_name = video_file.stem

            # Auto-detect label from filename
            if "normal" in video_name.lower():
                label = "normal"
            elif "anomaly" in video_name.lower():
                label = "anomaly"
            else:
                label = "unknown"
                print(
                    f"Warning: Could not determine label for {video_name}, using 'unknown'"
                )

            self.extract_frames_from_video(
                video_file, settings.FRAMES_DIR, video_name, label
            )


# Define the function that main.py is looking for
def extract_frames_from_fake_dataset():
    """Extract frames from the generated fake dataset"""
    extractor = FrameExtractor(fps=5)

    # Process UCF-Crime dataset
    ucf_crime_dir = settings.RAW_DATA_DIR / "ucf_crime"
    if ucf_crime_dir.exists():
        extractor.process_dataset(ucf_crime_dir, "ucf_crime")
    else:
        print(f"UCF-Crime directory not found: {ucf_crime_dir}")

    # Process VIRAT dataset (we're not generating VIRAT in our simplified version)
    virat_dir = settings.RAW_DATA_DIR / "virat"
    if virat_dir.exists():
        extractor.process_dataset(virat_dir, "virat")
    else:
        print(f"VIRAT directory not found: {virat_dir}")


if __name__ == "__main__":
    extract_frames_from_fake_dataset()
