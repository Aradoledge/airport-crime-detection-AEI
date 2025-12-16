import cv2
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config.settings as settings


class FakeDatasetGenerator:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir or settings.RAW_DATA_DIR
        self.video_width = 640
        self.video_height = 480
        self.fps = 10
        self.duration = 5  # seconds per video (shorter for testing)

    def create_background(self, style="airport"):
        """Create different background styles"""
        if style == "airport":
            # Airport terminal background
            bg = np.ones((self.video_height, self.video_width, 3), dtype=np.uint8) * 200
            # Add some airport-like features
            cv2.rectangle(
                bg, (0, 0), (self.video_width, 100), (100, 100, 150), -1
            )  # Top area
            cv2.rectangle(
                bg,
                (0, self.video_height - 50),
                (self.video_width, self.video_height),
                (80, 80, 80),
                -1,
            )  # Floor
            # Add some vertical lines (pillars)
            for x in range(100, self.video_width, 150):
                cv2.rectangle(
                    bg, (x, 100), (x + 20, self.video_height - 50), (150, 150, 180), -1
                )
            return bg
        elif style == "security_check":
            # Security checkpoint background
            bg = np.ones((self.video_height, self.video_width, 3), dtype=np.uint8) * 180
            cv2.rectangle(
                bg, (200, 100), (400, self.video_height - 100), (120, 120, 120), -1
            )  # Scanner
            return bg
        else:
            return (
                np.ones((self.video_height, self.video_width, 3), dtype=np.uint8) * 150
            )

    def draw_person(self, frame, position, color=(0, 0, 255), size=30):
        """Draw a simple person representation - FIXED VERSION"""
        x, y = position
        x = int(x)
        y = int(y)
        size = int(size)

        # Body - FIXED: convert all coordinates to integers
        cv2.rectangle(
            frame,
            (int(x - size // 3), int(y - size)),
            (int(x + size // 3), int(y)),
            color,
            -1,
        )
        # Head
        cv2.circle(frame, (int(x), int(y - size - 10)), int(size // 3), color, -1)
        # Legs
        cv2.line(
            frame,
            (int(x - size // 4), int(y)),
            (int(x - size // 4), int(y + size // 2)),
            color,
            3,
        )
        cv2.line(
            frame,
            (int(x + size // 4), int(y)),
            (int(x + size // 4), int(y + size // 2)),
            color,
            3,
        )
        return frame

    def draw_luggage(self, frame, position, color=(0, 255, 255), size=20):
        """Draw a luggage item - FIXED VERSION"""
        x, y = position
        x = int(x)
        y = int(y)
        size = int(size)

        cv2.rectangle(
            frame,
            (int(x - size), int(y - size)),
            (int(x + size), int(y + size)),
            color,
            -1,
        )
        cv2.rectangle(
            frame,
            (int(x - size), int(y - size)),
            (int(x + size), int(y + size)),
            (0, 0, 0),
            2,
        )
        return frame

    def generate_normal_video(self, video_path, num_people=3):
        """Generate a normal airport scene video"""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(video_path), fourcc, self.fps, (self.video_width, self.video_height)
        )

        background = self.create_background("airport")

        # Create people with random walking paths
        people = []
        for i in range(num_people):
            start_x = random.randint(50, self.video_width - 50)
            start_y = random.randint(150, self.video_height - 100)
            speed_x = random.uniform(-2, 2)
            speed_y = random.uniform(-1, 1)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            people.append(
                {"pos": [start_x, start_y], "speed": [speed_x, speed_y], "color": color}
            )

        for frame_num in range(self.fps * self.duration):
            frame = background.copy()

            # Update and draw people
            for person in people:
                # Move person
                person["pos"][0] += person["speed"][0]
                person["pos"][1] += person["speed"][1]

                # Bounce off edges
                if person["pos"][0] < 50 or person["pos"][0] > self.video_width - 50:
                    person["speed"][0] *= -1
                if person["pos"][1] < 150 or person["pos"][1] > self.video_height - 100:
                    person["speed"][1] *= -1

                # Draw person
                frame = self.draw_person(
                    frame, (person["pos"][0], person["pos"][1]), person["color"]
                )

            # Add some random luggage being carried
            if frame_num % 30 == 0 and random.random() > 0.7:
                luggage_pos = (
                    random.randint(100, self.video_width - 100),
                    random.randint(200, self.video_height - 150),
                )
                frame = self.draw_luggage(frame, luggage_pos)

            out.write(frame)

        out.release()
        print(f"Generated normal video: {video_path}")

    def generate_anomaly_video(self, video_path, anomaly_type="unattended_bag"):
        """Generate an anomaly video - SIMPLIFIED VERSION"""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(video_path), fourcc, self.fps, (self.video_width, self.video_height)
        )

        background = self.create_background("airport")

        if anomaly_type == "unattended_bag":
            # Person leaves a bag and walks away - SIMPLIFIED
            person_x = 100
            bag_x = 320
            bag_y = 300

            for frame_num in range(self.fps * self.duration):
                frame = background.copy()

                if frame_num < 15:  # First 1.5 seconds: person with bag
                    frame = self.draw_person(frame, (person_x, 300), (0, 0, 255))
                    frame = self.draw_luggage(frame, (person_x + 40, 300))
                    person_x += 3  # Move right
                elif frame_num == 15:  # Person leaves bag
                    bag_x = person_x + 40
                    frame = self.draw_person(frame, (person_x, 300), (0, 0, 255))
                    frame = self.draw_luggage(frame, (bag_x, bag_y), (0, 255, 255))
                elif frame_num > 15 and frame_num < 30:  # Person walks away
                    person_x += 3
                    frame = self.draw_person(frame, (person_x, 300), (0, 0, 255))
                    frame = self.draw_luggage(frame, (bag_x, bag_y), (0, 255, 255))
                else:  # Bag remains unattended
                    frame = self.draw_luggage(frame, (bag_x, bag_y), (0, 255, 255))
                    # Make bag flash to indicate anomaly
                    if frame_num % 10 < 5:
                        cv2.putText(
                            frame,
                            "UNATTENDED BAG!",
                            (bag_x - 80, bag_y - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

                out.write(frame)

        elif anomaly_type == "running_person":
            # Person running in restricted area - SIMPLIFIED
            person_x = 50

            for frame_num in range(self.fps * self.duration):
                frame = background.copy()

                if frame_num < 20:  # Normal walking
                    frame = self.draw_person(frame, (person_x, 200), (0, 0, 255))
                    person_x += 2
                else:  # Start running
                    frame = self.draw_person(
                        frame, (person_x, 200), (255, 0, 0)
                    )  # Red for anomaly
                    person_x += 5  # Faster movement
                    cv2.putText(
                        frame,
                        "RUNNING!",
                        (person_x - 40, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                out.write(frame)

        elif anomaly_type == "loitering":
            # Person loitering in one area - SIMPLIFIED
            center_x, center_y = 320, 240

            for frame_num in range(self.fps * self.duration):
                frame = background.copy()

                # Small circular movement pattern (loitering)
                angle = frame_num * 0.2
                offset_x = int(20 * np.cos(angle))
                offset_y = int(15 * np.sin(angle))

                current_x = center_x + offset_x
                current_y = center_y + offset_y
                frame = self.draw_person(
                    frame, (current_x, current_y), (255, 165, 0)
                )  # Orange for suspicious

                # Add loitering warning after some time
                if frame_num > 20:
                    cv2.putText(
                        frame,
                        "LOITERING",
                        (current_x - 40, current_y - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                out.write(frame)

        out.release()
        print(f"Generated anomaly video ({anomaly_type}): {video_path}")

    def generate_dataset(self, num_normal=5, num_anomaly=5):
        """Generate complete fake dataset"""
        print("Generating fake airport surveillance dataset...")

        # Create directories
        ucf_dir = self.output_dir / "ucf_crime"
        ucf_dir.mkdir(parents=True, exist_ok=True)

        # UCF-Crime style videos
        print("Generating UCF-Crime style videos...")

        # Normal videos
        for i in tqdm(range(num_normal), desc="Normal videos"):
            video_path = ucf_dir / f"normal_{i:03d}.avi"
            self.generate_normal_video(video_path, num_people=random.randint(2, 4))

        # Anomaly videos
        anomaly_types = ["unattended_bag", "running_person", "loitering"]
        for i in tqdm(range(num_anomaly), desc="Anomaly videos"):
            anomaly_type = random.choice(anomaly_types)
            video_path = ucf_dir / f"anomaly_{anomaly_type}_{i:03d}.avi"
            self.generate_anomaly_video(video_path, anomaly_type)

        print(f"Dataset generation complete!")
        print(f"Normal videos: {num_normal}")
        print(f"Anomaly videos: {num_anomaly}")
        print(f"Total videos: {num_normal + num_anomaly}")


def main():
    generator = FakeDatasetGenerator()
    generator.generate_dataset(num_normal=3, num_anomaly=3)  # Smaller for testing


if __name__ == "__main__":
    main()
