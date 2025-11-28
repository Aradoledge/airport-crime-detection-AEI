import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from pathlib import Path
import config.settings as settings
from src.detection.anomaly_detector import AnomalyDetector
import time
import argparse


class RealTimeDetector:
    def __init__(self):
        # Initialize YOLO model
        try:
            self.yolo_model = YOLO(settings.YOLO_MODEL)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

        # Initialize anomaly detector
        try:
            self.anomaly_detector = AnomalyDetector()
            if self.anomaly_detector.model:
                print("Anomaly detection model loaded successfully")
            else:
                print("Anomaly detection model not available")
        except Exception as e:
            print(f"Error loading anomaly detector: {e}")
            self.anomaly_detector = None

        # Detection parameters
        self.confidence_threshold = 0.5
        self.class_names = ["person", "backpack", "suitcase", "handbag", "knife"]

    def process_frame(self, frame):
        """
        Process a single frame for object detection and anomaly detection
        """
        results = {}
        annotated_frame = frame.copy()

        # Get frame dimensions first to ensure they're always available
        h, w = annotated_frame.shape[:2]

        # Run YOLO detection if available
        if self.yolo_model:
            try:
                yolo_results = self.yolo_model(
                    frame, conf=self.confidence_threshold, verbose=False
                )

                # Filter relevant detections
                relevant_detections = []
                if len(yolo_results) > 0:
                    boxes = yolo_results[0].boxes
                    for box in boxes:
                        class_id = int(box.cls.item())
                        class_name = self.yolo_model.names[class_id]
                        confidence = box.conf.item()

                        if (
                            class_name in self.class_names
                            and confidence > self.confidence_threshold
                        ):
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            relevant_detections.append(
                                {
                                    "class": class_name,
                                    "confidence": confidence,
                                    "bbox": [x1, y1, x2, y2],
                                }
                            )

                results["detections"] = relevant_detections
                annotated_frame = yolo_results[0].plot()

            except Exception as e:
                print(f"YOLO detection error: {e}")
                results["detections"] = []
        else:
            results["detections"] = []

        # Run anomaly detection if available
        if self.anomaly_detector and self.anomaly_detector.model:
            try:
                anomaly_score = self.anomaly_detector.detect_anomaly(frame)
                results["anomaly_score"] = anomaly_score
                results["is_anomaly"] = anomaly_score > 0.7

                # Add anomaly info to frame - now h and w are always available
                cv2.putText(
                    annotated_frame,
                    f"Anomaly: {anomaly_score:.3f}",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if results["is_anomaly"]:
                    cv2.putText(
                        annotated_frame,
                        "ALERT: ANOMALY DETECTED!",
                        (w // 6, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3,
                    )

            except Exception as e:
                print(f"Anomaly detection error: {e}")
                results["anomaly_score"] = 0.0
                results["is_anomaly"] = False
        else:
            results["anomaly_score"] = 0.0
            results["is_anomaly"] = False

        return results, annotated_frame

    def process_video(self, video_path, output_path=None):
        """
        Process a video file for detection
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'p' to pause")

        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results, annotated_frame = self.process_frame(frame)

                # Add info text
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Detections: {len(results['detections'])}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                # Display frame
                cv2.imshow("Airport Security Detection", annotated_frame)

                # Write frame if output path provided
                if output_path:
                    out.write(annotated_frame)

                frame_count += 1

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        print(f"Average FPS: {frame_count/processing_time:.2f}")

        # Cleanup
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

    def process_webcam(self):
        """
        Process live webcam feed
        """
        cap = cv2.VideoCapture(0)  # 0 for default camera

        if not cap.isOpened():
            print("Error: Cannot access webcam")
            return

        print("Starting webcam detection. Press 'q' to quit, 'p' to pause.")

        frame_count = 0
        start_time = time.time()
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                results, annotated_frame = self.process_frame(frame)

                # Add detection info
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    annotated_frame,
                    f"Detections: {len(results['detections'])}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if "anomaly_score" in results:
                    cv2.putText(
                        annotated_frame,
                        f"Anomaly: {results['anomaly_score']:.3f}",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Display frame
                cv2.imshow("Airport Security - Live Detection", annotated_frame)

                frame_count += 1

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                print("Paused" if paused else "Resumed")

        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Processed {frame_count} frames in {processing_time:.2f} seconds")
        if processing_time > 0:
            print(f"Average FPS: {frame_count/processing_time:.2f}")

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function when detect.py is run directly"""
    parser = argparse.ArgumentParser(description="Real-time Airport Security Detection")
    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help="Video source: webcam, or path to video file",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output video path (optional)"
    )

    args = parser.parse_args()

    detector = RealTimeDetector()

    if args.source == "webcam":
        detector.process_webcam()
    else:
        detector.process_video(args.source, args.output)


if __name__ == "__main__":
    main()
