import cv2
import numpy as np
from ultralytics import YOLO
import config.settings as settings


class YOLODetector:
    def __init__(self, model_path=settings.YOLO_MODEL):
        """
        Initialize YOLO detector with the specified model
        """
        try:
            self.model = YOLO(model_path)
            print(f"YOLO model '{model_path}' loaded successfully")
            self.class_names = self.model.names
            self.confidence_threshold = settings.YOLO_CONFIDENCE
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
            self.class_names = {}
            self.confidence_threshold = 0.5

    def process_frame(self, frame):
        """
        Process a single frame for object detection

        Args:
            frame: Input frame (numpy array)

        Returns:
            dict: Detection results
            numpy array: Annotated frame
        """
        if self.model is None:
            # Return empty detections and original frame if model not loaded
            return {"detections": []}, frame

        try:
            # Run YOLO inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)

            detections = []
            annotated_frame = frame.copy()

            if len(results) > 0:
                # Process detections
                boxes = results[0].boxes

                for box in boxes:
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")

                    # Filter by confidence and relevant classes
                    if (
                        confidence > self.confidence_threshold
                        and self._is_relevant_class(class_name)
                    ):
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        detections.append(
                            {
                                "class": class_name,
                                "confidence": confidence,
                                "bbox": [x1, y1, x2, y2],
                            }
                        )

                        # Draw bounding box and label
                        color = self._get_class_color(class_name)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                        label = f"{class_name} {confidence:.2f}"
                        label_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )[0]
                        cv2.rectangle(
                            annotated_frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color,
                            -1,
                        )
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

            return {"detections": detections}, annotated_frame

        except Exception as e:
            print(f"Error in YOLO processing: {e}")
            # Return original frame on error
            return {"detections": []}, frame

    def _is_relevant_class(self, class_name):
        """
        Check if the detected class is relevant for airport security
        """
        relevant_classes = [
            "person",
            "backpack",
            "suitcase",
            "handbag",
            "backpack",
            "knife",
            "gun",
            "pistol",
            "cell phone",
            "laptop",
            "bottle",
        ]
        return class_name in relevant_classes

    def _get_class_color(self, class_name):
        """
        Get color for different object classes
        """
        color_map = {
            "person": (0, 255, 0),  # Green
            "backpack": (255, 165, 0),  # Orange
            "suitcase": (255, 255, 0),  # Yellow
            "handbag": (255, 0, 255),  # Magenta
            "knife": (0, 0, 255),  # Red
            "gun": (0, 0, 255),  # Red
            "pistol": (0, 0, 255),  # Red
        }
        return color_map.get(class_name, (255, 255, 255))  # Default white

    def get_model_info(self):
        """
        Get information about the loaded model
        """
        if self.model is None:
            return "Model not loaded"

        return {
            "model_name": settings.YOLO_MODEL,
            "classes_loaded": len(self.class_names),
            "confidence_threshold": self.confidence_threshold,
        }