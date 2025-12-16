#!/usr/bin/env python3
"""
OpenVINO Hardware Accelerated Airport Security System
"""

import cv2
import numpy as np
import time
from openvino import Core
import multiprocessing as mp
from pathlib import Path
import config.settings as settings
from ultralytics import YOLO

class OpenVINOAcceleratedDetector:
    def __init__(self):
        print("🚀 Initializing OpenVINO Hardware Acceleration...")
        
        # Initialize OpenVINO Core
        self.core = Core()
        self.devices = self.core.available_devices
        print(f"Available devices: {self.devices}")
        
        # Choose device - GPU first, fallback to CPU
        if 'GPU' in self.devices:
            self.device = 'GPU'
            print("✅ Using Intel UHD Graphics GPU")
        else:
            self.device = 'CPU'
            print("⚠️ Using CPU (GPU not available)")
        
        # Load YOLOv8 model optimized for OpenVINO
        self.load_yolo_openvino()
        
        # Performance tracking
        self.fps_history = []
        self.frame_count = 0
        self.start_time = time.time()
        
        # Optimize OpenCV
        cv2.setUseOptimized(True)
        cv2.setNumThreads(mp.cpu_count())
        
    def load_yolo_openvino(self):
        """Load YOLO model optimized for OpenVINO"""
        try:
            print("🔄 Loading YOLO model for OpenVINO...")
            
            yolo_path = Path(settings.YOLO_MODEL)
            if not yolo_path.exists():
                print(f"❌ YOLO model not found at: {yolo_path}")
                print("⚠️ Falling back to standard YOLO...")
                self.load_standard_yolo()
                return
            
            # Try to find existing OpenVINO model
            openvino_path = yolo_path.with_suffix('_openvino_model')
            if openvino_path.exists():
                model_xml = openvino_path / (yolo_path.stem + ".xml")
                model_bin = openvino_path / (yolo_path.stem + ".bin")
                
                if model_xml.exists() and model_bin.exists():
                    print(f"📦 Loading OpenVINO model: {model_xml}")
                    self.model = self.core.read_model(model=str(model_xml), weights=str(model_bin))
                    self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device)
                    self.input_layer = self.compiled_model.input(0)
                    self.output_layer = self.compiled_model.output(0)
                    self.use_openvino = True
                    print("✅ OpenVINO YOLO model loaded")
                    return
            
            # Try to export to OpenVINO format
            print("🔄 Converting YOLO to OpenVINO format...")
            yolo_model = YOLO(str(yolo_path))
            yolo_model.export(format='openvino', imgsz=640, half=False)
            
            # Check if export succeeded
            openvino_path = yolo_path.with_suffix('_openvino_model')
            if openvino_path.exists():
                model_xml = openvino_path / (yolo_path.stem + ".xml")
                model_bin = openvino_path / (yolo_path.stem + ".bin")
                
                if model_xml.exists() and model_bin.exists():
                    print(f"📦 Loading converted OpenVINO model: {model_xml}")
                    self.model = self.core.read_model(model=str(model_xml), weights=str(model_bin))
                    self.compiled_model = self.core.compile_model(model=self.model, device_name=self.device)
                    self.input_layer = self.compiled_model.input(0)
                    self.output_layer = self.compiled_model.output(0)
                    self.use_openvino = True
                    print("✅ Converted OpenVINO YOLO model loaded")
                    return
            
            # Fallback to standard YOLO
            print("⚠️ Using standard YOLO (OpenVINO conversion failed)")
            self.load_standard_yolo()
            
        except Exception as e:
            print(f"❌ Error loading OpenVINO model: {e}")
            print("🔄 Falling back to standard YOLO...")
            self.load_standard_yolo()
    
    def load_standard_yolo(self):
        """Load standard YOLO model as fallback"""
        try:
            print("📦 Loading standard YOLO model...")
            self.yolo_model = YOLO(settings.YOLO_MODEL)
            self.use_openvino = False
            print("✅ Standard YOLO model loaded")
        except Exception as e:
            print(f"❌ Failed to load standard YOLO: {e}")
            self.yolo_model = None
            self.use_openvino = False
    
    def process_video_openvino(self, video_path: str, output_path: str = None):
        """Process video file with OpenVINO acceleration"""
        print(f"🎬 Processing video: {video_path}")
        
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer if output path is provided
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        print("Processing... Press 'q' to quit, 'p' to pause")
        
        paused = False
        start_time = time.time()
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections, annotated_frame = self.process_frame_openvino(frame)
                
                # Add frame info
                cv2.putText(
                    annotated_frame,
                    f"Frame: {self.frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                
                # Display
                cv2.imshow("Video Processing - OpenVINO", annotated_frame)
                
                # Write output
                if out:
                    out.write(annotated_frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
        
        # Performance stats
        total_time = time.time() - start_time
        print(f"\n✅ Processing complete!")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Processing FPS: {self.frame_count/total_time:.2f}")
        
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
    
    def process_frame_openvino(self, frame):
        """Process frame using OpenVINO acceleration"""
        # Resize for faster processing
        if frame.shape[1] > 640:
            scale = 640 / frame.shape[1]
            new_width = 640
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        if self.use_openvino and hasattr(self, 'compiled_model'):
            # Use OpenVINO model
            try:
                # Prepare input
                input_frame = cv2.resize(frame, (640, 640))
                input_blob = cv2.dnn.blobFromImage(
                    input_frame, 
                    scalefactor=1/255.0,
                    swapRB=True,
                    crop=False
                )
                
                # Run inference
                results = self.compiled_model([input_blob])[self.output_layer]
                detections = self.process_openvino_output(results, frame.shape)
                
                # Draw detections
                processed_frame = self.draw_detections(frame, detections)
                return detections, processed_frame
                
            except Exception as e:
                print(f"OpenVINO inference error: {e}, falling back to YOLO")
                self.use_openvino = False
        
        # Use standard YOLO
        if hasattr(self, 'yolo_model') and self.yolo_model:
            try:
                results = self.yolo_model(frame, conf=0.4, verbose=False, imgsz=320)
                detections = self.process_yolo_results(results)
                processed_frame = results[0].plot()
                return detections, processed_frame
            except Exception as e:
                print(f"YOLO inference error: {e}")
        
        # No detection available
        return [], frame
    
    def process_openvino_output(self, output, frame_shape):
        """Process OpenVINO model output"""
        detections = []
        
        # This is a simplified version - adjust based on your model output
        # Typically YOLOv8 OpenVINO output format needs custom processing
        if output is not None and len(output) > 0:
            # Placeholder - implement actual processing based on your model
            print(f"OpenVINO output shape: {output.shape}")
            # For now, return empty detections
            pass
            
        return detections
    
    def process_yolo_results(self, results):
        """Process YOLO results"""
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                class_id = int(box.cls.item())
                class_name = results[0].names[class_id]
                
                if class_name in ['person', 'backpack', 'suitcase', 'handbag', 'knife']:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf.item())
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class']}: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def calculate_fps(self):
        """Calculate and display FPS"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 1.0:  # Update FPS every second
            fps = self.frame_count / elapsed
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            self.frame_count = 0
            self.start_time = current_time
        
        # Return average FPS
        if self.fps_history:
            return np.mean(self.fps_history)
        return 0.0
    
    def process_webcam_openvino(self):
        """Process webcam with OpenVINO acceleration"""
        print("\n🎥 Starting OpenVINO-accelerated webcam detection")
        print("="*50)
        
        # Open camera with hardware acceleration if possible
        cap_backends = [
            cv2.CAP_V4L2,      # Video4Linux2
            cv2.CAP_FFMPEG,    # FFMPEG
            cv2.CAP_ANY        # Auto
        ]
        
        cap = None
        for backend in cap_backends:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"✅ Camera opened with backend: {backend}")
                break
        
        if not cap or not cap.isOpened():
            print("❌ Cannot open webcam")
            return
        
        # Configure camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Reduce latency
        
        print("\nControls:")
        print("  [Ctrl + C] - Quit")
        print("="*50)
        
        paused = False
        confidence = 0.4
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                detections, processed_frame = self.process_frame_openvino(frame)
                
                # Draw detections
                processed_frame = self.draw_detections(processed_frame, detections)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Device: {self.device}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Detections: {len(detections)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('OpenVINO Accelerated Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("⏸️ Paused" if paused else "▶️ Resumed")
            elif key == ord('+'):
                confidence = min(0.9, confidence + 0.05)
                print(f"Confidence: {confidence:.2f}")
            elif key == ord('-'):
                confidence = max(0.1, confidence - 0.05)
                print(f"Confidence: {confidence:.2f}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Performance summary
        print("\n" + "="*50)
        print("📊 PERFORMANCE SUMMARY:")
        print("="*50)
        if self.fps_history:
            print(f"Average FPS: {np.mean(self.fps_history):.1f}")
            print(f"Max FPS: {np.max(self.fps_history):.1f}")
            print(f"Min FPS: {np.min(self.fps_history):.1f}")
        print(f"Device used: {self.device}")
        print("="*50)

if __name__ == "__main__":
    detector = OpenVINOAcceleratedDetector()
    detector.process_webcam_openvino()