#!/usr/bin/env python3
"""
Test hardware acceleration options
"""

import cv2
import numpy as np
import time
from openvino import Core

def test_opencv_acceleration():
    print("🔍 Testing OpenCV Hardware Acceleration")
    print("="*50)
    
    # Check OpenCV version and features
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Optimized: {cv2.useOptimized()}")
    
    # Check available backends
    print("\n📹 Video Capture Backends:")
    backends = {
        'CAP_V4L2': cv2.CAP_V4L2,
        'CAP_FFMPEG': cv2.CAP_FFMPEG,
        'CAP_GSTREAMER': cv2.CAP_GSTREAMER,
    }
    
    for name, backend in backends.items():
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print(f"  ✅ {name}: Available")
                cap.release()
            else:
                print(f"  ❌ {name}: Not available")
        except:
            print(f"  ❌ {name}: Error")
    
    # Check OpenCL
    print("\n⚡ OpenCL Acceleration:")
    if cv2.ocl.haveOpenCL():
        print("  ✅ OpenCL available")
        cv2.ocl.setUseOpenCL(True)
        print(f"  OpenCL using device: {cv2.ocl.Device.getDefault().name()}")
    else:
        print("  ❌ OpenCL not available")

def test_openvino():
    print("\n🧠 Testing OpenVINO")
    print("="*50)
    
    try:
        core = Core()
        devices = core.available_devices
        print(f"Available devices: {devices}")
        
        for device in devices:
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"  {device}: {device_name}")
        
        return devices
    except Exception as e:
        print(f"❌ OpenVINO error: {e}")
        return []

def test_webcam_performance():
    print("\n🎯 Testing Webcam Performance")
    print("="*50)
    
    # Try different backends
    test_backends = [
        ('V4L2', cv2.CAP_V4L2),
        ('FFMPEG', cv2.CAP_FFMPEG),
        ('AUTO', cv2.CAP_ANY),
    ]
    
    for backend_name, backend in test_backends:
        print(f"\nTesting {backend_name}:")
        
        try:
            cap = cv2.VideoCapture(0, backend)
            if not cap.isOpened():
                print("  ❌ Cannot open camera")
                continue
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Warm up
            for _ in range(5):
                cap.read()
            
            # Benchmark
            frames = 0
            start_time = time.time()
            
            while frames < 50:  # Test 50 frames
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Simple processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                
                frames += 1
            
            end_time = time.time()
            fps = frames / (end_time - start_time)
            
            print(f"  FPS: {fps:.1f}")
            
            cap.release()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_yolo_performance():
    print("\n🤖 Testing YOLO Performance")
    print("="*50)
    
    try:
        from ultralytics import YOLO
        import torch
        
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load model
        model = YOLO('yolov8n.pt')
        
        # Create test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test inference
        times = []
        for i in range(5):
            start = time.time()
            results = model(test_image, verbose=False, imgsz=320)
            end = time.time()
            times.append(end - start)
            
            if i == 0 and len(results) > 0:
                print(f"  First inference: {times[0]*1000:.0f}ms")
                print(f"  Detected: {len(results[0].boxes)} objects")
        
        avg_time = np.mean(times[1:])  # Skip first (warm-up)
        print(f"  Average inference: {avg_time*1000:.0f}ms")
        print(f"  Estimated FPS: {1/avg_time:.1f}")
        
    except Exception as e:
        print(f"❌ YOLO test error: {e}")

def main():
    print("🖥️ HARDWARE ACCELERATION DIAGNOSTIC")
    print("="*60)
    
    # Run tests
    test_opencv_acceleration()
    devices = test_openvino()
    test_webcam_performance()
    test_yolo_performance()
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("="*60)
    
    if 'GPU' in devices:
        print("✅ Use OpenVINO with GPU acceleration")
        print("   Command: python openvino_accelerator.py")
    elif cv2.ocl.haveOpenCL():
        print("✅ Use OpenCL acceleration with OpenCV")
        print("   Command: Set cv2.ocl.setUseOpenCL(True)")
    else:
        print("✅ Use multi-threaded CPU processing")
        print("   Command: cv2.setNumThreads(4)")
    
    print("\nFor best performance on your Dell Latitude 5491:")
    print("1. Use OpenVINO with GPU (Intel UHD Graphics)")
    print("2. Set resolution to 640x480")
    print("3. Use YOLOv8n (nano) model")
    print("4. Process every 2nd frame if needed")

if __name__ == "__main__":
    main()
