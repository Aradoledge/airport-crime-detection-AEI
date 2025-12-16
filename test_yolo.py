#!/usr/bin/env python3
"""
Test script for YOLO detector
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_yolo_detector():
    """Test the YOLO detector with a sample image"""
    print("=== Testing YOLO Detector ===")
    
    try:
        from src.detection.yolo_detector import YOLODetector
        print("✓ YOLODetector imported successfully")
        
        # Initialize detector
        detector = YOLODetector()
        print("✓ YOLO detector initialized")
        
        # Test model info
        model_info = detector.get_model_info()
        print(f"Model info: {model_info}")
        
        # Create a test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("✓ Test image created")
        
        # Test processing
        results, annotated_frame = detector.process_frame(test_image)
        print(f"✓ Frame processed successfully")
        print(f"Detections: {len(results['detections'])}")
        
        for detection in results['detections']:
            print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing YOLO detector: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test with a real image from our dataset"""
    print("\n=== Testing with Real Image ===")
    
    try:
        from src.detection.yolo_detector import YOLODetector
        import config.settings as settings
        
        # Find a test image
        image_files = list(settings.FRAMES_DIR.rglob("*.jpg"))
        if not image_files:
            print("No test images found in dataset")
            return False
            
        test_image_path = image_files[0]
        print(f"Testing with: {test_image_path.name}")
        
        # Load image
        image = cv2.imread(str(test_image_path))
        if image is None:
            print("Failed to load test image")
            return False
            
        # Initialize detector
        detector = YOLODetector()
        
        # Process image
        results, annotated_frame = detector.process_frame(image)
        
        print(f"Detections: {len(results['detections'])}")
        for detection in results['detections']:
            print(f"  - {detection['class']} (confidence: {detection['confidence']:.2f})")
        
        # Save result for inspection
        output_path = settings.MODELS_DIR / "yolo_test_result.jpg"
        cv2.imwrite(str(output_path), annotated_frame)
        print(f"✓ Result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in real image test: {e}")
        return False

if __name__ == "__main__":
    success1 = test_yolo_detector()
    success2 = test_with_real_image()
    
    if success1 and success2:
        print("\n🎉 YOLO detector tests passed!")
    else:
        print("\n❌ Some YOLO tests failed.")
