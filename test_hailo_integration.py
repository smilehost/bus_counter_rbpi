#!/usr/bin/env python3
"""
HAILO Integration Test Script
Validates HAILO 8L AI Accelerator integration with the bus counter project
"""

import sys
import os
import time
import platform
import cv2
import numpy as np

def test_imports():
    """Test if all required modules can be imported"""
    print("=== Testing Module Imports ===")
    
    # Test basic modules
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        return False
    
    # Test HAILO modules
    hailo_available = False
    try:
        from hailo_platform import HEF, VDevice
        import hailort
        print(f"✓ HAILO Platform SDK: Available")
        hailo_available = True
    except ImportError as e:
        print(f"✗ HAILO Platform SDK: {e}")
    
    try:
        from hailo_yolo_detector import HailoYOLODetector, HAILO_AVAILABLE
        if HAILO_AVAILABLE:
            print("✓ HAILO YOLO Detector: Available")
        else:
            print("✗ HAILO YOLO Detector: Not available")
    except ImportError as e:
        print(f"✗ HAILO YOLO Detector: {e}")
    
    # Test Pi5 camera modules
    pi5_camera_available = False
    try:
        from pi5_camera import Pi5Camera, PI5_CAMERA_AVAILABLE
        if PI5_CAMERA_AVAILABLE:
            print("✓ Pi5 Camera Module: Available")
            pi5_camera_available = True
        else:
            print("✗ Pi5 Camera Module: Not available")
    except ImportError as e:
        print(f"✗ Pi5 Camera Module: {e}")
    
    # Test project modules
    try:
        from config import Config
        print("✓ Config Module: Available")
    except ImportError as e:
        print(f"✗ Config Module: {e}")
        return False
    
    try:
        from yolo_botsort_tracker import YOLOBoTSORTTracker
        print("✓ YOLO-BoTSORT Tracker: Available")
    except ImportError as e:
        print(f"✗ YOLO-BoTSORT Tracker: {e}")
        return False
    
    return hailo_available, pi5_camera_available


def test_platform_detection():
    """Test platform detection"""
    print("\n=== Testing Platform Detection ===")
    
    from config import Config
    config = Config()
    
    print(f"Platform: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Is Raspberry Pi: {config.IS_RASPBERRY_PI}")
    print(f"Is Pi5: {config.IS_PI5}")
    print(f"Use HAILO: {config.USE_HAILO}")
    print(f"Video Source: {config.VIDEO_SOURCE}")
    print(f"Camera Type: {config.CAMERA_TYPE}")
    print(f"YOLO Model: {config.YOLO_MODEL}")
    print(f"Display Size: {config.DISPLAY_SIZE}")
    print(f"Enable ReID: {config.ENABLE_REID}")


def test_hailo_device():
    """Test HAILO device detection"""
    print("\n=== Testing HAILO Device ===")
    
    try:
        from hailo_platform import VDevice
        
        # Try to create VDevice
        params = VDevice.create_params()
        with VDevice(params) as target:
            print("✓ HAILO device: Available and accessible")
            
            # Get device info
            try:
                device_info = target.get_device_info()
                print(f"  Device: {device_info.device_id}")
                print(f"  Architecture: {device_info.arch_name}")
            except:
                print("  Device info: Not available")
                
    except Exception as e:
        print(f"✗ HAILO device: {e}")
        return False
    
    return True


def test_camera_access():
    """Test camera access"""
    print("\n=== Testing Camera Access ===")
    
    # Test Pi5 camera module
    try:
        from pi5_camera import create_pi5_camera, print_camera_info
        
        print("Available cameras:")
        print_camera_info()
        
        # Try to create camera instance
        camera = create_pi5_camera(
            camera_index="/dev/video0",
            resolution=(640, 480),
            fps=30
        )
        
        print("✓ Pi5 Camera: Successfully created")
        
        # Try to read a frame
        ret, frame = camera.read()
        if ret and frame is not None:
            print(f"✓ Camera Frame: {frame.shape}")
            camera.release()
            return True
        else:
            print("✗ Camera Frame: Failed to read")
            camera.release()
            return False
            
    except Exception as e:
        print(f"✗ Pi5 Camera: {e}")
    
    # Fallback to OpenCV camera test
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✓ OpenCV Camera: {frame.shape}")
                cap.release()
                return True
            else:
                print("✗ OpenCV Camera: Failed to read frame")
        else:
            print("✗ OpenCV Camera: Failed to open")
        cap.release()
    except Exception as e:
        print(f"✗ OpenCV Camera: {e}")
    
    return False


def test_hailo_model():
    """Test HAILO model loading"""
    print("\n=== Testing HAILO Model ===")
    
    try:
        from hailo_yolo_detector import HailoYOLODetector, download_hailo_yolo_model
        
        # Check if model exists
        model_path = "yolov8n.hef"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Attempting to download...")
            try:
                model_path = download_hailo_yolo_model("yolov8n")
                print(f"✓ Model downloaded: {model_path}")
            except Exception as e:
                print(f"✗ Model download failed: {e}")
                return False
        
        # Try to load model
        try:
            detector = HailoYOLODetector(model_path)
            print("✓ HAILO Model: Successfully loaded")
            
            # Test model info
            if hasattr(detector, 'input_vstream_info') and detector.input_vstream_info:
                print(f"  Input shape: {detector.input_vstream_info.shape}")
            if hasattr(detector, 'output_vstream_info') and detector.output_vstream_info:
                print(f"  Output shape: {detector.output_vstream_info.shape}")
            
            return True
            
        except Exception as e:
            print(f"✗ HAILO Model loading failed: {e}")
            return False
            
    except ImportError:
        print("✗ HAILO YOLO Detector not available")
        return False


def test_tracker_initialization():
    """Test tracker initialization"""
    print("\n=== Testing Tracker Initialization ===")
    
    try:
        from yolo_botsort_tracker import YOLOBoTSORTTracker
        
        # Initialize tracker
        tracker = YOLOBoTSORTTracker()
        print("✓ Tracker: Successfully initialized")
        
        # Test configuration
        print(f"  Detector type: {'HAILO' if tracker.config.USE_HAILO else 'YOLO'}")
        print(f"  Model: {tracker.config.YOLO_MODEL}")
        print(f"  Device: {tracker.device}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tracker initialization failed: {e}")
        return False


def test_inference():
    """Test inference pipeline"""
    print("\n=== Testing Inference Pipeline ===")
    
    try:
        from yolo_botsort_tracker import YOLOBoTSORTTracker
        from pi5_camera import create_pi5_camera
        
        # Initialize tracker
        tracker = YOLOBoTSORTTracker()
        
        # Create test frame
        camera = create_pi5_camera(
            camera_index="/dev/video0",
            resolution=(640, 480),
            fps=30
        )
        
        print("Testing inference pipeline...")
        
        # Test a few frames
        for i in range(5):
            ret, frame = camera.read()
            if not ret:
                continue
            
            # Process frame
            start_time = time.time()
            processed_frame = tracker.process_frame(frame, i)
            process_time = time.time() - start_time
            
            print(f"  Frame {i}: {process_time*1000:.1f}ms")
            
            # Get performance stats
            if hasattr(tracker, 'performance_monitor'):
                stats = tracker.performance_monitor.get_statistics()
                if stats.get('fps', 0) > 0:
                    print(f"    FPS: {stats['fps']:.1f}")
            
            if tracker.config.USE_HAILO and hasattr(tracker.detector, 'get_performance_stats'):
                hailo_stats = tracker.detector.get_performance_stats()
                if hailo_stats.get('fps', 0) > 0:
                    print(f"    HAILO FPS: {hailo_stats['fps']:.1f}")
        
        print("✓ Inference pipeline: Working")
        camera.release()
        return True
        
    except Exception as e:
        print(f"✗ Inference pipeline failed: {e}")
        return False


def main():
    """Main test function"""
    print("HAILO Integration Test for Bus Counter")
    print("=" * 50)
    
    # Run tests
    test_results = []
    
    # Test imports
    hailo_available, pi5_camera_available = test_imports()
    test_results.append(("Module Imports", True))  # Always pass if we get here
    
    # Test platform detection
    test_platform_detection()
    test_results.append(("Platform Detection", True))
    
    # Test HAILO device
    if hailo_available:
        hailo_device_ok = test_hailo_device()
        test_results.append(("HAILO Device", hailo_device_ok))
    else:
        print("\n=== Skipping HAILO Device Test (Not Available) ===")
        test_results.append(("HAILO Device", None))
    
    # Test camera access
    camera_ok = test_camera_access()
    test_results.append(("Camera Access", camera_ok))
    
    # Test HAILO model
    if hailo_available:
        model_ok = test_hailo_model()
        test_results.append(("HAILO Model", model_ok))
    else:
        print("\n=== Skipping HAILO Model Test (Not Available) ===")
        test_results.append(("HAILO Model", None))
    
    # Test tracker initialization
    tracker_ok = test_tracker_initialization()
    test_results.append(("Tracker Initialization", tracker_ok))
    
    # Test inference
    if camera_ok and tracker_ok:
        inference_ok = test_inference()
        test_results.append(("Inference Pipeline", inference_ok))
    else:
        print("\n=== Skipping Inference Test (Prerequisites Failed) ===")
        test_results.append(("Inference Pipeline", None))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results:
        total += 1
        if result is True:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"✗ {test_name}: FAILED")
        else:
            print(f"- {test_name}: SKIPPED")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HAILO integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)