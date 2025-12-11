#!/usr/bin/env python3
"""
HAILO-optimized example usage for YOLO-BoTSORT tracker on Raspberry Pi 5
This script demonstrates HAILO-accelerated tracking with Pi5 camera integration
"""

import cv2
import time
import os
import sys
from yolo_botsort_tracker import YOLOBoTSORTTracker
from config import Config

# Import HAILO and Enhanced Camera modules
try:
    from hailo_yolo_detector import download_hailo_yolo_model, HAILO_AVAILABLE
    from enhanced_camera import create_enhanced_camera, test_all_methods
    ENHANCED_CAMERA_AVAILABLE = True
except ImportError as e:
    print(f"Error importing HAILO/Enhanced Camera modules: {e}")
    print("Please ensure HAILO SDK and Enhanced Camera modules are properly installed")
    sys.exit(1)


def example_hailo_cam0_tracking():
    """Example: HAILO-accelerated tracking with cam0 on Pi5"""
    print("=== HAILO + cam0 Tracking Example ===")
    
    # Check availability
    if not HAILO_AVAILABLE:
        print("Error: HAILO SDK not available")
        return
    
    if not ENHANCED_CAMERA_AVAILABLE:
        print("Error: Enhanced camera module not available")
        return
    
    # Initialize tracker with HAILO config
    tracker = YOLOBoTSORTTracker()
    
    # Configure for HAILO and Enhanced Camera
    tracker.config.USE_HAILO = True
    tracker.config.VIDEO_SOURCE = 0  # Use enhanced camera
    tracker.config.CAMERA_TYPE = "cam_module"  # Use camera module
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = False
    
    # HAILO-optimized settings
    tracker.config.YOLO_CONFIDENCE = 0.6
    tracker.config.PROCESS_EVERY_N_FRAMES = 1  # Process every frame for max accuracy
    tracker.config.MAX_DETECTIONS = 50
    tracker.config.ENABLE_REID = False  # Disable for performance
    
    # Performance settings for Pi5
    tracker.config.DISPLAY_SIZE = (1024, 768)
    tracker.config.BOTSORT_TRACKER['track_buffer'] = 20
    tracker.config.BOTSORT_TRACKER['with_reid'] = False
    tracker.config.MIN_TRACK_AGE = 8
    
    print("Starting HAILO-accelerated tracking with cam0...")
    print("Press 'q' to quit")
    
    # Process camera stream
    try:
        tracker.process_video(tracker.config.VIDEO_SOURCE)
    except Exception as e:
        print(f"Error during tracking: {e}")


def example_hailo_model_download():
    """Example: Download and test HAILO models"""
    print("=== HAILO Model Download Example ===")
    
    if not HAILO_AVAILABLE:
        print("Error: HAILO SDK not available")
        return
    
    # Available models
    models = ["yolov8n", "yolov8s", "yolov5m", "yolov5s"]
    
    print("Available HAILO models:")
    for i, model in enumerate(models):
        print(f"{i+1}. {model}")
    
    try:
        choice = input(f"Select model to download (1-{len(models)}) or press Enter for yolov8n: ").strip()
        if not choice:
            choice = "1"
        
        model_index = int(choice) - 1
        if 0 <= model_index < len(models):
            selected_model = models[model_index]
            print(f"Downloading {selected_model}...")
            
            # Download model
            model_path = download_hailo_yolo_model(selected_model)
            print(f"Model downloaded: {model_path}")
            
            # Test model with tracker
            print("Testing model...")
            tracker = YOLOBoTSORTTracker()
            tracker.config.USE_HAILO = True
            tracker.config.YOLO_MODEL = model_path
            tracker.config.SHOW_VIDEO = False
            tracker.config.SAVE_VIDEO = False
            
            # Test with a sample frame (if available)
            test_image = "test_frame.jpg"
            if os.path.exists(test_image):
                frame = cv2.imread(test_image)
                if frame is not None:
                    detections, inference_time = tracker.detector.detect(frame)
                    print(f"Test successful: {len(detections)} detections in {inference_time*1000:.1f}ms")
                else:
                    print("Could not load test image")
            else:
                print("No test image found, skipping model test")
        else:
            print("Invalid choice")
            
    except (ValueError, KeyboardInterrupt):
        print("Operation cancelled")


def example_hailo_performance_test():
    """Example: Performance testing with HAILO"""
    print("=== HAILO Performance Test ===")
    
    if not HAILO_AVAILABLE:
        print("Error: HAILO SDK not available")
        return
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    if not tracker.config.USE_HAILO:
        print("Error: HAILO not initialized")
        return
    
    # Configure for performance testing
    tracker.config.SHOW_VIDEO = False
    tracker.config.SAVE_VIDEO = False
    tracker.config.VIDEO_SOURCE = 0  # Use first available camera
    
    print("Running HAILO performance test...")
    print("This will test inference speed with different settings")
    print("Press 'q' to stop the test")
    
    try:
        # Test different frame skip settings
        frame_skips = [1, 2, 3, 5]
        
        for skip in frame_skips:
            print(f"\n--- Testing with PROCESS_EVERY_N_FRAMES = {skip} ---")
            
            tracker.config.PROCESS_EVERY_N_FRAMES = skip
            
            # Run for a short duration
            start_time = time.time()
            frame_count = 0# Initialize camera
            # We add 'and False' to PI5_CAMERA_AVAILABLE to force it to skip to the 'else' block
            # since you are using a USB camera, not the Pi Camera Module.
            # Initialize camera
            if PI5_CAMERA_AVAILABLE: 
                # This should now work because we fixed the environment!
                print("Attempting to load Pi5 Camera Module...")
                camera = create_pi5_camera(
                    camera_index="/dev/video0",
                    resolution=(640, 480),
                    fps=30
                )
            else:
                # Fallback only if the module fails
                print("Pi5 Camera not found, falling back to USB/Default...")
                camera = cv2.VideoCapture(0)
            
            try:
                while time.time() - start_time < 10:  # Test for 10 seconds
                    ret, frame = camera.read()
                    if not ret:
                        break
                    
                    # Process frame
                    processed_frame = tracker.process_frame(frame, frame_count)
                    frame_count += 1
                    
                    # Display if enabled
                    if tracker.config.SHOW_VIDEO:
                        cv2.imshow('Performance Test', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                # Calculate performance
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                
                # Get HAILO performance stats
                if hasattr(tracker.detector, 'get_performance_stats'):
                    hailo_stats = tracker.detector.get_performance_stats()
                    print(f"Results for skip={skip}:")
                    print(f"  Overall FPS: {fps:.1f}")
                    print(f"  HAILO FPS: {hailo_stats.get('fps', 0):.1f}")
                    print(f"  Avg inference time: {hailo_stats.get('avg_inference_time', 0):.1f}ms")
                    print(f"  Min inference time: {hailo_stats.get('min_inference_time', 0):.1f}ms")
                    print(f"  Max inference time: {hailo_stats.get('max_inference_time', 0):.1f}ms")
                else:
                    print(f"Results for skip={skip}: FPS = {fps:.1f}")
                
            finally:
                if hasattr(camera, 'release'):
                    camera.release()
                else:
                    camera.release()
                cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Performance test error: {e}")


def example_hailo_camera_info():
    """Example: Display camera and HAILO information"""
    print("=== HAILO + Camera Info Example ===")
    
    # Print HAILO availability
    print(f"HAILO SDK Available: {HAILO_AVAILABLE}")
    print(f"Enhanced Camera Module Available: {ENHANCED_CAMERA_AVAILABLE}")
    
    # Print camera information
    if ENHANCED_CAMERA_AVAILABLE:
        print("\n--- Camera Information ---")
        print("Testing all camera methods...")
        working_methods = test_all_methods()
        
        if working_methods:
            print(f"Working camera methods: {working_methods}")
            
            # Test enhanced camera with best method
            print(f"\nTesting enhanced camera with method: {working_methods[0]}")
            try:
                with create_enhanced_camera(preferred_method=working_methods[0]) as cam:
                    info = cam.get_camera_info()
                    print(f"Camera info: {info}")
            except Exception as e:
                print(f"Error testing enhanced camera: {e}")
        else:
            print("No working camera methods found!")
    
    # Print HAILO model information
    if HAILO_AVAILABLE:
        print("\n--- HAILO Model Information ---")
        
        # Check for existing models
        hef_files = [f for f in os.listdir('.') if f.endswith('.hef')]
        if hef_files:
            print("Available HAILO models:")
            for model in hef_files:
                print(f"  - {model}")
        else:
            print("No HAILO models found. Use model download example to get models.")
    
    # Print system information
    print("\n--- System Information ---")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    
    try:
        import platform
        print(f"Machine: {platform.machine()}")
        print(f"Processor: {platform.processor()}")
        print(f"Architecture: {platform.architecture()}")
    except:
        pass
    
    # Check OpenCV information
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check for GPU availability
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except:
        print("PyTorch not available")


def main():
    """Main function with HAILO-specific menu"""
    print("HAILO-Optimized YOLO-BoTSORT Tracker Examples")
    print("=" * 50)
    print("1. HAILO + cam0 tracking")
    print("2. Download HAILO models")
    print("3. HAILO performance test")
    print("4. Camera and system info")
    print("5. Run all examples")
    print("=" * 50)
    
    try:
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == "1":
            example_hailo_cam0_tracking()
        elif choice == "2":
            example_hailo_model_download()
        elif choice == "3":
            example_hailo_performance_test()
        elif choice == "4":
            example_hailo_camera_info()
        elif choice == "5":
            print("Running all examples...")
            example_hailo_camera_info()
            example_hailo_model_download()
            example_hailo_performance_test()
            example_hailo_cam0_tracking()
        else:
            print("Invalid choice. Running default HAILO + cam0 example...")
            example_hailo_cam0_tracking()
    
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()