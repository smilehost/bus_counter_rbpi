#!/usr/bin/env python3
"""
Test script for GStreamer camera implementation
"""

import cv2
import time
from pi5_camera import Pi5Camera, find_available_cameras

def test_camera_detection():
    """Test camera detection"""
    print("Testing camera detection...")
    cameras = find_available_cameras()
    
    print(f"Found {len(cameras)} cameras:")
    for i, cam in enumerate(cameras):
        print(f"  {i}: {cam['name']} (Type: {cam['type']}, Index: {cam['index']})")
    
    return cameras

def test_camera_initialization(camera_type="auto"):
    """Test camera initialization and frame capture"""
    print(f"\nTesting camera initialization with type='{camera_type}'...")
    
    try:
        # Initialize camera
        camera = Pi5Camera(camera_type=camera_type, resolution=(640, 480), fps=30)
        
        # Get camera info
        info = camera.get_camera_info()
        print(f"Camera info: {info}")
        
        # Test frame capture
        print("Testing frame capture...")
        frame_count = 0
        start_time = time.time()
        
        for i in range(30):  # Capture 30 frames
            ret, frame = camera.read()
            if ret:
                frame_count += 1
                if i % 10 == 0:
                    print(f"  Captured frame {i+1}: {frame.shape}")
            else:
                print(f"  Failed to capture frame {i+1}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if frame_count > 0:
            print(f"Successfully captured {frame_count} frames in {elapsed:.2f} seconds")
            print(f"Average FPS: {frame_count/elapsed:.2f}")
        else:
            print("No frames captured successfully")
        
        # Release camera
        camera.release()
        return frame_count > 0
        
    except Exception as e:
        print(f"Error testing camera: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("GStreamer Camera Test")
    print("=" * 60)
    
    # Test camera detection
    cameras = test_camera_detection()
    
    if not cameras:
        print("No cameras found. Please check camera connections and drivers.")
        return
    
    # Test each available camera type
    camera_types = set(cam['type'] for cam in cameras)
    
    for cam_type in camera_types:
        success = test_camera_initialization(cam_type)
        if success:
            print(f"✓ Camera type '{cam_type}' test passed")
        else:
            print(f"✗ Camera type '{cam_type}' test failed")
    
    print("\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    main()