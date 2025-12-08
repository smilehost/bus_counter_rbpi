#!/usr/bin/env python3
"""
Example usage of YOLO-BoTSORT tracker for bus counting
This script demonstrates different ways to use the tracking system
"""

import cv2
import time
from yolo_botsort_tracker import YOLOBoTSORTTracker
from config import Config

def example_webcam_tracking():
    """Example: Real-time tracking from webcam"""
    print("=== Webcam Tracking Example ===")
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Configure for webcam
    tracker.config.VIDEO_SOURCE = 0  # Webcam
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = False
    tracker.config.YOLO_CONFIDENCE = 0.6
    tracker.config.PROCESS_EVERY_N_FRAMES = 1  # Process every frame for better visualization
    
    # Test if webcam is available
    cap = cv2.VideoCapture(tracker.config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open webcam (source {tracker.config.VIDEO_SOURCE})")
        print("Please check:")
        print("1. Webcam is connected")
        print("2. Webcam is not being used by another application")
        print("3. Try different camera indices (0, 1, 2, etc.)")
        cap.release()
        return
    
    cap.release()
    
    print("Starting webcam tracking...")
    print("Press 'q' to quit")
    
    # Process webcam stream
    tracker.process_video(tracker.config.VIDEO_SOURCE)


def find_working_camera():
    """Find a working camera index"""
    for index in range(5):  # Try camera indices 0-4
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Try to read a frame
            ret, _ = cap.read()
            cap.release()
            if ret:
                return index
    return None


def example_webcam_tracking_with_fallback():
    """Example: Real-time tracking from webcam with fallback"""
    print("=== Webcam Tracking Example (with fallback) ===")
    
    # Find working camera
    camera_index = find_working_camera()
    if camera_index is None:
        print("Error: No working camera found")
        print("Please check:")
        print("1. Webcam is connected")
        print("2. Webcam drivers are installed")
        print("3. Webcam is not being used by another application")
        return
    
    print(f"Using camera index: {camera_index}")
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Configure for webcam
    tracker.config.VIDEO_SOURCE = camera_index
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = False
    tracker.config.YOLO_CONFIDENCE = 0.6
    tracker.config.PROCESS_EVERY_N_FRAMES = 1  # Process every frame for better visualization
    
    print("Starting webcam tracking...")
    print("Press 'q' to quit")
    
    # Process webcam stream
    tracker.process_video(tracker.config.VIDEO_SOURCE)


def example_video_file_tracking(video_path="test_video.mp4"):
    """Example: Process video file and save results"""
    print("=== Video File Tracking Example ===")
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Configure for video file
    tracker.config.VIDEO_SOURCE = video_path
    tracker.config.OUTPUT_VIDEO = "output_" + video_path
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = True
    tracker.config.YOLO_CONFIDENCE = 0.5
    
    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {tracker.config.OUTPUT_VIDEO}")
    
    # Process video file
    tracker.process_video(tracker.config.VIDEO_SOURCE, tracker.config.OUTPUT_VIDEO)


def example_batch_processing(video_files):
    """Example: Process multiple video files in batch"""
    print("=== Batch Processing Example ===")
    
    for video_file in video_files:
        print(f"\nProcessing: {video_file}")
        
        # Initialize tracker for each video
        tracker = YOLOBoTSORTTracker()
        
        # Configure
        tracker.config.VIDEO_SOURCE = video_file
        tracker.config.OUTPUT_VIDEO = f"output_{video_file}"
        tracker.config.SHOW_VIDEO = False  # No display for batch processing
        tracker.config.SAVE_VIDEO = True
        tracker.config.YOLO_CONFIDENCE = 0.5
        
        # Process video
        tracker.process_video(tracker.config.VIDEO_SOURCE, tracker.config.OUTPUT_VIDEO)
        
        print(f"Completed: {video_file}")


def example_custom_settings():
    """Example: Use custom tracking settings"""
    print("=== Custom Settings Example ===")
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Custom YOLO settings
    tracker.config.YOLO_MODEL = "yolov8s.pt"  # Use medium model
    tracker.config.YOLO_CONFIDENCE = 0.7
    tracker.config.YOLO_IOU_THRESHOLD = 0.5
    
    # Custom BoTSORT settings
    tracker.config.BOTSORT_TRACKER['track_high_thresh'] = 0.8
    tracker.config.BOTSORT_TRACKER['track_low_thresh'] = 0.2
    tracker.config.BOTSORT_TRACKER['track_buffer'] = 50
    tracker.config.BOTSORT_TRACKER['with_reid'] = True
    
    # Custom counting settings
    tracker.config.COUNTING_DIRECTION = "both"  # Count both directions
    tracker.config.MIN_TRACK_AGE = 15  # Require more frames before counting
    
    # Configure video source
    tracker.config.VIDEO_SOURCE = 0  # Webcam
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = False
    
    print("Using custom tracking settings:")
    print(f"  YOLO Model: {tracker.config.YOLO_MODEL}")
    print(f"  Confidence: {tracker.config.YOLO_CONFIDENCE}")
    print(f"  ReID Enabled: {tracker.config.BOTSORT_TRACKER['with_reid']}")
    print(f"  Counting Direction: {tracker.config.COUNTING_DIRECTION}")
    
    # Process video
    tracker.process_video(tracker.config.VIDEO_SOURCE)


def example_performance_optimization():
    """Example: Performance optimization settings"""
    print("=== Performance Optimization Example ===")
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Performance settings
    tracker.config.YOLO_MODEL = "yolov8n.pt"  # Use nano model for speed
    tracker.config.PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
    tracker.config.MAX_DETECTIONS = 50  # Limit detections
    
    # Smaller display size
    tracker.config.DISPLAY_SIZE = (640, 480)
    
    # Disable features for speed
    tracker.config.BOTSORT_TRACKER['with_reid'] = False
    
    # Configure video source
    tracker.config.VIDEO_SOURCE = 0  # Webcam
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = False
    
    print("Performance optimization settings:")
    print(f"  Processing every {tracker.config.PROCESS_EVERY_N_FRAMES} frames")
    print(f"  Max detections: {tracker.config.MAX_DETECTIONS}")
    print(f"  Display size: {tracker.config.DISPLAY_SIZE}")
    print(f"  ReID disabled: {not tracker.config.BOTSORT_TRACKER['with_reid']}")
    
    # Process video
    tracker.process_video(tracker.config.VIDEO_SOURCE)


def example_raspberry_pi_settings():
    """Example: Settings optimized for Raspberry Pi 5"""
    print("=== Raspberry Pi 5 Optimization Example ===")
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Pi 5 optimized settings
    tracker.config.YOLO_MODEL = "yolov8n.pt"  # Use nano model
    tracker.config.YOLO_CONFIDENCE = 0.6
    tracker.config.PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame
    tracker.config.DISPLAY_SIZE = (800, 600)
    
    # Reduce tracking complexity
    tracker.config.BOTSORT_TRACKER['track_buffer'] = 20
    tracker.config.BOTSORT_TRACKER['with_reid'] = False  # Disable ReID for performance
    tracker.config.BOTSORT_TRACKER['match_thresh'] = 0.7
    
    # Reduce counting requirements
    tracker.config.MIN_TRACK_AGE = 8
    
    # Configure video source
    tracker.config.VIDEO_SOURCE = 0  # Webcam or Pi camera
    tracker.config.SHOW_VIDEO = True
    tracker.config.SAVE_VIDEO = False
    
    print("Raspberry Pi 5 optimized settings:")
    print(f"  Model: {tracker.config.YOLO_MODEL}")
    print(f"  Frame skip: {tracker.config.PROCESS_EVERY_N_FRAMES}")
    print(f"  ReID disabled: {not tracker.config.BOTSORT_TRACKER['with_reid']}")
    print(f"  Min track age: {tracker.config.MIN_TRACK_AGE}")
    
    # Process video
    tracker.process_video(tracker.config.VIDEO_SOURCE)


def main():
    """Main function with menu"""
    print("YOLO-BoTSORT Tracker Examples")
    print("=" * 40)
    print("1. Webcam tracking")
    print("2. Video file processing")
    print("3. Custom settings")
    print("4. Performance optimization")
    print("5. Raspberry Pi 5 optimization")
    print("6. Run all examples (if available)")
    print("=" * 40)
    
    try:
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            example_webcam_tracking_with_fallback()
        elif choice == "2":
            video_path = input("Enter video file path (or press Enter for default): ").strip()
            if not video_path:
                video_path = "test_video.mp4"
            example_video_file_tracking(video_path)
        elif choice == "3":
            example_custom_settings()
        elif choice == "4":
            example_performance_optimization()
        elif choice == "5":
            example_raspberry_pi_settings()
        elif choice == "6":
            print("Running multiple examples...")
            # Note: This would require actual video files
            print("Note: Batch processing requires actual video files")
        else:
            print("Invalid choice. Running default webcam example...")
            example_webcam_tracking()
    
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Error running example: {e}")


if __name__ == "__main__":
    main()