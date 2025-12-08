import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import argparse
import os
import sys

from config import Config
from botsort_tracker import BoTSORT
from utils import Visualizer, BusCounter, PerformanceMonitor, resize_frame
from class_names import get_class_name, COCO_CLASS_NAMES

class YOLOBoTSORTTracker:
    """Main tracking class combining YOLO detection with BoTSORT tracking"""
    
    def __init__(self, config_path=None):
        # Load configuration
        self.config = Config() if config_path is None else self._load_config(config_path)
        
        # Initialize YOLO model
        self.yolo_model = YOLO(self.config.YOLO_MODEL)
        
        # Initialize BoTSORT tracker
        self.tracker = BoTSORT(self.config.BOTSORT_TRACKER)
        
        # Initialize utilities
        self.visualizer = Visualizer(self.config)
        self.bus_counter = BusCounter(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Video writer
        self.video_writer = None
        
        # Device setup
        self.device = self._setup_device()
        
        print(f"YOLO-BoTSORT Tracker initialized")
        print(f"YOLO Model: {self.config.YOLO_MODEL}")
        print(f"Device: {self.device}")
        print(f"ReID Enabled: {self.config.BOTSORT_TRACKER['with_reid']}")
        
        # Print available YOLO classes
        self.print_available_classes()
    
    def _setup_device(self):
        """Setup computation device"""
        if torch.cuda.is_available():
            device = "cuda"
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print("GPU not available, using CPU")
        
        return device
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        # This would load from a JSON/YAML file in a real implementation
        return Config()
    
    def process_video(self, video_source, output_path=None):
        """Process video file or camera stream"""
        # Open video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if needed
        if self.config.SAVE_VIDEO and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                             self.config.DISPLAY_SIZE)
        
        # Process frames
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Start performance timer
                self.performance_monitor.start_frame_timer()
                
                # Process frame
                processed_frame = self.process_frame(frame, frame_count)
                
                # End performance timer
                self.performance_monitor.end_frame_timer()
                
                # Display frame
                if self.config.SHOW_VIDEO:
                    cv2.imshow('YOLO-BoTSORT Tracking', processed_frame)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Write frame to output video
                if self.video_writer:
                    self.video_writer.write(processed_frame)
                
                frame_count += 1
                
                # Print progress
                if frame_count % 30 == 0:
                    stats = self.performance_monitor.get_statistics()
                    counts = self.bus_counter.get_counts()
                    print(f"Frame {frame_count}/{total_frames} | "
                          f"FPS: {stats.get('fps', 0):.1f} | "
                          f"Count: {counts['total']} | "
                          f"Active Tracks: {len(self.tracker.tracks)}")
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def process_frame(self, frame, frame_count):
        """Process a single frame"""
        # Initialize variables
        tracks = []
        
        # Only run detection and tracking every N frames for performance
        if frame_count % self.config.PROCESS_EVERY_N_FRAMES == 0:
            # Start detection timer
            detection_start = time.time()
            
            # YOLO detection
            results = self.yolo_model(frame, conf=self.config.YOLO_CONFIDENCE,
                                     iou=self.config.YOLO_IOU_THRESHOLD,
                                     classes=self.config.YOLO_CLASSES,
                                     verbose=False)
            
            detection_time = time.time() - detection_start
            self.performance_monitor.add_detection_time(detection_time)
            
            # Extract detections
            detections = self._extract_detections(results[0])
            
            # Start tracking timer
            tracking_start = time.time()
            
            # BoTSORT tracking
            tracks = self.tracker.update(detections, frame)
            
            tracking_time = time.time() - tracking_start
            self.performance_monitor.add_tracking_time(tracking_time)
            
            # Update bus counter
            height = frame.shape[0]
            self.bus_counter.update(tracks, height)
        else:
            # On skipped frames, just get current tracks without updating
            tracks = self.tracker._get_active_tracks()
        
        # Resize frame for display
        display_frame = resize_frame(frame, self.config.DISPLAY_SIZE)
        
        # Draw tracking results (even on skipped frames)
        annotated_frame = self.visualizer.draw_tracking_results(
            display_frame, tracks, show_trails=True, show_ids=True
        )
        
        # Draw counting line
        line_y = int(self.config.DISPLAY_SIZE[1] * 0.5)
        self.visualizer.draw_counting_line(annotated_frame, line_y)
        
        # Draw statistics
        counts = self.bus_counter.get_counts()
        stats = self.performance_monitor.get_statistics()
        
        all_stats = {
            'FPS': f"{stats.get('fps', 0):.1f}",
            'Active Tracks': len(tracks),
            'Count In': counts['count_in'],
            'Count Out': counts['count_out'],
            'Total Count': counts['total'],
            'Frame': frame_count
        }
        
        self.visualizer.draw_statistics(annotated_frame, all_stats)
        
        return annotated_frame
    
    def _extract_detections(self, results):
        """Extract detections from YOLO results"""
        detections = []
        detected_classes = set()
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            
            for i, box in enumerate(boxes):
                # Include class ID in detection: [x1, y1, x2, y2, confidence, class_id]
                detection = np.concatenate([box, [scores[i]], [classes[i]]])
                detections.append(detection)
                
                # Track detected classes for printing
                class_name = get_class_name(classes[i])
                detected_classes.add((classes[i], class_name))
        
        # Print detected classes
        if detected_classes:
            print("Detected classes:")
            for class_id, class_name in sorted(detected_classes):
                print(f"  - {class_name} (ID: {class_id})")
        
        if len(detections) > 0:
            return np.array(detections)
        else:
            return np.array([])
    
    def _print_final_statistics(self):
        """Print final tracking statistics"""
        print("\n" + "="*50)
        print("FINAL TRACKING STATISTICS")
        print("="*50)
    
    def print_available_classes(self):
        """Print all available YOLO classes"""
        print("\nAvailable YOLO Classes:")
        print("="*40)
        for class_id, class_name in COCO_CLASS_NAMES.items():
            if class_id in self.config.YOLO_CLASSES:
                print(f"{class_id:2d}: {class_name} ✓ (enabled)")
            else:
                print(f"{class_id:2d}: {class_name}")
        print("="*40)
        print(f"Enabled classes: {[get_class_name(cid) for cid in self.config.YOLO_CLASSES]}")
        print("="*40)
        
        counts = self.bus_counter.get_counts()
        stats = self.performance_monitor.get_statistics()
        
        print(f"Total Count: {counts['total']}")
        print(f"Count In: {counts['count_in']}")
        print(f"Count Out: {counts['count_out']}")
        print(f"Average FPS: {stats.get('fps', 0):.2f}")
        print(f"Average Frame Time: {stats.get('avg_frame_time', 0):.2f} ms")
        print(f"Average Detection Time: {stats.get('avg_detection_time', 0):.2f} ms")
        print(f"Average Tracking Time: {stats.get('avg_tracking_time', 0):.2f} ms")
        print("="*50)


def main():
    """Main function"""
    # Create default config for argument defaults
    default_config = Config()
    
    parser = argparse.ArgumentParser(description='YOLO-BoTSORT Object Tracking')
    parser.add_argument('--source', type=str, default=str(default_config.VIDEO_SOURCE),
                       help='Video source (camera index or file path)')
    parser.add_argument('--output', type=str, default=default_config.OUTPUT_VIDEO,
                       help='Output video file path')
    parser.add_argument('--model', type=str, default=default_config.YOLO_MODEL,
                       help='YOLO model path')
    parser.add_argument('--conf', type=float, default=default_config.YOLO_CONFIDENCE,
                       help='Detection confidence threshold')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video')
    
    args = parser.parse_args()
    
    # Update config based on arguments
    config = Config()
    config.VIDEO_SOURCE = args.source
    config.OUTPUT_VIDEO = args.output
    config.YOLO_MODEL = args.model
    config.YOLO_CONFIDENCE = args.conf
    config.SAVE_VIDEO = not args.no_save
    config.SHOW_VIDEO = not args.no_display
    
    # Initialize tracker
    tracker = YOLOBoTSORTTracker()
    
    # Update tracker config
    tracker.config.VIDEO_SOURCE = config.VIDEO_SOURCE
    tracker.config.OUTPUT_VIDEO = config.OUTPUT_VIDEO
    tracker.config.YOLO_MODEL = config.YOLO_MODEL
    tracker.config.YOLO_CONFIDENCE = config.YOLO_CONFIDENCE
    tracker.config.SAVE_VIDEO = config.SAVE_VIDEO
    tracker.config.SHOW_VIDEO = config.SHOW_VIDEO
    
    # Process video
    print(f"Starting tracking with source: {config.VIDEO_SOURCE}")
    tracker.process_video(config.VIDEO_SOURCE, config.OUTPUT_VIDEO if config.SAVE_VIDEO else None)


if __name__ == "__main__":
    main()