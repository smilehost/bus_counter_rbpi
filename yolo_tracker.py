import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import argparse
import os
import sys

from config import Config
from trackers import BoTSORT, ByteTrack
from utils import Visualizer, BusCounter, PerformanceMonitor, resize_frame
from class_names import get_class_name, COCO_CLASS_NAMES

# Enforce use of enhanced camera module
from enhanced_camera import EnhancedCamera, create_enhanced_camera, test_all_methods
ENHANCED_CAMERA_AVAILABLE = True

class YOLOTracker:
    """Main tracking class combining YOLO detection with configurable tracking"""
    
    def __init__(self, config_path=None):
        # Load configuration
        self.config = Config() if config_path is None else self._load_config(config_path)
        
        # Load detection zone configuration if available
        self.config.load_zone_config()
        
        # Device setup - use device from config (must be done before detector initialization)
        self.device = self.config.DEVICE
        
        # Initialize detector (HAILO or regular YOLO)
        self.detector = self._initialize_detector()
        
        # Initialize tracker based on config
        tracker_type = getattr(self.config, 'TRACKER_TYPE', 'botsort')
        self.tracker = self._create_tracker(tracker_type)
        
        # Initialize utilities
        self.visualizer = Visualizer(self.config)
        self.bus_counter = BusCounter(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Video writer
        self.video_writer = None
        
        # Camera setup
        self.camera = None
        
        print(f"YOLO Tracker initialized")
        print(f"Detector Type: YOLO")
        print(f"Model: {self.config.YOLO_MODEL}")
        print(f"Device: {self.device}")
        print(f"Tracker: {tracker_type}")
        
        # Print device information
        self.config.print_device_info()
        
        # Print available YOLO classes
        self.print_available_classes()
    
    def _create_tracker(self, tracker_type):
        """Factory method to create tracker instance based on config
        
        Args:
            tracker_type: String specifying tracker type ('botsort' or 'bytetrack')
            
        Returns:
            Tracker instance
        """
        tracker_config = self._get_tracker_config(tracker_type)
        
        if tracker_type == 'botsort':
            return BoTSORT(tracker_config)
        elif tracker_type == 'bytetrack':
            return ByteTrack(tracker_config)
        else:
            raise ValueError(f"Unknown tracker type: {tracker_type}. "
                            f"Supported types: 'botsort', 'bytetrack'")

    def _get_tracker_config(self, tracker_type):
        """Get configuration for specific tracker type
        
        Args:
            tracker_type: String specifying tracker type
            
        Returns:
            Configuration dictionary for the tracker
        """
        if tracker_type == 'botsort':
            config = self.config.BOTSORT_TRACKER.copy()
            if hasattr(self.config, 'ENABLE_REID'):
                config['with_reid'] = self.config.ENABLE_REID
        elif tracker_type == 'bytetrack':
            config = self.config.BYETRACK_TRACKER.copy() if hasattr(self.config, 'BYETRACK_TRACKER') else {}
        else:
            config = {}
        
        config['device'] = self.device
        config['reid_device'] = self.config.REID_DEVICE
        return config
    
    def _initialize_detector(self):
        """Initialize detector (YOLO)"""
        try:
            detector = YOLO(self.config.YOLO_MODEL)
            # Move model to the configured device
            if self.device == "cuda":
                detector.to('cuda')
                print(f"YOLO detector moved to CUDA")
            elif self.device == "cpu":
                detector.to('cpu')
                print(f"YOLO detector moved to CPU")
            
            print(f"YOLO detector initialized: {self.config.YOLO_MODEL}")
            return detector
        except Exception as e:
            print(f"Failed to initialize YOLO detector with model {self.config.YOLO_MODEL}: {e}")
            # Try with a default model as last resort
            try:
                default_model = "yolo11n.pt"
                print(f"Trying with default model: {default_model}")
                detector = YOLO(default_model)
                # Move model to the configured device
                if self.device == "cuda":
                    detector.to('cuda')
                elif self.device == "cpu":
                    detector.to('cpu')
                
                self.config.YOLO_MODEL = default_model
                print(f"YOLO detector initialized with default model: {default_model}")
                return detector
            except Exception as e2:
                print(f"Failed to initialize YOLO detector with default model: {e2}")
                raise RuntimeError(f"Could not initialize any detector: YOLO with {self.config.YOLO_MODEL} failed, YOLO with default failed")
    
    def _load_config(self, config_path):
        """Load configuration from file"""
        # This would load from a JSON/YAML file in a real implementation
        return Config()
    
    def process_video(self, video_source, output_path=None):
        """Process video file or camera stream"""
        
        # Initialize camera using enhanced camera module (enforced)
        if isinstance(video_source, (str, int)) and (str(video_source).startswith('/dev/video') or isinstance(video_source, int)):
            # Always use enhanced camera
            try:
                self.camera = create_enhanced_camera(
                    width=self.config.CAM_WIDTH,
                    height=self.config.CAM_HEIGHT,
                    fps=self.config.CAM_FPS,
                    preferred_method=self.config.PREFERRED_CAMERA_METHOD
                )
            except Exception as e:
                print(f"[ERROR] Failed to create enhanced camera: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            # Regular OpenCV camera or video file
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"[ERROR] Could not open video source {video_source}")
                return
            self.camera = cap
        
        # Get video properties
        try:
            if hasattr(self.camera, 'get_camera_info'):
                # Pi5 camera
                camera_info = self.camera.get_camera_info()
                fps = camera_info.get('target_fps', 30)
                width, height = camera_info.get('resolution', self.config.DISPLAY_SIZE)
            else:
                # OpenCV camera
                fps = int(self.camera.get(cv2.CAP_PROP_FPS))
                width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception as e:
            print(f"[ERROR] Failed to get video properties: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if needed
        if self.config.SAVE_VIDEO and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, 
                                             self.config.DISPLAY_SIZE)
        
        # Process frames
        frame_count = 0
        total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT)) if hasattr(self.camera, 'get') else 0
        
        try:
            # Test initial frame read
            if hasattr(self.camera, 'read'):
                ret, test_frame = self.camera.read()
            else:
                return
                
            while True:
                # Read frame from camera
                if hasattr(self.camera, 'read'):
                    # Pi5 camera or OpenCV camera
                    ret, frame = self.camera.read()
                else:
                    # Fallback
                    ret, frame = False, None
                
                if not ret or frame is None:
                    break
                
                # Start performance timer
                self.performance_monitor.start_frame_timer()
                
                # Process frame
                processed_frame = self.process_frame(frame, frame_count)
                
                # End performance timer
                self.performance_monitor.end_frame_timer()
                
                # Display frame
                if self.config.SHOW_VIDEO:
                    cv2.imshow('YOLO Tracking', processed_frame)
                    
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
            if hasattr(self.camera, 'release'):
                self.camera.release()
            elif self.camera is not None:
                self.camera.release()
                
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def _is_detection_in_zone(self, bbox, frame_width, frame_height):
        """Check if a detection's bounding box is within the detection zone
        
        Args:
            bbox: Bounding box in pixel coordinates [x1, y1, x2, y2]
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            True if the detection is considered to be in the detection zone
        """
        zone = self.config.DETECTION_ZONE
        
        # Convert normalized zone coordinates to pixel coordinates
        zone_x1 = zone['x1'] * frame_width
        zone_y1 = zone['y1'] * frame_height
        zone_x2 = zone['x2'] * frame_width
        zone_y2 = zone['y2'] * frame_height
        
        # Get detection bounding box coordinates
        det_x1, det_y1, det_x2, det_y2 = bbox
        
        # Calculate center point of detection
        det_center_x = (det_x1 + det_x2) / 2
        det_center_y = (det_y1 + det_y2) / 2
        
        # Check if center is within zone
        if (zone_x1 <= det_center_x <= zone_x2 and
            zone_y1 <= det_center_y <= zone_y2):
            return True
        
        # Calculate overlap area for partial detection
        overlap_x1 = max(det_x1, zone_x1)
        overlap_y1 = max(det_y1, zone_y1)
        overlap_x2 = min(det_x2, zone_x2)
        overlap_y2 = min(det_y2, zone_y2)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return False  # No overlap
        
        # Calculate areas
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
        
        # Check if overlap exceeds margin threshold
        if det_area > 0:
            overlap_ratio = overlap_area / det_area
            return overlap_ratio >= self.config.DETECTION_ZONE_MARGIN
        
        return False
    
    def _filter_detections_by_zone(self, detections, frame_width, frame_height):
        """Filter detections to only include those within the detection zone
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, confidence, class_id, ...]
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Filtered array of detections
        """
        if not self.config.DETECTION_ZONE_ENABLED:
            return detections
        
        if len(detections) == 0:
            return detections
        
        filtered_detections = []
        for detection in detections:
            bbox = detection[:4]  # [x1, y1, x2, y2]
            if self._is_detection_in_zone(bbox, frame_width, frame_height):
                filtered_detections.append(detection)
        
        return np.array(filtered_detections) if filtered_detections else np.array([])
    
    def _filter_tracks_by_zone(self, tracks, frame_width, frame_height):
        """Filter tracks to only include those within the detection zone
        
        Args:
            tracks: List of track objects
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Filtered list of tracks
        """
        if not self.config.DETECTION_ZONE_ENABLED:
            return tracks
        
        filtered_tracks = []
        for track in tracks:
            bbox = track.bbox
            if self._is_detection_in_zone(bbox, frame_width, frame_height):
                filtered_tracks.append(track)
            else:
                # Mark track as lost if it's outside the zone
                # This will cause the tracker to eventually delete it
                track.state = 2  # Lost state
        
        return filtered_tracks
    
    def process_frame(self, frame, frame_count):
        """Process a single frame"""
        # Initialize variables
        tracks = []
        
        # Trackers handle prediction internally
        
        # Process every frame for more stable tracking
        # Start detection timer
        detection_start = time.time()
        
        # Detection (YOLO)
        try:
            # Use the correct YOLO API - results() method instead of direct call
            results = self.detector.predict(frame, conf=self.config.YOLO_CONFIDENCE,
                                     iou=self.config.YOLO_IOU_THRESHOLD,
                                     classes=self.config.YOLO_CLASSES,
                                     verbose=False)
            
            detection_time = time.time() - detection_start
            # Extract detections
            detections = self._extract_detections(results[0])
        except Exception as e:
            raise e
        
        self.performance_monitor.add_detection_time(detection_time)
        
        # DEBUG: Log detection results
        if frame_count % 30 == 0:  # Log every 30 frames
            print(f"[DEBUG] Detection time: {detection_time*1000:.2f}ms")
            print(f"[DEBUG] Detections found: {len(detections)}")
            if len(detections) > 0:
                print(f"[DEBUG] Detection sample: {detections[0]}")
                # Check if detections are valid
                for i, det in enumerate(detections[:3]):  # Check first 3 detections
                    x1, y1, x2, y2, conf, cls = det
                    print(f"[DEBUG] Detection {i}: bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), conf={conf:.3f}, class={int(cls)}")
                    
                    # Check for invalid coordinates
                    if x2 <= x1 or y2 <= y1:
                        print(f"[WARNING] Invalid bbox in detection {i}: x2<=x1 or y2<=y1")
                    if conf <= 0:
                        print(f"[WARNING] Invalid confidence in detection {i}: {conf}")
                    if not (0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1] and
                           0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0]):
                        print(f"[WARNING] Bbox out of frame bounds in detection {i}")
            else:
                print("[DEBUG] No detections - checking possible causes:")
                print(f"[DEBUG] - Frame too dark/bright? min={frame.min()}, max={frame.max()}")
                print(f"[DEBUG] - Color space issue? BGR format expected")
                print(f"[DEBUG] - Confidence too high? threshold={self.config.YOLO_CONFIDENCE}")
        
        # Filter detections by detection zone BEFORE sending to tracker
        height = frame.shape[0]
        width = frame.shape[1]
        
        if self.config.DETECTION_ZONE_ENABLED:
            detections_before_filter = len(detections)
            detections = self._filter_detections_by_zone(detections, width, height)
            if frame_count % 30 == 0:
                print(f"[DEBUG] Detection zone filter: {detections_before_filter} -> {len(detections)} detections")
        
        # Extract ReID features if tracker supports it
        reid_features = None
        if hasattr(self.tracker, 'extract_reid_features') and len(detections) > 0:
            try:
                boxes = detections[:, :4]  # Extract bounding boxes
                reid_features = self.tracker.extract_reid_features(frame, boxes)
                if reid_features is not None and frame_count % 30 == 0:
                    print(f"[DEBUG] Extracted ReID features: {reid_features.shape}")
            except Exception as e:
                print(f"[ERROR] ReID feature extraction failed: {e}")
                reid_features = None
        
        # Append ReID features to detections if available
        if reid_features is not None:
            # detections format: [x1, y1, x2, y2, confidence, class_id, reid_features...]
            detections_with_reid = []
            for i, detection in enumerate(detections):
                if i < len(reid_features):
                    detection_with_reid = np.concatenate([detection, reid_features[i]])
                    detections_with_reid.append(detection_with_reid)
                else:
                    detections_with_reid.append(detection)
            detections = np.array(detections_with_reid) if detections_with_reid else detections
        
        # Start tracking timer
        tracking_start = time.time()
        
        # Tracking
        tracks = self.tracker.update(detections, frame)
        
        tracking_time = time.time() - tracking_start
        self.performance_monitor.add_tracking_time(tracking_time)
        
        # Filter tracks by detection zone AFTER tracker update
        # This stops tracking objects that have left the zone
        if self.config.DETECTION_ZONE_ENABLED:
            tracks_before_filter = len(tracks)
            tracks = self._filter_tracks_by_zone(tracks, width, height)
            if frame_count % 30 == 0:
                print(f"[DEBUG] Track zone filter: {tracks_before_filter} -> {len(tracks)} tracks")
        
        # DEBUG: Log tracking results
        if frame_count % 30 == 0:  # Log every 30 frames
            print(f"[DEBUG] Tracking time: {tracking_time*1000:.2f}ms")
            print(f"[DEBUG] Active tracks: {len(tracks)}")
            for i, track in enumerate(tracks[:3]):  # Check first 3 tracks
                print(f"[DEBUG] Track {i}: ID={track.track_id}, state={track.state}, age={track.age}, hits={track.hits}")
                if hasattr(track, 'bbox') and track.bbox is not None:
                    x1, y1, x2, y2 = track.bbox
                    print(f"[DEBUG]   bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        
        # Update bus counter
        height = frame.shape[0]
        width = frame.shape[1]
        self.bus_counter.update(tracks, height, width)
        
        # Resize frame for display
        display_frame = resize_frame(frame, self.config.DISPLAY_SIZE)
        
        # Draw tracking results (even on skipped frames)
        annotated_frame = self.visualizer.draw_tracking_results(
            display_frame, tracks, original_frame_size=frame.shape[:2], show_trails=True, show_ids=True
        )
        
        # Draw detection zone if enabled
        self.visualizer.draw_detection_zone(annotated_frame)
        
        # Draw counting line - calculate correct position accounting for resize scaling and offset
        # The counting line position is based on the original frame height
        if self.config.COUNTING_LINE_Y is None:
            counting_line_y = frame.shape[0] // 2
        else:
            counting_line_y = self.config.COUNTING_LINE_Y
        
        # Calculate the scale factor used by resize_frame()
        frame_height, frame_width = frame.shape[:2]
        target_width, target_height = self.config.DISPLAY_SIZE
        scale = min(target_width / frame_width, target_height / frame_height)
        
        # Calculate new dimensions after resize
        new_height = int(frame_height * scale)
        
        # Calculate y_offset (letterboxing) used by resize_frame()
        y_offset = (target_height - new_height) // 2
        
        # Calculate the correct visual line position on the display frame
        line_y = int(counting_line_y * scale) + y_offset
        
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
    
    def print_camera_info(self):
        """Print available camera information"""
        # Always use enhanced camera methods
        working_methods = test_all_methods()
        print(f"Working camera methods: {working_methods}")
        
        # Test enhanced camera with best method
        if working_methods:
            print(f"\nTesting enhanced camera with method: {working_methods[0]}")
            try:
                with create_enhanced_camera(preferred_method=working_methods[0]) as cam:
                    info = cam.get_camera_info()
                    print(f"Camera info: {info}")
            except Exception as e:
                print(f"Error testing enhanced camera: {e}")
    
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
    
    parser = argparse.ArgumentParser(description='YOLO Object Tracking')
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
    tracker = YOLOTracker()
    
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
