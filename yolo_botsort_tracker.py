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

# Import HAILO and Pi5 camera modules
try:
    from hailo_yolo_detector import HailoYOLODetector, download_hailo_yolo_model
    HAILO_AVAILABLE = True
except ImportError:
    print("Warning: HAILO detector not available. Using regular YOLO.")
    HAILO_AVAILABLE = False

try:
    from pi5_camera import Pi5Camera, create_pi5_camera, print_camera_info, PI5_CAMERA_AVAILABLE
except ImportError:
    print("Warning: Pi5 camera module not available. Using OpenCV.")
    PI5_CAMERA_AVAILABLE = False

class YOLOBoTSORTTracker:
    """Main tracking class combining YOLO detection with BoTSORT tracking"""
    
    def __init__(self, config_path=None):
        # Load configuration
        self.config = Config() if config_path is None else self._load_config(config_path)
        
        # Initialize detector (HAILO or regular YOLO)
        self.detector = self._initialize_detector()
        
        # Initialize BoTSORT tracker
        # Update ReID settings based on config
        tracker_config = self.config.BOTSORT_TRACKER.copy()
        if hasattr(self.config, 'ENABLE_REID'):
            tracker_config['with_reid'] = self.config.ENABLE_REID
        
        self.tracker = BoTSORT(tracker_config)
        
        # Initialize utilities
        self.visualizer = Visualizer(self.config)
        self.bus_counter = BusCounter(self.config)
        self.performance_monitor = PerformanceMonitor()
        
        # Video writer
        self.video_writer = None
        
        # Camera setup
        self.camera = None
        
        # Device setup
        self.device = self._setup_device()
        
        print(f"YOLO-BoTSORT Tracker initialized")
        print(f"Detector Type: {'HAILO' if self.config.USE_HAILO else 'YOLO'}")
        print(f"Model: {self.config.YOLO_MODEL}")
        print(f"Device: {self.device}")
        print(f"ReID Enabled: {tracker_config['with_reid']}")
        
        # Print available YOLO classes
        self.print_available_classes()
    
    def _initialize_detector(self):
        """Initialize detector (HAILO or regular YOLO)"""
        if self.config.USE_HAILO and HAILO_AVAILABLE:
            try:
                # Check if HAILO model exists, download if needed
                if not os.path.exists(self.config.YOLO_MODEL):
                    print(f"HAILO model not found: {self.config.YOLO_MODEL}")
                    model_name = self.config.YOLO_MODEL.replace('.hef', '')
                    print(f"Downloading {model_name} from HAILO model zoo...")
                    self.config.YOLO_MODEL = download_hailo_yolo_model(model_name)
                
                # Initialize HAILO detector
                detector = HailoYOLODetector(self.config.YOLO_MODEL, self.config)
                print(f"HAILO detector initialized: {self.config.YOLO_MODEL}")
                return detector
                
            except Exception as e:
                print(f"Failed to initialize HAILO detector: {e}")
                print("Falling back to regular YOLO detector")
                self.config.USE_HAILO = False
        
        # Fallback to regular YOLO
        self.config.USE_HAILO = False
        
        # Check if we need to switch model type from HAILO to YOLO
        if isinstance(self.config.YOLO_MODEL, str) and self.config.YOLO_MODEL.endswith('.hef'):
            print(f"Warning: Current model {self.config.YOLO_MODEL} is a HAILO model, but HAILO is not available.")
            print("Switching to default YOLO model for fallback...")
            self.config.YOLO_MODEL = "yolo11n.pt"  # Default YOLO model
        
        print(f"[DEBUG] Initializing YOLO detector with model: {self.config.YOLO_MODEL}")
        print(f"[DEBUG] Model path type: {type(self.config.YOLO_MODEL)}")
        
        try:
            detector = YOLO(self.config.YOLO_MODEL)
            print(f"YOLO detector initialized: {self.config.YOLO_MODEL}")
            return detector
        except Exception as e:
            print(f"Failed to initialize YOLO detector with model {self.config.YOLO_MODEL}: {e}")
            # Try with a default model as last resort
            try:
                default_model = "yolo11n.pt"
                print(f"Trying with default model: {default_model}")
                detector = YOLO(default_model)
                self.config.YOLO_MODEL = default_model
                print(f"YOLO detector initialized with default model: {default_model}")
                return detector
            except Exception as e2:
                print(f"Failed to initialize YOLO detector with default model: {e2}")
                raise RuntimeError(f"Could not initialize any detector: HAILO failed, YOLO with {self.config.YOLO_MODEL} failed, YOLO with default failed")
    
    def _setup_device(self):
        """Setup computation device"""
        if self.config.USE_HAILO:
            return "hailo"  # HAILO device
        elif torch.cuda.is_available():
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
        
        print(f"[DEBUG] Initializing camera with source: {video_source}")
        print(f"[DEBUG] Video source type: {type(video_source)}")
        
        # Initialize camera using enhanced camera module (enforced)
        if isinstance(video_source, (str, int)) and (str(video_source).startswith('/dev/video') or isinstance(video_source, int)):
            # Always use enhanced camera
            print(f"[DEBUG] Creating enhanced camera...")
            try:
                self.camera = create_enhanced_camera(
                    width=self.config.CAM_WIDTH,
                    height=self.config.CAM_HEIGHT,
                    fps=self.config.CAM_FPS,
                    preferred_method=self.config.PREFERRED_CAMERA_METHOD
                )
                print(f"[DEBUG] Enhanced camera created successfully")
                if hasattr(self.camera, 'get_camera_info'):
                    camera_info = self.camera.get_camera_info()
                    print(f"[DEBUG] Camera info: {camera_info}")
            except Exception as e:
                print(f"[ERROR] Failed to create enhanced camera: {e}")
                import traceback
                traceback.print_exc()
                return
        else:
            # Regular OpenCV camera or video file
            print(f"[DEBUG] Creating OpenCV camera...")
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                print(f"[ERROR] Could not open video source {video_source}")
                return
            self.camera = cap
            print(f"[DEBUG] OpenCV camera created successfully")
        
        # Get video properties
        print(f"[DEBUG] Getting video properties...")
        try:
            if hasattr(self.camera, 'get_camera_info'):
                # Pi5 camera
                print(f"[DEBUG] Using Pi5 camera method...")
                camera_info = self.camera.get_camera_info()
                fps = camera_info.get('target_fps', 30)
                width, height = camera_info.get('resolution', self.config.DISPLAY_SIZE)
                print(f"[DEBUG] Pi5 camera info: {camera_info}")
            else:
                # OpenCV camera
                print(f"[DEBUG] Using OpenCV camera method...")
                fps = int(self.camera.get(cv2.CAP_PROP_FPS))
                width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[DEBUG] OpenCV camera properties - FPS: {fps}, Width: {width}, Height: {height}")
            
            print(f"[DEBUG] Final video properties: {width}x{height} @ {fps} FPS")
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
        # Get total frame count if available (may not be available for camera streams)
        if hasattr(self.camera, 'get'):
            total_frames = int(self.camera.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            total_frames = 0  # Unknown for camera streams
        
        print(f"[DEBUG] Starting frame processing loop...")
        print(f"[DEBUG] Total frames: {total_frames}")
        print(f"[DEBUG] Camera has 'read' method: {hasattr(self.camera, 'read')}")
        print(f"[DEBUG] Camera type: {type(self.camera)}")
        
        try:
            print("Starting frame processing loop...")
            while True:
                print(f"[DEBUG] Reading frame {frame_count}...")
                # Read frame from camera
                if hasattr(self.camera, 'read'):
                    # Pi5 camera or OpenCV camera
                    ret, frame = self.camera.read()
                    print(f"Frame read: ret={ret}, frame_shape={frame.shape if frame is not None else None}")
                else:
                    # Fallback
                    print("Camera doesn't have read method")
                    ret, frame = False, None
                
                if not ret or frame is None:
                    print(f"Failed to read frame: ret={ret}, frame is None={frame is None}")
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
                    
                    # Add HAILO performance stats if available
                    if self.config.USE_HAILO and hasattr(self.detector, 'get_performance_stats'):
                        hailo_stats = self.detector.get_performance_stats()
                        print(f"Frame {frame_count}/{total_frames} | "
                              f"FPS: {stats.get('fps', 0):.1f} | "
                              f"HAILO FPS: {hailo_stats.get('fps', 0):.1f} | "
                              f"Count: {counts['total']} | "
                              f"Active Tracks: {len(self.tracker.tracks)}")
                    else:
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
    
    def process_frame(self, frame, frame_count):
        """Process a single frame"""
        # Initialize variables
        tracks = []
        
        # DEBUG: Log frame processing info
        should_debug = frame_count % 10 == 0
        if should_debug:
            print(f"\n--- PROCESSING FRAME {frame_count} ---")
        
        # Always predict track positions for smooth tracking
        self.tracker._predict_tracks()
        
        # Process every frame for more stable tracking
        # Start detection timer
        detection_start = time.time()
        
        # Detection (HAILO or YOLO)
        print(f"[DEBUG] Detector type: {type(self.detector)}")
        print(f"[DEBUG] USE_HAILO: {self.config.USE_HAILO}")
        print(f"[DEBUG] Detector has 'detect' method: {hasattr(self.detector, 'detect')}")
        
        if self.config.USE_HAILO:
            # HAILO inference
            print(f"[DEBUG] Using HAILO inference...")
            detections, inference_time = self.detector.detect(
                frame,
                confidence_threshold=self.config.YOLO_CONFIDENCE,
                iou_threshold=self.config.YOLO_IOU_THRESHOLD,
                classes=self.config.YOLO_CLASSES
            )
            detection_time = inference_time
        else:
            # Regular YOLO inference
            print(f"[DEBUG] Using YOLO inference...")
            try:
                results = self.detector(frame, conf=self.config.YOLO_CONFIDENCE,
                                       iou=self.config.YOLO_IOU_THRESHOLD,
                                       classes=self.config.YOLO_CLASSES,
                                       verbose=False)
                
                detection_time = time.time() - detection_start
                # Extract detections
                detections = self._extract_detections(results[0])
            except Exception as e:
                print(f"[ERROR] YOLO inference failed: {e}")
                print(f"[ERROR] Detector type: {type(self.detector)}")
                print(f"[ERROR] Detector: {self.detector}")
                raise e
        
        self.performance_monitor.add_detection_time(detection_time)
        
        # DEBUG: Log detection info
        if should_debug:
            print(f"Detections found: {len(detections)}")
            if len(detections) == 0:
                print("NO DETECTIONS - Checking for potential ghost tracks...")
        
        # Start tracking timer
        tracking_start = time.time()
        
        # BoTSORT tracking
        tracks_before = len(self.tracker.tracks)
        tracks = self.tracker.update(detections, frame)
        tracks_after = len(tracks)
        
        tracking_time = time.time() - tracking_start
        self.performance_monitor.add_tracking_time(tracking_time)
        
        # DEBUG: Log track changes
        if should_debug:
            print(f"Tracks before: {tracks_before}, Tracks after: {tracks_after}")
            if len(detections) == 0 and tracks_after > 0:
                print(f"WARNING: {tracks_after} tracks active with NO detections (GHOSTING!)")
                for track in tracks:
                    print(f"  Ghost track {track.track_id}: time_since_update={track.time_since_update}, bbox={track.to_tlbr()}")
        
        # Update bus counter
        height = frame.shape[0]
        self.bus_counter.update(tracks, height)
        
        # Resize frame for display
        display_frame = resize_frame(frame, self.config.DISPLAY_SIZE)
        
        # Draw tracking results (even on skipped frames)
        annotated_frame = self.visualizer.draw_tracking_results(
            display_frame, tracks, original_frame_size=frame.shape[:2], show_trails=True, show_ids=True
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