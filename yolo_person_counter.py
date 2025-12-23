import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO
import os
import sys
from collections import defaultdict, deque
import statistics

from config import Config
from class_names import get_class_name, COCO_CLASS_NAMES
from utils import resize_frame


class YOLOPersonCounter:
    """
    YOLO-based person detection and counting class
    Processes each frame independently and provides statistics
    """
    
    def __init__(self, model_path=None, confidence=0.5, device=None):
        """
        Initialize the YOLO Person Counter
        
        Args:
            model_path (str): Path to YOLO model file
            confidence (float): Detection confidence threshold
            device (str): Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        # Load configuration
        self.config = Config()
        
        # Override with provided parameters
        if model_path:
            self.config.YOLO_MODEL = model_path
        if confidence is not None:
            self.config.YOLO_CONFIDENCE = confidence
        
        # Setup device
        if device:
            self.device = device
        else:
            self.device = self.config.DEVICE
            
        # Initialize YOLO model
        self.detector = self._initialize_detector()
        
        # Statistics tracking
        self.frame_counts = []  # List of person counts per frame
        self.frame_data = []    # Detailed data for each frame
        self.total_frames = 0
        self.processing_times = []
        
        # Person class ID (COCO dataset)
        self.person_class_id = 0
        
        print(f"YOLO Person Counter initialized")
        print(f"Model: {self.config.YOLO_MODEL}")
        print(f"Device: {self.device}")
        print(f"Confidence threshold: {self.config.YOLO_CONFIDENCE}")
    
    def _initialize_detector(self):
        """Initialize YOLO detector"""
        try:
            detector = YOLO(self.config.YOLO_MODEL)
            
            # Move model to the configured device
            if self.device == "cuda" and torch.cuda.is_available():
                detector.to('cuda')
                print(f"YOLO detector moved to CUDA")
            elif self.device == "cpu":
                detector.to('cpu')
                print(f"YOLO detector moved to CPU")
            
            return detector
        except Exception as e:
            print(f"Failed to initialize YOLO detector: {e}")
            # Try with a default model as last resort
            try:
                default_model = "yolo11n.pt"
                print(f"Trying with default model: {default_model}")
                detector = YOLO(default_model)
                if self.device == "cuda" and torch.cuda.is_available():
                    detector.to('cuda')
                elif self.device == "cpu":
                    detector.to('cpu')
                
                self.config.YOLO_MODEL = default_model
                print(f"YOLO detector initialized with default model: {default_model}")
                return detector
            except Exception as e2:
                print(f"Failed to initialize YOLO detector with default model: {e2}")
                raise RuntimeError(f"Could not initialize YOLO detector")
    
    def process_video(self, video_path, output_path=None, show_preview=True):
        """
        Process video file and count people in each frame
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path to save output video (optional)
            show_preview (bool): Whether to show live preview
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        
        # Setup video writer if needed
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        # Reset statistics
        self.frame_counts = []
        self.frame_data = []
        self.total_frames = 0
        self.processing_times = []
        
        # Process frames
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                start_time = time.time()
                processed_frame, person_count, detections = self.process_frame(frame, frame_count)
                processing_time = time.time() - start_time
                
                # Store statistics
                self.frame_counts.append(person_count)
                self.frame_data.append({
                    'frame_number': frame_count,
                    'person_count': person_count,
                    'detections': detections,
                    'processing_time': processing_time
                })
                self.processing_times.append(processing_time)
                self.total_frames += 1
                
                # Write frame to output video
                if video_writer:
                    video_writer.write(processed_frame)
                
                # Display frame
                if show_preview:
                    cv2.imshow('YOLO Person Counter', processed_frame)
                    
                    # Check for exit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print progress
                if frame_count % 30 == 0:
                    self._print_progress(frame_count, total_frames)
                
                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            self._print_final_statistics()
    
    def process_frame(self, frame, frame_number=None):
        """
        Process a single frame and detect people
        
        Args:
            frame: Input frame
            frame_number (int): Frame number for tracking
            
        Returns:
            tuple: (processed_frame, person_count, detections)
        """
        # Make a copy for drawing
        processed_frame = frame.copy()
        
        # Run YOLO detection
        confidence = self.config.YOLO_CONFIDENCE if self.config.YOLO_CONFIDENCE is not None else 0.5
        results = self.detector.predict(
            frame,
            conf=confidence,
            classes=[self.person_class_id],  # Only detect people
            verbose=False
        )
        
        # Extract person detections
        person_detections = []
        person_count = 0
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                confidence = scores[i]
                
                person_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
                
                # Draw bounding box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Person {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(processed_frame, (x1, y1 - label_size[1] - 10), 
                              (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(processed_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                person_count += 1
        
        # Draw statistics on frame
        self._draw_statistics(processed_frame, person_count, frame_number)
        
        return processed_frame, person_count, person_detections
    
    def _draw_statistics(self, frame, person_count, frame_number=None):
        """Draw statistics on the frame"""
        # Calculate statistics
        if len(self.frame_counts) > 0:
            avg_count = np.mean(self.frame_counts)
            max_count = max(self.frame_counts)
            min_count = min(self.frame_counts)
            mode_count = self.get_mode_value()
        else:
            avg_count = max_count = min_count = mode_count = 0
        
        # Prepare text
        texts = [
            f"Current Count: {person_count}",
            f"Average: {avg_count:.1f}",
            f"Min: {min_count}",
            f"Max: {max_count}",
            f"Mode: {mode_count}",
            f"Frames: {len(self.frame_counts)}"
        ]
        
        if frame_number is not None:
            texts.append(f"Frame: {frame_number}")
        
        # Draw background rectangle
        text_height = 25
        bg_height = len(texts) * text_height + 10
        cv2.rectangle(frame, (10, 10), (300, bg_height), (0, 0, 0), -1)
        
        # Draw text
        for i, text in enumerate(texts):
            y_pos = 30 + i * text_height
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0, 255, 0), 2)
    
    def _print_progress(self, frame_count, total_frames):
        """Print processing progress"""
        if len(self.frame_counts) > 0:
            current_count = self.frame_counts[-1]
            avg_count = np.mean(self.frame_counts)
            mode_count = self.get_mode_value()
            fps = 1.0 / np.mean(self.processing_times[-10:]) if len(self.processing_times) >= 10 else 0
            
            print(f"Frame {frame_count}/{total_frames} | "
                  f"Current: {current_count} | "
                  f"Average: {avg_count:.1f} | "
                  f"Mode: {mode_count} | "
                  f"FPS: {fps:.1f}")
    
    def _print_final_statistics(self):
        """Print final statistics"""
        if len(self.frame_counts) == 0:
            print("No frames processed")
            return
        
        print("\n" + "="*50)
        print("FINAL PERSON COUNTING STATISTICS")
        print("="*50)
        
        # Basic statistics
        total_count = sum(self.frame_counts)
        avg_count = np.mean(self.frame_counts)
        median_count = np.median(self.frame_counts)
        mode_count = self.get_mode_value()
        max_count = max(self.frame_counts)
        min_count = min(self.frame_counts)
        std_dev = np.std(self.frame_counts)
        
        print(f"Total frames processed: {self.total_frames}")
        print(f"Total people detected (sum): {total_count}")
        print(f"Average people per frame: {avg_count:.2f}")
        print(f"Median people per frame: {median_count:.2f}")
        print(f"Mode people per frame: {mode_count}")
        print(f"Max people in a frame: {max_count}")
        print(f"Min people in a frame: {min_count}")
        print(f"Standard deviation: {std_dev:.2f}")
        
        # Performance statistics
        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time
            print(f"\nPerformance:")
            print(f"Average processing time: {avg_time*1000:.2f} ms")
            print(f"Average FPS: {fps:.2f}")
        
        # Frame distribution
        print(f"\nFrame Distribution:")
        count_distribution = defaultdict(int)
        for count in self.frame_counts:
            count_distribution[count] += 1
        
        for count in sorted(count_distribution.keys()):
            percentage = (count_distribution[count] / len(self.frame_counts)) * 100
            print(f"  {count} people: {count_distribution[count]} frames ({percentage:.1f}%)")
        
        print("="*50)
    
    def get_mode_value(self):
        """Calculate the mode (most frequent value) of person counts"""
        if len(self.frame_counts) == 0:
            return 0
        
        try:
            # Use statistics.mode for most common value
            return statistics.mode(self.frame_counts)
        except statistics.StatisticsError:
            # If there's no clear mode (all values appear once), return the average
            return int(np.mean(self.frame_counts))
    
    def get_frame_data(self, frame_range=None):
        """
        Get detailed frame data
        
        Args:
            frame_range (tuple): Optional (start, end) frame range
            
        Returns:
            list: Frame data for specified range or all frames
        """
        if frame_range is None:
            return self.frame_data
        
        start, end = frame_range
        return self.frame_data[start:end]
    
    def get_counts_summary(self):
        """
        Get a summary of person counts
        
        Returns:
            dict: Summary statistics
        """
        if len(self.frame_counts) == 0:
            return {}
        
        return {
            'total_frames': self.total_frames,
            'total_people': sum(self.frame_counts),
            'average': np.mean(self.frame_counts),
            'median': np.median(self.frame_counts),
            'mode': self.get_mode_value(),
            'max': max(self.frame_counts),
            'min': min(self.frame_counts),
            'std_dev': np.std(self.frame_counts)
        }