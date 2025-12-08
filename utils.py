import cv2
import numpy as np
from collections import defaultdict, deque
import random
from class_names import get_class_name

class Visualizer:
    """Visualization utilities for object tracking"""
    
    def __init__(self, config):
        self.config = config
        self.colors = self._generate_colors(100)
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
    def _generate_colors(self, num_colors):
        """Generate random colors for tracking visualization"""
        colors = []
        for i in range(num_colors):
            hue = i * 180 // num_colors
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def draw_tracking_results(self, frame, tracks, original_frame_size=None, show_trails=True, show_ids=True):
        """Draw tracking results on frame"""
        annotated_frame = frame.copy()
        
        # Only log drawing info occasionally to reduce spam
        # This is called every frame, so we use a simple counter
        if not hasattr(self, 'draw_counter'):
            self.draw_counter = 0
        self.draw_counter += 1
        
        if self.draw_counter % 10 == 0:  # Log every 10 frames
            print(f"Drawing {len(tracks)} tracks")
        
        # Calculate scale factor if original frame size is provided
        scale_x = scale_y = 1.0
        if original_frame_size:
            scale_x = frame.shape[1] / original_frame_size[1]
            scale_y = frame.shape[0] / original_frame_size[0]
        
        for track in tracks:
            track_id = track.track_id
            bbox = track.bbox
            color = self.colors[track_id % len(self.colors)]
            
            if self.draw_counter % 10 == 0:  # Log every 10 frames
                print(f"  Drawing track {track_id} (state: {track.state}, hits: {track.hits}, age: {track.age})")
            
            # Scale bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color,
                         self.config.LINE_THICKNESS)
            
            # Draw track ID, class name and confidence
            if show_ids:
                # Get class name if available
                class_name = ""
                if hasattr(track, 'class_id'):
                    class_name = get_class_name(track.class_id)
                
                label = f"ID: {track_id}"
                if class_name:
                    label += f" {class_name}"
                label += f" Conf: {track.score:.2f}"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                            self.config.FONT_SIZE,
                                            self.config.FONT_THICKNESS)[0]
                
                # Background for text
                cv2.rectangle(annotated_frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Text
                cv2.putText(annotated_frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, self.config.FONT_SIZE,
                          self.config.COLORS['text'], self.config.FONT_THICKNESS)
            
            # Draw center point (using scaled coordinates)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(annotated_frame, (center_x, center_y), 3,
                      self.config.COLORS['center'], -1)
            
            # Update and draw trail (using scaled coordinates)
            if show_trails:
                self.track_history[track_id].append((center_x, center_y))
                self._draw_trail(annotated_frame, track_id, color)
        
        return annotated_frame
    
    def _draw_trail(self, frame, track_id, color):
        """Draw movement trail for a track"""
        points = list(self.track_history[track_id])
        if len(points) > 1:
            for i in range(1, len(points)):
                # Fade trail based on age
                alpha = i / len(points)
                thickness = max(1, int(self.config.LINE_THICKNESS * alpha))
                
                cv2.line(frame, points[i-1], points[i], color, thickness)
    
    def draw_counting_line(self, frame, line_y):
        """Draw counting line on frame"""
        height, width = frame.shape[:2]
        cv2.line(frame, (0, line_y), (width, line_y),
                self.config.COLORS['trail'], 2)
        
        # Add label
        cv2.putText(frame, "Counting Line", (10, line_y - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.COLORS['trail'], 2)
    
    def draw_statistics(self, frame, stats):
        """Draw tracking statistics on frame"""
        y_offset = 30
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (10, y_offset),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                      self.config.COLORS['text'], 2)
            y_offset += 30


class BusCounter:
    """Bus counting utility using tracking data"""
    
    def __init__(self, config):
        self.config = config
        self.count_in = 0
        self.count_out = 0
        self.counted_tracks = set()
        self.track_positions = defaultdict(list)
        
    def update(self, tracks, frame_height):
        """Update counter with new tracking data"""
        if self.config.COUNTING_LINE_Y is None:
            self.config.COUNTING_LINE_Y = frame_height // 2
        
        line_y = self.config.COUNTING_LINE_Y
        
        for track in tracks:
            track_id = track.track_id
            
            # Get center position
            bbox = track.bbox
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Store position history
            self.track_positions[track_id].append(center_y)
            
            # Check if track should be counted
            if (track_id not in self.counted_tracks and
                len(self.track_positions[track_id]) >= self.config.MIN_TRACK_AGE):
                
                positions = self.track_positions[track_id]
                
                # Check direction
                if self._crossed_line(positions, line_y):
                    if self._is_moving_down(positions):
                        self.count_out += 1
                    elif self._is_moving_up(positions):
                        self.count_in += 1
                    
                    self.counted_tracks.add(track_id)
    
    def _crossed_line(self, positions, line_y):
        """Check if track crossed the counting line"""
        if len(positions) < 2:
            return False
        
        prev_pos = positions[-2]
        curr_pos = positions[-1]
        
        return (prev_pos < line_y and curr_pos >= line_y) or \
               (prev_pos > line_y and curr_pos <= line_y)
    
    def _is_moving_down(self, positions):
        """Check if track is moving downward"""
        if len(positions) < 2:
            return False
        
        return positions[-1] > positions[-2]
    
    def _is_moving_up(self, positions):
        """Check if track is moving upward"""
        if len(positions) < 2:
            return False
        
        return positions[-1] < positions[-2]
    
    def get_counts(self):
        """Get current counts"""
        return {
            'count_in': self.count_in,
            'count_out': self.count_out,
            'total': self.count_in + self.count_out
        }
    
    def reset(self):
        """Reset counter"""
        self.count_in = 0
        self.count_out = 0
        self.counted_tracks.clear()
        self.track_positions.clear()


class PerformanceMonitor:
    """Monitor tracking performance"""
    
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.tracking_times = deque(maxlen=30)
        
    def start_frame_timer(self):
        """Start frame processing timer"""
        self.frame_start_time = cv2.getTickCount()
    
    def end_frame_timer(self):
        """End frame processing timer"""
        if hasattr(self, 'frame_start_time'):
            frame_time = (cv2.getTickCount() - self.frame_start_time) / cv2.getTickFrequency()
            self.frame_times.append(frame_time)
    
    def add_detection_time(self, time):
        """Add detection processing time"""
        self.detection_times.append(time)
    
    def add_tracking_time(self, time):
        """Add tracking processing time"""
        self.tracking_times.append(time)
    
    def get_fps(self):
        """Get current FPS"""
        if len(self.frame_times) > 0:
            avg_time = np.mean(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0
    
    def get_statistics(self):
        """Get performance statistics"""
        stats = {}
        
        if len(self.frame_times) > 0:
            stats['fps'] = self.get_fps()
            stats['avg_frame_time'] = np.mean(self.frame_times) * 1000  # ms
        
        if len(self.detection_times) > 0:
            stats['avg_detection_time'] = np.mean(self.detection_times) * 1000  # ms
        
        if len(self.tracking_times) > 0:
            stats['avg_tracking_time'] = np.mean(self.tracking_times) * 1000  # ms
        
        return stats


def resize_frame(frame, target_size):
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale = min(target_width / width, target_height / height)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate position to center the resized frame
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    
    # Place resized frame on canvas
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
    
    return canvas


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """Apply non-maximum suppression to detections"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by scores
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the best detection
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining detections
        remaining = indices[1:]
        ious = np.array([calculate_iou(boxes[current], boxes[i]) for i in remaining])
        
        # Keep detections with IoU below threshold
        indices = remaining[ious < iou_threshold]
    
    return keep