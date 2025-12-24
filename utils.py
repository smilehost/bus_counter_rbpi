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
    
    def draw_counting_line(self, frame, line_y=None):
        """Draw counting line on frame
        
        Args:
            frame: Input frame
            line_y: Optional y-coordinate for horizontal line (backward compatibility)
        """
        height, width = frame.shape[:2]
        
        if self.config.COUNTING_LINE_ROTATED:
            # Draw rotated counting line with directional arrows
            endpoints = self.config.COUNTING_LINE_ENDPOINTS
            
            # Convert normalized endpoints to pixel coordinates
            p1 = (int(endpoints['x1'] * width), int(endpoints['y1'] * height))
            p2 = (int(endpoints['x2'] * width), int(endpoints['y2'] * height))
            
            # Draw the main line
            cv2.line(frame, p1, p2, self.config.COLORS['trail'], 3)
            
            # Calculate line direction vector
            line_dir = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            line_length = np.linalg.norm(line_dir)
            
            if line_length > 0:
                # Calculate normal vector (perpendicular to line)
                normal = np.array([-line_dir[1], line_dir[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Draw "IN" direction arrow (opposite to normal)
                in_start = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
                in_end = in_start - normal * 30  # 30 pixels arrow length
                self._draw_arrow(frame, in_start, in_end, (0, 255, 0), "IN")
                
                # Draw "OUT" direction arrow (along normal)
                out_end = in_start + normal * 30
                self._draw_arrow(frame, in_start, out_end, (0, 0, 255), "OUT")
            
            # Add label
            label_pos = (p1[0] + 10, p1[1] - 10)
            cv2.putText(frame, "Counting Line", label_pos,
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.COLORS['trail'], 2)
        else:
            # Original horizontal line (backward compatibility)
            if line_y is None:
                line_y = height // 2
            
            cv2.line(frame, (0, line_y), (width, line_y),
                    self.config.COLORS['trail'], 2)
            
            # Add label
            cv2.putText(frame, "Counting Line", (10, line_y - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.config.COLORS['trail'], 2)
    
    def _draw_arrow(self, frame, start, end, color, label):
        """Draw an arrow with label
        
        Args:
            frame: Input frame
            start: Start point (x, y)
            end: End point (x, y)
            color: Arrow color in BGR
            label: Text label to display
        """
        # Convert to numpy arrays for vector operations
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        
        # Draw arrow shaft
        cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), color, 3)
        
        # Calculate arrow head
        arrow_length = 15
        arrow_angle = np.pi / 6  # 30 degrees
        
        # Direction vector
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction = direction / length
            
            # Arrow head points
            left_wing = end - direction * arrow_length * np.cos(arrow_angle) + \
                       np.array([-direction[1], direction[0]]) * arrow_length * np.sin(arrow_angle)
            right_wing = end - direction * arrow_length * np.cos(arrow_angle) + \
                        np.array([direction[1], -direction[0]]) * arrow_length * np.sin(arrow_angle)
            
            # Draw arrow head
            cv2.line(frame, tuple(end.astype(int)), tuple(left_wing.astype(int)), color, 3)
            cv2.line(frame, tuple(end.astype(int)), tuple(right_wing.astype(int)), color, 3)
        
        # Add label
        label_pos = (int(end[0]) + 5, int(end[1]) - 5)
        cv2.putText(frame, label, label_pos,
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def draw_detection_zone(self, frame):
        """Draw detection zone on frame with semi-transparent overlay"""
        if not self.config.DETECTION_ZONE_ENABLED:
            return frame
        
        height, width = frame.shape[:2]
        zone = self.config.DETECTION_ZONE
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(zone['x1'] * width)
        y1 = int(zone['y1'] * height)
        x2 = int(zone['x2'] * width)
        y2 = int(zone['y2'] * height)
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        
        # Draw filled rectangle with semi-transparency
        # Use a distinct color (e.g., yellow/orange) for the zone
        zone_color = (0, 165, 255)  # Orange in BGR
        cv2.rectangle(overlay, (x1, y1), (x2, y2), zone_color, -1)
        
        # Blend the overlay with the original frame (30% opacity)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw border with thicker line
        border_color = (0, 200, 255)  # Brighter orange for border
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 3)
        
        # Add label
        label = "Detection Zone"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, 2)[0]
        
        # Background for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     border_color, -1)
        
        # Text
        cv2.putText(frame, label, (x1, y1 - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                  (255, 255, 255), 2)
        
        return frame
    
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
        self.track_positions = defaultdict(list)  # For horizontal line: stores y positions
        self.track_positions_xy = defaultdict(list)  # For rotated line: stores (x, y) tuples
        self.frame_count = 0  # For debug logging
        
    def update(self, tracks, frame_height, frame_width):
        """Update counter with new tracking data"""
        self.frame_count += 1
        
        # DEBUG: Log counting configuration and state every 30 frames
        if self.frame_count % 30 == 0:
            print(f"\n[DEBUG COUNTER] Frame {self.frame_count}")
            print(f"[DEBUG COUNTER] COUNTING_LINE_ROTATED: {self.config.COUNTING_LINE_ROTATED}")
            print(f"[DEBUG COUNTER] COUNTING_LINE_Y: {self.config.COUNTING_LINE_Y}")
            print(f"[DEBUG COUNTER] COUNTING_DIRECTION: {self.config.COUNTING_DIRECTION}")
            print(f"[DEBUG COUNTER] MIN_TRACK_AGE: {self.config.MIN_TRACK_AGE}")
            print(f"[DEBUG COUNTER] DETECTION_ZONE_ENABLED: {self.config.DETECTION_ZONE_ENABLED}")
            if self.config.DETECTION_ZONE_ENABLED:
                zone = self.config.DETECTION_ZONE
                print(f"[DEBUG COUNTER] Detection Zone: x1={zone['x1']}, y1={zone['y1']}, x2={zone['x2']}, y2={zone['y2']}")
            print(f"[DEBUG COUNTER] Tracks received: {len(tracks)}")
            print(f"[DEBUG COUNTER] Counted tracks so far: {len(self.counted_tracks)}")
            print(f"[DEBUG COUNTER] Current counts: IN={self.count_in}, OUT={self.count_out}")
        
        # Handle rotated counting line mode
        if self.config.COUNTING_LINE_ROTATED:
            # Convert normalized endpoints to pixel coordinates
            endpoints = self.config.COUNTING_LINE_ENDPOINTS
            line_p1 = (int(endpoints['x1'] * frame_width), int(endpoints['y1'] * frame_height))
            line_p2 = (int(endpoints['x2'] * frame_width), int(endpoints['y2'] * frame_height))
            
            if self.frame_count % 30 == 0:
                print(f"[DEBUG COUNTER] Rotated line mode - Line: {line_p1} to {line_p2}")
            
            for track in tracks:
                track_id = track.track_id
                
                # Skip counting if detection zone is enabled and track is outside zone
                if self.config.DETECTION_ZONE_ENABLED:
                    in_zone = self._is_in_zone(track.bbox, frame_width, frame_height)
                    if self.frame_count % 30 == 0:
                        print(f"[DEBUG COUNTER] Track {track_id}: in_zone={in_zone}")
                    if not in_zone:
                        continue
                
                # Get center position
                bbox = track.bbox
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Store position history
                self.track_positions_xy[track_id].append((center_x, center_y))
                
                # Check if track should be counted
                if (track_id not in self.counted_tracks and
                    len(self.track_positions_xy[track_id]) >= self.config.MIN_TRACK_AGE):
                    
                    positions = self.track_positions_xy[track_id]
                    
                    if self.frame_count % 30 == 0:
                        print(f"[DEBUG COUNTER] Track {track_id}: age={len(positions)}, min_age={self.config.MIN_TRACK_AGE}")
                        print(f"[DEBUG COUNTER]   Last positions: {positions[-3:]}")  # Last 3 positions
                        print(f"[DEBUG COUNTER]   Current pos: ({center_x:.1f}, {center_y:.1f})")
                    
                    # Check if track path crossed the counting line
                    if self._line_segments_intersect(positions[-2], positions[-1], line_p1, line_p2):
                        # Determine crossing direction
                        direction = self._get_crossing_direction(positions[-2], positions[-1], line_p1, line_p2)
                        
                        print(f"[DEBUG COUNTER] *** COUNTING Track {track_id} as {direction.upper()} (crossed rotated line) ***")
                        
                        # Update counts based on direction
                        if self.config.COUNTING_DIRECTION == "both":
                            if direction == "in":
                                self.count_in += 1
                            elif direction == "out":
                                self.count_out += 1
                        elif self.config.COUNTING_DIRECTION == "in" and direction == "in":
                            self.count_in += 1
                        elif self.config.COUNTING_DIRECTION == "out" and direction == "out":
                            self.count_out += 1
                        
                        self.counted_tracks.add(track_id)
                    elif self.frame_count % 30 == 0:
                        print(f"[DEBUG COUNTER] Track {track_id}: did NOT cross rotated line")
                elif self.frame_count % 30 == 0:
                    if track_id in self.counted_tracks:
                        print(f"[DEBUG COUNTER] Track {track_id}: already counted")
                    else:
                        print(f"[DEBUG COUNTER] Track {track_id}: age={len(self.track_positions_xy[track_id])}, too young (min_age={self.config.MIN_TRACK_AGE})")
        else:
            # Original horizontal line mode (backward compatibility)
            if self.config.COUNTING_LINE_Y is None:
                self.config.COUNTING_LINE_Y = frame_height // 2
            
            line_y = self.config.COUNTING_LINE_Y
            
            if self.frame_count % 30 == 0:
                print(f"[DEBUG COUNTER] Horizontal line mode - Line Y: {line_y} (frame height: {frame_height})")
            
            for track in tracks:
                track_id = track.track_id
                
                # Skip counting if detection zone is enabled and track is outside zone
                if self.config.DETECTION_ZONE_ENABLED:
                    in_zone = self._is_in_zone(track.bbox, frame_width, frame_height)
                    if self.frame_count % 30 == 0:
                        print(f"[DEBUG COUNTER] Track {track_id}: in_zone={in_zone}")
                    if not in_zone:
                        continue
                
                # Get center position
                bbox = track.bbox
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Store position history
                self.track_positions[track_id].append(center_y)
                
                # Check if track should be counted
                if (track_id not in self.counted_tracks and
                    len(self.track_positions[track_id]) >= self.config.MIN_TRACK_AGE):
                    
                    positions = self.track_positions[track_id]
                    
                    if self.frame_count % 30 == 0:
                        print(f"[DEBUG COUNTER] Track {track_id}: age={len(positions)}, min_age={self.config.MIN_TRACK_AGE}")
                        print(f"[DEBUG COUNTER]   Last positions: {positions[-5:]}")  # Last 5 positions
                        print(f"[DEBUG COUNTER]   Line Y: {line_y}, Current Y: {center_y}")
                    
                    # Check direction
                    if self._crossed_line(positions, line_y):
                        if self._is_moving_down(positions):
                            print(f"[DEBUG COUNTER] *** COUNTING Track {track_id} as OUT (moving down) ***")
                            self.count_out += 1
                        elif self._is_moving_up(positions):
                            print(f"[DEBUG COUNTER] *** COUNTING Track {track_id} as IN (moving up) ***")
                            self.count_in += 1
                        
                        self.counted_tracks.add(track_id)
                    elif self.frame_count % 30 == 0:
                        print(f"[DEBUG COUNTER] Track {track_id}: did NOT cross line")
                elif self.frame_count % 30 == 0:
                    if track_id in self.counted_tracks:
                        print(f"[DEBUG COUNTER] Track {track_id}: already counted")
                    else:
                        print(f"[DEBUG COUNTER] Track {track_id}: age={len(self.track_positions[track_id])}, too young (min_age={self.config.MIN_TRACK_AGE})")
    
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
    
    def _line_segments_intersect(self, p1, p2, p3, p4):
        """Check if two line segments intersect using parametric equations
        
        Args:
            p1, p2: Endpoints of first line segment (track movement)
            p3, p4: Endpoints of second line segment (counting line)
            
        Returns:
            True if the line segments intersect
        """
        # Convert to numpy arrays for vector operations
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        p4 = np.array(p4)
        
        # Direction vectors
        d1 = p2 - p1  # Track movement direction
        d2 = p4 - p3  # Counting line direction
        
        # Cross product of direction vectors
        cross_d1_d2 = np.cross(d1, d2)
        
        # Check if lines are parallel
        if abs(cross_d1_d2) < 1e-10:
            return False  # Parallel lines don't intersect (or are collinear)
        
        # Vector from p3 to p1
        d3 = p1 - p3
        
        # Calculate parameters for intersection point
        t = np.cross(d3, d2) / cross_d1_d2
        u = np.cross(d3, d1) / cross_d1_d2
        
        # Check if intersection point lies within both line segments
        # t and u should be in [0, 1] for the segments to intersect
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def _get_crossing_direction(self, prev_pos, curr_pos, line_p1, line_p2):
        """Determine crossing direction using normal vector calculation
        
        Args:
            prev_pos: Previous position (x, y) tuple
            curr_pos: Current position (x, y) tuple
            line_p1, line_p2: Endpoints of the counting line
            
        Returns:
            "in" or "out" based on crossing direction relative to line normal
        """
        # Convert to numpy arrays
        prev_pos = np.array(prev_pos)
        curr_pos = np.array(curr_pos)
        line_p1 = np.array(line_p1)
        line_p2 = np.array(line_p2)
        
        # Calculate the direction vector of the counting line
        line_dir = line_p2 - line_p1
        
        # Calculate the normal vector (perpendicular to the line)
        # For a line from p1 to p2, the normal is (-dy, dx)
        normal = np.array([-line_dir[1], line_dir[0]])
        
        # Normalize the normal vector
        normal_length = np.linalg.norm(normal)
        if normal_length > 0:
            normal = normal / normal_length
        
        # Calculate the movement vector of the track
        movement = curr_pos - prev_pos
        
        # Calculate the dot product of movement with normal
        # Positive dot product means movement in direction of normal ("out")
        # Negative dot product means movement opposite to normal ("in")
        dot_product = np.dot(movement, normal)
        
        if dot_product > 0:
            return "out"
        else:
            return "in"
    
    def _is_in_zone(self, bbox, frame_width, frame_height):
        """Check if a track's bounding box is within the detection zone
        
        Args:
            bbox: Bounding box in pixel coordinates [x1, y1, x2, y2]
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            True if the track is considered to be in the detection zone
        """
        zone = self.config.DETECTION_ZONE
        
        # Convert normalized zone coordinates to pixel coordinates
        zone_x1 = zone['x1'] * frame_width
        zone_y1 = zone['y1'] * frame_height
        zone_x2 = zone['x2'] * frame_width
        zone_y2 = zone['y2'] * frame_height
        
        # Get track bounding box coordinates
        track_x1, track_y1, track_x2, track_y2 = bbox
        
        # Calculate center point of track
        track_center_x = (track_x1 + track_x2) / 2
        track_center_y = (track_y1 + track_y2) / 2
        
        # Check if center is within zone
        if (zone_x1 <= track_center_x <= zone_x2 and
            zone_y1 <= track_center_y <= zone_y2):
            return True
        
        # Calculate overlap area for partial detection
        overlap_x1 = max(track_x1, zone_x1)
        overlap_y1 = max(track_y1, zone_y1)
        overlap_x2 = min(track_x2, zone_x2)
        overlap_y2 = min(track_y2, zone_y2)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return False  # No overlap
        
        # Calculate areas
        overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        track_area = (track_x2 - track_x1) * (track_y2 - track_y1)
        
        # Check if overlap exceeds margin threshold
        if track_area > 0:
            overlap_ratio = overlap_area / track_area
            return overlap_ratio >= self.config.DETECTION_ZONE_MARGIN
        
        return False
    
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
        self.track_positions_xy.clear()


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