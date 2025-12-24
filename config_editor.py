"""
ConfigEditor - Interactive GUI editor for detection zone and counting line configuration.

This module provides an interactive OpenCV-based GUI for editing the detection zone
and counting line configuration for the bus counter system.
"""

import cv2
import numpy as np
from enum import Enum
from config import Config


class EditorMode(Enum):
    """Editor mode enumeration"""
    VIEW = "view"
    ZONE = "zone"
    LINE = "line"


class ConfigEditor:
    """Interactive GUI editor for detection zone and counting line configuration"""
    
    def __init__(self, camera_source=None, config=None):
        """
        Initialize the ConfigEditor.
        
        Args:
            camera_source: Video file path or camera index (default: from config)
            config: Config instance (default: creates new Config)
        """
        # Initialize configuration
        self.config = config if config is not None else Config()
        self.config.load_zone_config()
        
        # Camera setup
        self.camera_source = camera_source if camera_source is not None else self.config.VIDEO_SOURCE
        self.cap = None
        # Use default frame size from config (width, height)
        self.frame_width, self.frame_height = self.config.CONFIG_EDITOR_FRAME_SIZE
        self.original_frame_width = None
        self.original_frame_height = None
        self.use_blank_frame = False  # Flag to use blank frame when no video source
        
        # Editor state
        self.mode = EditorMode.VIEW
        self.editor_active = False
        self.running = True
        
        # Mouse interaction state
        self.drawing = False
        self.mouse_start = None
        self.mouse_current = None
        self.line_points = []  # Store points for line drawing
        self.selected_handle = None  # For editing existing elements
        
        # Temporary storage for pixel coordinates
        self.temp_zone = None
        self.temp_line = None
        
        # Colors (BGR)
        self.colors = {
            'zone_border': (0, 200, 255),      # Bright orange
            'zone_fill': (0, 165, 255),         # Orange
            'line': (255, 0, 0),                # Blue
            'handle': (0, 255, 0),              # Green
            'handle_selected': (0, 255, 255),   # Yellow
            'text': (255, 255, 255),            # White
            'background': (0, 0, 0)             # Black
        }
        
        # Handle size for editing
        self.handle_size = 10
        
        # Window name
        self.window_name = "Config Editor"
        
    def initialize_camera(self):
        """Initialize camera or video source"""
        try:
            if isinstance(self.camera_source, str):
                # Video file
                self.cap = cv2.VideoCapture(self.camera_source)
            else:
                # Camera index
                self.cap = cv2.VideoCapture(int(self.camera_source))
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera/video source: {self.camera_source}")
            
            # Get original frame dimensions
            ret, frame = self.cap.read()
            if ret:
                self.original_frame_height, self.original_frame_width = frame.shape[:2]
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
                print(f"Camera initialized: {self.original_frame_width}x{self.original_frame_height}")
                print(f"Config editor will use fixed frame size: {self.frame_width}x{self.frame_height}")
            else:
                raise RuntimeError("Failed to read frame from camera/video source")
            
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            # Fall back to blank frame mode
            self.use_blank_frame = True
            print("Using blank frame mode with default size: {}x{}".format(self.frame_width, self.frame_height))
            return True
    
    def start(self):
        """Start the editor main loop"""
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        # If using blank frame mode, create a blank frame
        if self.use_blank_frame:
            self.blank_frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            # Add some visual content to the blank frame
            cv2.putText(self.blank_frame, "No Video Source", (self.frame_width//2 - 100, self.frame_height//2 - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(self.blank_frame, "Use mouse to draw detection zone/counting line",
                       (self.frame_width//2 - 200, self.frame_height//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\n" + "=" * 50)
        print("CONFIG EDITOR STARTED")
        print("=" * 50)
        print("Press 'e' to enter/exit editor mode")
        print("Press 'z' to switch to zone editing mode")
        print("Press 'l' to switch to line editing mode")
        print("Press 's' to save configuration")
        print("Press 'r' to reset to defaults")
        print("Press 'd' to delete current zone/line")
        print("Press 'q' to quit")
        print("=" * 50 + "\n")
        
        while self.running:
            # Get frame (either from camera or blank frame)
            if self.use_blank_frame:
                frame = self.blank_frame.copy()
                ret = True
            else:
                ret, frame = self.cap.read()
                
                if not ret:
                    # Loop video or restart camera
                    if isinstance(self.camera_source, str):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Lost camera connection. Attempting to reconnect...")
                        if not self.initialize_camera():
                            break
                        continue
            
            # Resize frame to default editor size
            if not self.use_blank_frame:
                frame = self._resize_to_editor_size(frame)
            
            # Process frame
            display_frame = self._process_frame(frame)
            
            # Show frame
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # Key pressed
                self._handle_keyboard(key)
        
        # Cleanup
        self._cleanup()
    
    def _process_frame(self, frame):
        """
        Process frame for display with overlays.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame with overlays
        """
        display_frame = frame.copy()
        
        if self.editor_active:
            # Draw current zone
            self._draw_zone(display_frame)
            
            # Draw current counting line
            self._draw_counting_line(display_frame)
            
            # Draw editing handles if in edit mode
            if self.mode in [EditorMode.ZONE, EditorMode.LINE]:
                self._draw_handles(display_frame)
            
            # Draw temporary elements while drawing
            if self.drawing:
                self._draw_temporary(display_frame)
            
            # Draw UI overlay
            self._draw_ui_overlay(display_frame)
        else:
            # View mode - just show current configuration
            self._draw_zone(display_frame)
            self._draw_counting_line(display_frame)
            self._draw_view_overlay(display_frame)
        
        return display_frame
    
    def _resize_to_editor_size(self, frame):
        """
        Resize frame to the default editor frame size while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            
        Returns:
            Resized frame with letterboxing if needed
        """
        target_width, target_height = self.config.CONFIG_EDITOR_FRAME_SIZE
        original_height, original_width = frame.shape[:2]
        
        # Calculate scaling factor to fit within target dimensions
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas with target size and letterbox color
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate letterbox padding
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        
        # Place resized frame in center of canvas
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        
        return canvas
    
    def _draw_zone(self, frame):
        """
        Draw detection zone on frame.
        
        Args:
            frame: Frame to draw on
        """
        if not self.config.DETECTION_ZONE_ENABLED:
            return
        
        height, width = frame.shape[:2]
        zone = self.config.DETECTION_ZONE
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(zone['x1'] * width)
        y1 = int(zone['y1'] * height)
        x2 = int(zone['x2'] * width)
        y2 = int(zone['y2'] * height)
        
        # Ensure valid coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors['zone_fill'], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw border
        border_color = self.colors['zone_border'] if self.mode != EditorMode.ZONE else self.colors['handle_selected']
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 3)
        
        # Add label
        label = "Detection Zone"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), border_color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
    
    def _draw_counting_line(self, frame):
        """
        Draw counting line on frame with directional arrows.
        
        Args:
            frame: Frame to draw on
        """
        height, width = frame.shape[:2]
        
        if self.config.COUNTING_LINE_ROTATED:
            # Rotated line mode
            endpoints = self.config.COUNTING_LINE_ENDPOINTS
            
            # Convert normalized endpoints to pixel coordinates
            p1 = (int(endpoints['x1'] * width), int(endpoints['y1'] * height))
            p2 = (int(endpoints['x2'] * width), int(endpoints['y2'] * height))
            
            # Draw main line
            line_color = self.colors['line'] if self.mode != EditorMode.LINE else self.colors['handle_selected']
            cv2.line(frame, p1, p2, line_color, 3)
            
            # Calculate direction vectors
            line_dir = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            line_length = np.linalg.norm(line_dir)
            
            if line_length > 0:
                # Calculate normal vector
                normal = np.array([-line_dir[1], line_dir[0]])
                normal = normal / np.linalg.norm(normal)
                
                # Draw directional arrows
                center = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
                
                # "IN" arrow (green)
                in_end = center - normal * 30
                self._draw_arrow(frame, center, in_end, (0, 255, 0), "IN")
                
                # "OUT" arrow (red)
                out_end = center + normal * 30
                self._draw_arrow(frame, center, out_end, (0, 0, 255), "OUT")
            
            # Add label
            label_pos = (p1[0] + 10, p1[1] - 10)
            cv2.putText(frame, "Counting Line", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
        else:
            # Horizontal line mode (backward compatibility)
            if self.config.COUNTING_LINE_Y is None:
                self.config.COUNTING_LINE_Y = height // 2
            
            line_y = int(self.config.COUNTING_LINE_Y)
            
            # Draw horizontal line
            line_color = self.colors['line'] if self.mode != EditorMode.LINE else self.colors['handle_selected']
            cv2.line(frame, (0, line_y), (width, line_y), line_color, 2)
            
            # Draw directional arrows
            center_x = width // 2
            
            # "IN" arrow (green) - pointing UP
            in_end = (center_x, line_y - 30)
            self._draw_arrow(frame, (center_x, line_y), in_end, (0, 255, 0), "IN")
            
            # "OUT" arrow (red) - pointing DOWN
            out_end = (center_x, line_y + 30)
            self._draw_arrow(frame, (center_x, line_y), out_end, (0, 0, 255), "OUT")
            
            # Add label
            label_pos = (10, line_y - 10)
            cv2.putText(frame, "Counting Line", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)
    
    def _draw_arrow(self, frame, start, end, color, label):
        """
        Draw an arrow with label.
        
        Args:
            frame: Frame to draw on
            start: Start point (x, y)
            end: End point (x, y)
            color: Arrow color in BGR
            label: Text label
        """
        start = np.array(start, dtype=np.float32)
        end = np.array(end, dtype=np.float32)
        
        # Draw arrow shaft
        cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), color, 3)
        
        # Calculate arrow head
        arrow_length = 15
        arrow_angle = np.pi / 6
        
        direction = end - start
        length = np.linalg.norm(direction)
        
        if length > 0:
            direction = direction / length
            
            # Arrow head points
            left_wing = end - direction * arrow_length * np.cos(arrow_angle) + \
                       np.array([-direction[1], direction[0]]) * arrow_length * np.sin(arrow_angle)
            right_wing = end - direction * arrow_length * np.cos(arrow_angle) + \
                       np.array([direction[1], -direction[0]]) * arrow_length * np.sin(arrow_angle)
            
            cv2.line(frame, tuple(end.astype(int)), tuple(left_wing.astype(int)), color, 3)
            cv2.line(frame, tuple(end.astype(int)), tuple(right_wing.astype(int)), color, 3)
        
        # Add label
        label_pos = (int(end[0]) + 5, int(end[1]) - 5)
        cv2.putText(frame, label, label_pos,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _draw_handles(self, frame):
        """
        Draw editing handles for zone or line.
        
        Args:
            frame: Frame to draw on
        """
        height, width = frame.shape[:2]
        
        if self.mode == EditorMode.ZONE and self.config.DETECTION_ZONE_ENABLED:
            zone = self.config.DETECTION_ZONE
            corners = [
                (int(zone['x1'] * width), int(zone['y1'] * height)),  # Top-left
                (int(zone['x2'] * width), int(zone['y1'] * height)),  # Top-right
                (int(zone['x2'] * width), int(zone['y2'] * height)),  # Bottom-right
                (int(zone['x1'] * width), int(zone['y2'] * height)),  # Bottom-left
            ]
            
            for corner in corners:
                cv2.circle(frame, corner, self.handle_size, self.colors['handle'], -1)
                cv2.circle(frame, corner, self.handle_size, self.colors['text'], 2)
        
        elif self.mode == EditorMode.LINE and self.config.COUNTING_LINE_ROTATED:
            endpoints = self.config.COUNTING_LINE_ENDPOINTS
            points = [
                (int(endpoints['x1'] * width), int(endpoints['y1'] * height)),
                (int(endpoints['x2'] * width), int(endpoints['y2'] * height))
            ]
            
            for point in points:
                cv2.circle(frame, point, self.handle_size, self.colors['handle'], -1)
                cv2.circle(frame, point, self.handle_size, self.colors['text'], 2)
    
    def _draw_temporary(self, frame):
        """
        Draw temporary elements while drawing.
        
        Args:
            frame: Frame to draw on
        """
        if self.mouse_start is None or self.mouse_current is None:
            return
        
        if self.mode == EditorMode.ZONE:
            # Draw temporary zone rectangle
            x1, y1 = self.mouse_start
            x2, y2 = self.mouse_current
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.colors['zone_fill'], -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['handle_selected'], 2)
            
        elif self.mode == EditorMode.LINE:
            if len(self.line_points) == 1:
                # Draw line from first point to current mouse position
                p1 = self.line_points[0]
                p2 = self.mouse_current
                cv2.line(frame, p1, p2, self.colors['handle_selected'], 2)
    
    def _draw_ui_overlay(self, frame):
        """
        Draw UI overlay with mode indicator and instructions.
        
        Args:
            frame: Frame to draw on
        """
        # Create overlay background
        overlay_height = 180
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        cv2.addWeighted(overlay, 0.7, frame[:overlay_height, :], 0.3, 0, frame[:overlay_height, :])
        
        # Mode indicator
        mode_text = f"MODE: {self.mode.value.upper()}"
        mode_color = self.colors['handle_selected'] if self.editor_active else self.colors['text']
        cv2.putText(frame, mode_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, mode_color, 2)
        
        # Instructions based on current mode
        y_offset = 60
        instructions = self._get_instructions()
        
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            y_offset += 25
        
        # Current configuration status
        y_offset += 10
        status_lines = self._get_status_lines()
        for status in status_lines:
            cv2.putText(frame, status, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 20
    
    def _draw_view_overlay(self, frame):
        """
        Draw overlay for view mode.
        
        Args:
            frame: Frame to draw on
        """
        overlay_height = 60
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        cv2.addWeighted(overlay, 0.7, frame[:overlay_height, :], 0.3, 0, frame[:overlay_height, :])
        
        cv2.putText(frame, "VIEW MODE - Press 'e' to enter editor", (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
    
    def _get_instructions(self):
        """Get instructions based on current mode"""
        if self.mode == EditorMode.VIEW:
            return [
                "Press 'z' - Zone editing mode",
                "Press 'l' - Line editing mode",
                "Press 's' - Save configuration",
                "Press 'r' - Reset to defaults",
                "Press 'e' - Exit editor mode",
                "Press 'q' - Quit"
            ]
        elif self.mode == EditorMode.ZONE:
            return [
                "Click and drag to draw detection zone",
                "Press 'l' - Switch to line mode",
                "Press 'v' - Switch to view mode",
                "Press 'd' - Delete zone",
                "Press 's' - Save, 'r' - Reset",
                "Press 'e' - Exit editor mode"
            ]
        elif self.mode == EditorMode.LINE:
            if self.config.COUNTING_LINE_ROTATED:
                # Rotated line mode
                return [
                    "Click twice to draw rotated counting line",
                    "Press 'z' - Switch to zone mode",
                    "Press 'v' - Switch to view mode",
                    "Press 'd' - Delete line",
                    "Press 's' - Save, 'r' - Reset",
                    "Press 'e' - Exit editor mode"
                ]
            else:
                # Horizontal line mode
                return [
                    "Horizontal line mode - line at middle of frame",
                    "Press 'z' - Switch to zone mode",
                    "Press 'v' - Switch to view mode",
                    "Press 'd' - Delete line (use rotated mode instead)",
                    "Press 's' - Save, 'r' - Reset",
                    "Press 'e' - Exit editor mode"
                ]
        return []
    
    def _get_status_lines(self):
        """Get status lines showing current configuration"""
        status = []
        
        if self.config.DETECTION_ZONE_ENABLED:
            zone = self.config.DETECTION_ZONE
            status.append(f"Zone: ({zone['x1']:.2f}, {zone['y1']:.2f}) -> ({zone['x2']:.2f}, {zone['y2']:.2f})")
        else:
            status.append("Zone: Disabled")
        
        if self.config.COUNTING_LINE_ROTATED:
            line = self.config.COUNTING_LINE_ENDPOINTS
            status.append(f"Line: Rotated ({line['x1']:.2f}, {line['y1']:.2f}) -> ({line['x2']:.2f}, {line['y2']:.2f})")
        else:
            # Horizontal line mode
            if self.config.COUNTING_LINE_Y is None:
                self.config.COUNTING_LINE_Y = 0.5  # Default middle of frame
            line_y_pct = self.config.COUNTING_LINE_Y / self.frame_height if self.frame_height else 0.5
            status.append(f"Line: Horizontal at Y={line_y_pct:.1%} (IN=UP, OUT=DOWN)")
        
        return status
    
    def _mouse_callback(self, event, x, y, flags, param):
        """
        Handle mouse events for drawing and editing.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        if not self.editor_active:
            return
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_mouse_down(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._handle_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._handle_mouse_up(x, y)
    
    def _handle_mouse_down(self, x, y):
        """Handle mouse button down event"""
        if self.mode == EditorMode.ZONE:
            self.drawing = True
            self.mouse_start = (x, y)
            self.mouse_current = (x, y)
        elif self.mode == EditorMode.LINE:
            if len(self.line_points) == 0:
                self.line_points.append((x, y))
                self.drawing = True
                self.mouse_start = (x, y)
                self.mouse_current = (x, y)
            elif len(self.line_points) == 1:
                self.line_points.append((x, y))
                self._finalize_line()
    
    def _handle_mouse_move(self, x, y):
        """Handle mouse move event"""
        if self.drawing:
            self.mouse_current = (x, y)
    
    def _handle_mouse_up(self, x, y):
        """Handle mouse button up event"""
        if self.mode == EditorMode.ZONE and self.drawing:
            self._finalize_zone()
        self.drawing = False
    
    def _finalize_zone(self):
        """Finalize zone drawing and save to config"""
        if self.mouse_start is None or self.mouse_current is None:
            return
        
        x1, y1 = self.mouse_start
        x2, y2 = self.mouse_current
        
        # Ensure valid coordinates
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # Check minimum size (at least 50x50 pixels)
        if abs(x2 - x1) < 50 or abs(y2 - y1) < 50:
            print("Zone too small, ignoring")
            return
        
        # Convert to normalized coordinates
        norm_x1 = max(0.0, min(1.0, x1 / self.frame_width))
        norm_y1 = max(0.0, min(1.0, y1 / self.frame_height))
        norm_x2 = max(0.0, min(1.0, x2 / self.frame_width))
        norm_y2 = max(0.0, min(1.0, y2 / self.frame_height))
        
        # Update config
        self.config.DETECTION_ZONE = {
            'x1': norm_x1,
            'y1': norm_y1,
            'x2': norm_x2,
            'y2': norm_y2
        }
        self.config.DETECTION_ZONE_ENABLED = True
        
        print(f"Zone set: ({norm_x1:.2f}, {norm_y1:.2f}) -> ({norm_x2:.2f}, {norm_y2:.2f})")
        
        # Reset mouse state
        self.mouse_start = None
        self.mouse_current = None
    
    def _finalize_line(self):
        """Finalize line drawing and save to config"""
        if len(self.line_points) != 2:
            return
        
        p1, p2 = self.line_points
        
        # Check minimum length (at least 50 pixels)
        line_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if line_length < 50:
            print("Line too short, ignoring")
            self.line_points = []
            return
        
        # Convert to normalized coordinates
        norm_x1 = max(0.0, min(1.0, p1[0] / self.frame_width))
        norm_y1 = max(0.0, min(1.0, p1[1] / self.frame_height))
        norm_x2 = max(0.0, min(1.0, p2[0] / self.frame_width))
        norm_y2 = max(0.0, min(1.0, p2[1] / self.frame_height))
        
        # Update config
        self.config.COUNTING_LINE_ENDPOINTS = {
            'x1': norm_x1,
            'y1': norm_y1,
            'x2': norm_x2,
            'y2': norm_y2
        }
        self.config.COUNTING_LINE_ROTATED = True
        
        print(f"Line set: ({norm_x1:.2f}, {norm_y1:.2f}) -> ({norm_x2:.2f}, {norm_y2:.2f})")
        
        # Reset line points
        self.line_points = []
        self.drawing = False
    
    def _handle_keyboard(self, key):
        """
        Handle keyboard input.
        
        Args:
            key: Key code
        """
        # Toggle editor mode
        if key == ord('e'):
            self.editor_active = not self.editor_active
            if self.editor_active:
                print("Editor mode activated")
            else:
                print("Editor mode deactivated")
        
        # Only handle editing keys when editor is active
        if self.editor_active:
            # Zone mode
            if key == ord('z'):
                self.mode = EditorMode.ZONE
                self.line_points = []  # Reset line points
                print("Switched to ZONE editing mode")
            
            # Line mode
            elif key == ord('l'):
                self.mode = EditorMode.LINE
                self.line_points = []  # Reset line points
                print("Switched to LINE editing mode")
            
            # View mode
            elif key == ord('v'):
                self.mode = EditorMode.VIEW
                self.line_points = []  # Reset line points
                print("Switched to VIEW mode")
            
            # Save configuration
            elif key == ord('s'):
                self.config.save_zone_config()
                print("Configuration saved")
            
            # Reset to defaults
            elif key == ord('r'):
                self._reset_to_defaults()
                print("Reset to defaults")
            
            # Delete current element
            elif key == ord('d'):
                self._delete_current_element()
        
        # Quit
        if key == ord('q'):
            self.running = False
            print("Quitting...")
    
    def _reset_to_defaults(self):
        """Reset configuration to default values"""
        self.config.DETECTION_ZONE = {
            'x1': 0.0,
            'y1': 0.0,
            'x2': 1.0,
            'y2': 1.0
        }
        self.config.DETECTION_ZONE_ENABLED = False
        self.config.COUNTING_LINE_ROTATED = False
        self.config.COUNTING_LINE_ENDPOINTS = {
            'x1': 0.2,
            'y1': 0.5,
            'x2': 0.8,
            'y2': 0.5
        }
        self.line_points = []
    
    def _delete_current_element(self):
        """Delete the current element based on mode"""
        if self.mode == EditorMode.ZONE:
            self.config.DETECTION_ZONE_ENABLED = False
            print("Detection zone deleted")
        elif self.mode == EditorMode.LINE:
            self.config.COUNTING_LINE_ROTATED = False
            print("Counting line deleted")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Config editor closed")


def main():
    """Main entry point for running the config editor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Config Editor for Bus Counter')
    parser.add_argument('--source', type=str, default=None,
                       help='Camera source (camera index or video file path)')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index (alternative to --source)')
    
    args = parser.parse_args()
    
    # Determine camera source
    camera_source = None
    if args.source is not None:
        if args.source.isdigit():
            camera_source = int(args.source)
        else:
            camera_source = args.source
    elif args.camera is not None:
        camera_source = args.camera
    
    # Create and start editor
    editor = ConfigEditor(camera_source=camera_source)
    editor.start()


if __name__ == '__main__':
    main()
