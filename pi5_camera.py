"""
Raspberry Pi 5 Camera Module for cam0
This module provides optimized camera initialization and handling for Raspberry Pi 5
"""

import cv2
import numpy as np
import time
import platform
from typing import Optional, Tuple, Dict, Any

try:
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("Warning: picamera2 not available. Falling back to OpenCV.")
    PICAMERA2_AVAILABLE = False

# Export PI5_CAMERA_AVAILABLE for compatibility with import statements
PI5_CAMERA_AVAILABLE = PICAMERA2_AVAILABLE


class Pi5Camera:
    """
    Raspberry Pi 5 optimized camera handler
    Supports both USB cameras and Pi camera modules with focus on cam0
    """
    
    def __init__(self, camera_type: str = "auto", camera_index: str = "/dev/video0", 
                 resolution: Tuple[int, int] = (1920, 1080), fps: int = 30):
        """
        Initialize Pi5 camera
        
        Args:
            camera_type: Type of camera ("usb", "rpi", "csi", or "auto")
            camera_index: Camera device path or index
            resolution: Camera resolution (width, height)
            fps: Frames per second
        """
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.picam2 = None
        self.is_pi5 = self._detect_pi5()
        self.frame_count = 0
        self.last_frame_time = 0
        self.actual_fps = 0
        
        # Camera configuration
        self.config = {
            'resolution': resolution,
            'fps': fps,
            'format': 'BGR888',
            'buffer_count': 3
        }
        
        # Initialize camera
        self._initialize_camera()
    
    def _detect_pi5(self) -> bool:
        """Detect if running on Raspberry Pi 5"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'BCM2712' in cpuinfo  # Pi5 uses BCM2712
        except:
            return False
    
    def _initialize_camera(self):
        """Initialize camera based on type and availability"""
        if self.camera_type == "auto":
            self.camera_type = self._auto_detect_camera()
        
        print(f"Initializing camera: type={self.camera_type}, index={self.camera_index}")
        
        if self.camera_type == "rpi" and PICAMERA2_AVAILABLE:
            self._initialize_picamera2()
        else:
            self._initialize_opencv_camera()
    
    def _auto_detect_camera(self) -> str:
        """Auto-detect available camera type"""
        # Try Pi camera first if available
        if PICAMERA2_AVAILABLE and self._check_picamera_available():
            print("Detected Raspberry Pi camera module")
            return "rpi"
        
        # Fall back to USB camera
        print("Using USB camera")
        return "usb"
    
    def _check_picamera_available(self) -> bool:
        """Check if Pi camera is available"""
        try:
            # Try to create a temporary Picamera2 instance
            test_cam = Picamera2()
            test_cam.stop()
            return True
        except:
            return False
    
    def _initialize_picamera2(self):
        """Initialize Raspberry Pi camera using picamera2"""
        try:
            self.picam2 = Picamera2()
            
            # Configure camera for optimal performance
            config = self.picam2.create_video_configuration(
                main=self.config,
                lores=None,
                display=None,
                transform=None,
                colour_space=self.config['format']
            )
            
            # Apply configuration
            self.picam2.configure(config)
            
            # Set camera controls for better quality
            controls_dict = {
                controls.AeEnable: True,          # Auto exposure
                controls.AwbEnable: True,         # Auto white balance
                controls.AfMode: controls.AfModeEnum.Continuous,  # Auto focus (if available)
                controls.NoiseReductionMode: controls.NoiseReductionModeEnum.HighQuality,
                controls.Sharpness: 1.0,
                controls.Contrast: 1.0,
                controls.Saturation: 1.0,
            }
            
            # Apply controls
            self.picam2.set_controls(controls_dict)
            
            # Start camera
            self.picam2.start()
            
            # Wait for camera to stabilize
            time.sleep(2.0)
            
            print("PiCamera2 initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize PiCamera2: {e}")
            print("Falling back to OpenCV camera")
            self._initialize_opencv_camera()
    
    def _initialize_opencv_camera(self):
        """Initialize USB camera using OpenCV"""
        try:
            # Handle both string device paths and integer indices
            if isinstance(self.camera_index, str) and self.camera_index.startswith('/dev/video'):
                # For Linux device paths, we need to find the corresponding index
                cap_index = self._get_camera_index_from_device(self.camera_index)
            else:
                cap_index = int(self.camera_index) if isinstance(self.camera_index, str) else self.camera_index
            
            self.cap = cv2.VideoCapture(cap_index)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.camera_index}")
            
            # Set camera properties for optimal performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for lower latency
            
            # Additional settings for USB cameras
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Auto exposure
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Auto white balance
            
            # Wait for camera to stabilize
            time.sleep(1.0)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"OpenCV camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenCV camera: {e}")
    
    def _get_camera_index_from_device(self, device_path: str) -> int:
        """Convert device path to camera index"""
        try:
            # Extract number from /dev/videoX
            device_num = int(device_path.split('video')[-1])
            return device_num
        except:
            # Fallback to 0
            return 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        current_time = time.time()
        
        try:
            if self.picam2 is not None:
                # Use picamera2
                frame = self.picam2.capture_array()
                
                # Convert from RGB to BGR for OpenCV compatibility
                if frame.shape[2] == 3:  # RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Calculate actual FPS
                if self.last_frame_time > 0:
                    frame_time = current_time - self.last_frame_time
                    self.actual_fps = 1.0 / frame_time
                
                self.last_frame_time = current_time
                self.frame_count += 1
                
                return True, frame
            
            elif self.cap is not None:
                # Use OpenCV
                ret, frame = self.cap.read()
                
                if ret:
                    # Calculate actual FPS
                    if self.last_frame_time > 0:
                        frame_time = current_time - self.last_frame_time
                        self.actual_fps = 1.0 / frame_time
                    
                    self.last_frame_time = current_time
                    self.frame_count += 1
                
                return ret, frame
            
            else:
                return False, None
                
        except Exception as e:
            return False, None
    
    def release(self):
        """Release camera resources"""
        try:
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
                print("PiCamera2 released")
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                print("OpenCV camera released")
                
        except Exception as e:
            print(f"Error releasing camera: {e}")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information"""
        info = {
            'camera_type': self.camera_type,
            'camera_index': self.camera_index,
            'resolution': self.resolution,
            'target_fps': self.fps,
            'actual_fps': self.actual_fps,
            'frame_count': self.frame_count,
            'is_pi5': self.is_pi5
        }
        
        if self.picam2 is not None:
            # Get PiCamera2 specific info
            sensor_modes = self.picam2.sensor_modes
            info['sensor_modes'] = len(sensor_modes) if sensor_modes else 0
            
        elif self.cap is not None:
            # Get OpenCV camera specific info
            info['backend'] = self.cap.getBackendName()
            info['buffer_size'] = int(self.cap.get(cv2.CAP_PROP_BUFFERSIZE))
            info['brightness'] = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            info['contrast'] = self.cap.get(cv2.CAP_PROP_CONTRAST)
            info['saturation'] = self.cap.get(cv2.CAP_PROP_SATURATION)
        
        return info
    
    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        self.resolution = (width, height)
        
        if self.picam2 is not None:
            # Reconfigure picamera2
            self.config['resolution'] = (width, height)
            config = self.picam2.create_video_configuration(main=self.config)
            self.picam2.configure(config)
        
        elif self.cap is not None:
            # Set OpenCV camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def set_fps(self, fps: int):
        """Set camera FPS"""
        self.fps = fps
        
        if self.picam2 is not None:
            # Reconfigure picamera2
            self.config['fps'] = fps
            config = self.picam2.create_video_configuration(main=self.config)
            self.picam2.configure(config)
        
        elif self.cap is not None:
            # Set OpenCV camera FPS
            self.cap.set(cv2.CAP_PROP_FPS, fps)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def create_pi5_camera(camera_type: str = "auto", camera_index: str = "/dev/video0",
                    resolution: Tuple[int, int] = (1920, 1080), fps: int = 30) -> Pi5Camera:
    """
    Factory function to create Pi5 camera
    
    Args:
        camera_type: Type of camera ("usb", "rpi", "csi", or "auto")
        camera_index: Camera device path or index
        resolution: Camera resolution (width, height)
        fps: Frames per second
        
    Returns:
        Pi5Camera instance
    """
    return Pi5Camera(camera_type, camera_index, resolution, fps)


def find_available_cameras() -> list:
    """
    Find available cameras on the system
    
    Returns:
        List of available camera information
    """
    cameras = []
    
    # Check for Pi camera
    if PICAMERA2_AVAILABLE:
        try:
            test_cam = Picamera2()
            cameras.append({
                'type': 'rpi',
                'index': 'picamera',
                'name': 'Raspberry Pi Camera',
                'available': True
            })
            test_cam.stop()
        except:
            pass
    
    # Check for USB cameras
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras.append({
                    'type': 'usb',
                    'index': i,
                    'name': f'USB Camera {i}',
                    'resolution': f'{width}x{height}',
                    'available': True
                })
            cap.release()
    
    return cameras


def print_camera_info():
    """Print information about available cameras"""
    cameras = find_available_cameras()
    
    print("Available Cameras:")
    print("=" * 50)
    
    for i, cam in enumerate(cameras):
        print(f"{i}: {cam['name']}")
        print(f"   Type: {cam['type']}")
        print(f"   Index: {cam['index']}")
        if 'resolution' in cam:
            print(f"   Resolution: {cam['resolution']}")
        print(f"   Available: {cam['available']}")
        print()
    
    if not cameras:
        print("No cameras found!")
        print("Please check:")
        print("1. Camera is properly connected")
        print("2. Camera drivers are installed")
        print("3. User has permission to access camera devices")