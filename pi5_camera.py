"""
Raspberry Pi 5 Camera Module for cam0
This module provides optimized camera initialization and handling for Raspberry Pi 5
"""

import cv2
import numpy as np
import time
import platform
from typing import Optional, Tuple, Dict, Any

# Check if GStreamer is available for Pi camera support
import subprocess
import sys

def check_gstreamer():
    """Check if GStreamer is available"""
    try:
        result = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.getBuildInformation())"],
                              capture_output=True, text=True)
        return "GStreamer" in result.stdout
    except:
        return False

GSTREAMER_AVAILABLE = check_gstreamer()
PI5_CAMERA_AVAILABLE = GSTREAMER_AVAILABLE


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
        
        if self.camera_type == "rpi" and GSTREAMER_AVAILABLE:
            self._initialize_gstreamer_camera()
        else:
            self._initialize_opencv_camera()
    
    def _auto_detect_camera(self) -> str:
        """Auto-detect available camera type"""
        # Try Pi camera first if GStreamer is available
        if GSTREAMER_AVAILABLE and self._check_picamera_available():
            print("Detected Raspberry Pi camera module (using GStreamer)")
            return "rpi"
        
        # Fall back to USB camera
        print("Using USB camera")
        return "usb"
    
    def _check_picamera_available(self) -> bool:
        """Check if Pi camera is available using GStreamer"""
        pipelines = [
            # Try libcamerasrc first (newer Pi setups)
            "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=1",
            # Try rpicamsrc (older Pi setups)
            "rpicamsrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=1",
            # Try v4l2src as fallback
            "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=1"
        ]
        
        for pipeline in pipelines:
            try:
                test_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    test_cap.release()
                    if ret and frame is not None:
                        return True
            except:
                pass
        
        return False
    
    def _initialize_gstreamer_camera(self):
        """Initialize Raspberry Pi camera using GStreamer pipeline"""
        try:
            # Try different GStreamer pipeline configurations for Raspberry Pi camera
            pipelines = [
                # Pipeline 1: Using libcamera source (newer Pi 5 setups)
                (
                    f"libcamerasrc "
                    f"! video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.fps}/1 "
                    f"! videoconvert "
                    f"! video/x-raw,format=BGR "
                    f"! appsink drop=1"
                ),
                # Pipeline 2: Using rpicamsrc (older Pi setups)
                (
                    f"rpicamsrc "
                    f"preview=false "
                    f"! video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.fps}/1 "
                    f"! videoconvert "
                    f"! video/x-raw,format=BGR "
                    f"! appsink drop=1"
                ),
                # Pipeline 3: Using v4l2src as fallback
                (
                    f"v4l2src device=/dev/video0 "
                    f"! video/x-raw,width={self.resolution[0]},height={self.resolution[1]},framerate={self.fps}/1 "
                    f"! videoconvert "
                    f"! video/x-raw,format=BGR "
                    f"! appsink drop=1"
                )
            ]
            
            pipeline_success = False
            for i, pipeline in enumerate(pipelines, 1):
                try:
                    print(f"Attempting GStreamer pipeline {i}: {pipeline}")
                    
                    # Initialize OpenCV with GStreamer backend
                    self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                    
                    if self.cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = self.cap.read()
                        if ret and test_frame is not None:
                            pipeline_success = True
                            print(f"GStreamer pipeline {i} successful")
                            break
                        else:
                            self.cap.release()
                    else:
                        self.cap.release()
                        
                except Exception as pipeline_error:
                    print(f"Pipeline {i} failed: {pipeline_error}")
                    if self.cap is not None:
                        self.cap.release()
                    self.cap = None
                    continue
            
            if not pipeline_success:
                raise RuntimeError("All GStreamer pipeline attempts failed")
            
            # Wait for camera to stabilize
            time.sleep(2.0)
            
            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"GStreamer Pi camera initialized: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")
            
        except Exception as e:
            print(f"Failed to initialize GStreamer camera: {e}")
            print("Falling back to USB camera")
            self.camera_type = "usb"
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
            if self.cap is not None:
                # Use OpenCV (with or without GStreamer backend)
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
            print(f"Error reading frame: {e}")
            return False, None
    
    def release(self):
        """Release camera resources"""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                if self.camera_type == "rpi":
                    print("GStreamer Pi camera released")
                else:
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
        
        if self.cap is not None:
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
        
        if self.cap is not None:
            # Set OpenCV camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # For GStreamer, we need to reinitialize the pipeline with new resolution
            if self.camera_type == "rpi":
                print("Note: For GStreamer Pi camera, you may need to reinitialize the camera for resolution changes to take effect.")
    
    def set_fps(self, fps: int):
        """Set camera FPS"""
        self.fps = fps
        
        if self.cap is not None:
            # Set OpenCV camera FPS
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            
            # For GStreamer, we need to reinitialize the pipeline with new FPS
            if self.camera_type == "rpi":
                print("Note: For GStreamer Pi camera, you may need to reinitialize the camera for FPS changes to take effect.")
    
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
    
    # Check for Pi camera using GStreamer
    if GSTREAMER_AVAILABLE:
        pipelines = [
            ("libcamerasrc", "Raspberry Pi Camera (libcamera)"),
            ("rpicamsrc", "Raspberry Pi Camera (rpicamsrc)"),
            ("v4l2src device=/dev/video0", "Raspberry Pi Camera (V4L2)")
        ]
        
        for pipeline_source, camera_name in pipelines:
            try:
                pipeline = f"{pipeline_source} ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink drop=1"
                test_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if test_cap.isOpened():
                    ret, frame = test_cap.read()
                    test_cap.release()
                    if ret and frame is not None:
                        cameras.append({
                            'type': 'rpi',
                            'index': pipeline_source,
                            'name': camera_name,
                            'available': True
                        })
                        break  # Found working Pi camera, stop checking other pipelines
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