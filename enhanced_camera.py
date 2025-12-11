#!/usr/bin/env python3
"""
Enhanced Camera Module for Raspberry Pi 5
Implements multiple camera access methods from pi5_ai_test project
Supports Pi Camera, USB Camera, and various low-latency capture methods
"""

import cv2
import subprocess
import numpy as np
import time
import socket
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

# Try to import picamera2
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
    print("picamera2 is available")
except ImportError:
    print("Warning: picamera2 not available. Will try libcamera/OpenCV.")
    PICAMERA_AVAILABLE = False


class RpicamRawCapture:
    """
    Ultra low-latency capture using rpicam-vid raw output
    Reads raw YUV frames directly - fastest method
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.running = False
        self.frame_size = int(width * height * 1.5)  # YUV420 = 1.5 bytes per pixel
        
    def start(self):
        """Start rpicam-vid with raw YUV output to pipe"""
        for cmd_name in ['rpicam-vid', 'libcamera-vid']:
            try:
                cmd = [
                    cmd_name,
                    '--nopreview',
                    '-t', '0',
                    '--width', str(self.width),
                    '--height', str(self.height),
                    '--framerate', str(self.fps),
                    '--codec', 'yuv420',
                    '--flush',
                    '-o', '-'
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=self.frame_size * 2
                )
                self.running = True
                print(f"{cmd_name} raw capture started: {self.width}x{self.height}@{self.fps}fps")
                return True
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Failed: {e}")
                continue
        return False
        
    def read(self):
        """Read a raw YUV frame and convert to BGR"""
        if not self.running or self.process is None:
            return False, None
            
        try:
            # Read exact frame size
            raw_data = self.process.stdout.read(self.frame_size)
            if len(raw_data) != self.frame_size:
                return False, None
                
            # Convert YUV420 to BGR
            yuv = np.frombuffer(raw_data, dtype=np.uint8).reshape((int(self.height * 1.5), self.width))
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            return True, frame
            
        except Exception as e:
            return False, None
            
    def release(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None
            
    def isOpened(self):
        return self.running and self.process is not None


class LibcameraCapture:
    """
    Capture frames using rpicam-vid (Pi5) or libcamera-vid and pipe to OpenCV
    This works when rpicam-hello works but picamera2 is not installed
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.running = False
        
    def start(self):
        """Start the rpicam/libcamera capture process"""
        # Try rpicam-vid first (Pi5 with Bookworm), then libcamera-vid
        commands_to_try = [
            ['rpicam-vid', '--inline', '--nopreview', '-t', '0',
             '--width', str(self.width), '--height', str(self.height),
             '--framerate', str(self.fps), '--codec', 'mjpeg', '-o', '-'],
            ['libcamera-vid', '--inline', '--nopreview', '-t', '0',
             '--width', str(self.width), '--height', str(self.height),
             '--framerate', str(self.fps), '--codec', 'mjpeg', '-o', '-'],
        ]
        
        for cmd in commands_to_try:
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8
                )
                self.running = True
                print(f"{cmd[0]} started: {self.width}x{self.height}@{self.fps}fps")
                return True
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Failed to start {cmd[0]}: {e}")
                continue
                
        return False
            
    def read(self):
        """Read a frame from the libcamera stream"""
        if not self.running or self.process is None:
            return False, None
            
        try:
            # Read JPEG data from pipe
            # JPEG starts with 0xFFD8 and ends with 0xFFD9
            jpeg_data = b''
            while True:
                byte = self.process.stdout.read(1)
                if not byte:
                    return False, None
                jpeg_data += byte
                
                # Check for JPEG start marker
                if len(jpeg_data) >= 2 and jpeg_data[-2:] == b'\xff\xd8':
                    jpeg_data = b'\xff\xd8'
                    
                # Check for JPEG end marker
                if len(jpeg_data) > 2 and jpeg_data[-2:] == b'\xff\xd9':
                    break
                    
                # Safety limit
                if len(jpeg_data) > 10**7:
                    return False, None
                    
            # Decode JPEG to frame
            frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return True, frame
                
        except Exception as e:
            print(f"Error reading frame: {e}")
            
        return False, None
        
    def release(self):
        """Stop the capture process"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            
    def isOpened(self):
        return self.running and self.process is not None


class RpicamTcpCapture:
    """
    Capture frames using rpicam-vid with TCP output - LOW LATENCY version
    """
    def __init__(self, width=640, height=480, fps=30, port=8888):
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.process = None
        self.cap = None
        self.running = False
        
    def start(self):
        """Start rpicam-vid with TCP output and connect OpenCV"""
        # Find available port
        for port in range(self.port, self.port + 10):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:  # Port is available
                self.port = port
                break
        
        # Try rpicam-vid first, then libcamera-vid
        for cmd_name in ['rpicam-vid', 'libcamera-vid']:
            try:
                # Low latency settings
                cmd = [
                    cmd_name, 
                    '--inline',
                    '--nopreview', 
                    '-t', '0',
                    '--width', str(self.width), 
                    '--height', str(self.height),
                    '--framerate', str(self.fps),
                    '--codec', 'yuv420',  # Raw format = lowest latency
                    '--flush',  # Flush output immediately
                    '--listen', 
                    '-o', f'tcp://0.0.0.0:{self.port}'
                ]
                
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Wait for server to start
                time.sleep(2)
                
                # Connect OpenCV to TCP stream
                self.cap = cv2.VideoCapture(f'tcp://127.0.0.1:{self.port}')
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.running = True
                        print(f"{cmd_name} TCP started on port {self.port}: {self.width}x{self.height}@{self.fps}fps")
                        return True
                        
                # If failed, cleanup and try next
                if self.cap:
                    self.cap.release()
                if self.process:
                    self.process.terminate()
                    self.process.wait()
                    
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"TCP capture failed with {cmd_name}: {e}")
                continue
                
        return False
        
    def read(self):
        if self.cap and self.running:
            return self.cap.read()
        return False, None
        
    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.process:
            self.process.terminate()
            self.process.wait()
            
    def isOpened(self):
        return self.running and self.cap is not None and self.cap.isOpened()


class GStreamerCapture:
    """
    Capture frames using GStreamer pipeline with libcamera
    More efficient than raw pipe method
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
    def start(self):
        """Start GStreamer capture"""
        # GStreamer pipeline for libcamera on Pi5
        gst_pipeline = (
            f'libcamerasrc ! '
            f'video/x-raw,width={self.width},height={self.height},framerate={self.fps}/1 ! '
            f'videoconvert ! '
            f'video/x-raw,format=BGR ! '
            f'appsink drop=1'
        )
        
        try:
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                print(f"GStreamer capture started: {self.width}x{self.height}@{self.fps}fps")
                return True
        except Exception as e:
            print(f"GStreamer failed: {e}")
            
        return False
        
    def read(self):
        if self.cap:
            return self.cap.read()
        return False, None
        
    def release(self):
        if self.cap:
            self.cap.release()
            
    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()


class EnhancedCamera:
    """
    Enhanced camera class with multiple capture methods
    Automatically selects the best available method
    """
    
    def __init__(self, width=640, height=480, fps=30, preferred_method=None):
        """
        Initialize enhanced camera
        
        Args:
            width: Frame width
            height: Frame height
            fps: Frames per second
            preferred_method: Preferred capture method ('picamera2', 'rpicam_raw', 'gstreamer', 
                              'rpicam_tcp', 'libcamera', 'opencv')
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.preferred_method = preferred_method
        self.camera = None
        self.camera_type = None
        self.frame_count = 0
        self.start_time = None
        
        # Initialize camera
        self.init_camera()
        
    def init_camera(self):
        """Initialize camera with multiple fallback methods"""
        print("\n" + "="*50)
        print("INITIALIZING ENHANCED CAMERA")
        print("="*50)
        
        # Define methods to try in order
        methods = []
        
        if self.preferred_method:
            # If preferred method is specified, try it first
            methods.append(self.preferred_method)
        
        # Default order of methods to try
        default_methods = [
            'picamera2',      # Best option if available
            'rpicam_raw',     # Lowest latency without picamera2
            'gstreamer',      # Efficient with libcamera
            'rpicam_tcp',     # Most reliable for Pi5
            'opencv_v4l2',    # Direct V4L2 access
            'libcamera',      # Pipe method
            'opencv'          # Regular OpenCV
        ]
        
        # Add methods not already in the list
        for method in default_methods:
            if method not in methods:
                methods.append(method)
        
        # Try each method
        for method in methods:
            print(f"Trying {method}...")
            if self._try_camera_method(method):
                print(f"✓ {method} initialized successfully")
                return
            else:
                print(f"✗ {method} failed")
                
        print("\n" + "="*50)
        print("ERROR: Could not initialize any camera!")
        print("="*50)
        print("Please try:")
        print("  1. sudo apt install -y python3-picamera2 python3-libcamera")
        print("  2. pip install picamera2 (in venv with --system-site-packages)")
        print("  3. Check: ls /dev/video*")
        print("  4. Test: rpicam-hello -t 2000")
        print("="*50)
        
    def _try_camera_method(self, method):
        """Try a specific camera method"""
        try:
            if method == 'picamera2' and PICAMERA_AVAILABLE:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.camera_type = 'picamera2'
                time.sleep(0.5)  # Wait for camera to warm up
                return True
                
            elif method == 'rpicam_raw':
                raw_cap = RpicamRawCapture(self.width, self.height, self.fps)
                if raw_cap.start():
                    time.sleep(0.5)
                    ret, test_frame = raw_cap.read()
                    if ret and test_frame is not None:
                        self.camera = raw_cap
                        self.camera_type = 'rpicam_raw'
                        return True
                    raw_cap.release()
                    
            elif method == 'gstreamer':
                gst_cap = GStreamerCapture(self.width, self.height, self.fps)
                if gst_cap.start():
                    ret, test_frame = gst_cap.read()
                    if ret and test_frame is not None:
                        self.camera = gst_cap
                        self.camera_type = 'gstreamer'
                        return True
                    gst_cap.release()
                    
            elif method == 'rpicam_tcp':
                tcp_cap = RpicamTcpCapture(self.width, self.height, self.fps)
                if tcp_cap.start():
                    self.camera = tcp_cap
                    self.camera_type = 'rpicam_tcp'
                    return True
                    
            elif method == 'opencv_v4l2':
                for dev in ['/dev/video0', '/dev/video1', '/dev/video2']:
                    try:
                        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            cap.set(cv2.CAP_PROP_FPS, self.fps)
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                self.camera = cap
                                self.camera_type = 'opencv'
                                print(f"  Using device: {dev}")
                                return True
                            cap.release()
                    except Exception as e:
                        print(f"  {dev}: {e}")
                        
            elif method == 'libcamera':
                libcam = LibcameraCapture(self.width, self.height, self.fps)
                if libcam.start():
                    time.sleep(1)  # Wait for camera to start
                    ret, test_frame = libcam.read()
                    if ret and test_frame is not None:
                        self.camera = libcam
                        self.camera_type = 'libcamera'
                        return True
                    libcam.release()
                    
            elif method == 'opencv':
                for camera_index in [0, 1, 2]:
                    try:
                        cap = cv2.VideoCapture(camera_index)
                        if cap.isOpened():
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                            cap.set(cv2.CAP_PROP_FPS, self.fps)
                            ret, test_frame = cap.read()
                            if ret and test_frame is not None:
                                self.camera = cap
                                self.camera_type = 'opencv'
                                print(f"  Using camera index: {camera_index}")
                                return True
                            cap.release()
                    except:
                        pass
                        
        except Exception as e:
            print(f"  Error: {e}")
            
        return False
        
    def read(self):
        """
        Read frame from camera
        
        Returns:
            Tuple of (success, frame)
        """
        if self.camera is None:
            return False, None
            
        try:
            if self.camera_type == 'picamera2':
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Update frame count
                if self.start_time is None:
                    self.start_time = time.time()
                self.frame_count += 1
                
                return True, frame
            else:
                # Works for all other methods
                ret, frame = self.camera.read()
                if ret and frame is not None:
                    # Update frame count
                    if self.start_time is None:
                        self.start_time = time.time()
                    self.frame_count += 1
                    
                return ret, frame
                
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
            
    def get_fps(self):
        """Calculate actual FPS"""
        if self.start_time and self.frame_count > 0:
            elapsed = time.time() - self.start_time
            return self.frame_count / elapsed
        return 0
        
    def get_camera_info(self):
        """Get camera information"""
        return {
            'camera_type': self.camera_type,
            'resolution': (self.width, self.height),
            'target_fps': self.fps,
            'actual_fps': self.get_fps(),
            'frame_count': self.frame_count
        }
        
    def release(self):
        """Release camera resources"""
        if self.camera is None:
            return
            
        try:
            if self.camera_type == 'picamera2':
                self.camera.stop()
                self.camera.close()
            elif hasattr(self.camera, 'release'):
                self.camera.release()
                
            print(f"Camera ({self.camera_type}) released")
            
        except Exception as e:
            print(f"Error releasing camera: {e}")
            
        finally:
            self.camera = None
            self.camera_type = None
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def create_enhanced_camera(width=640, height=480, fps=30, preferred_method=None):
    """
    Factory function to create enhanced camera
    
    Args:
        width: Frame width
        height: Frame height
        fps: Frames per second
        preferred_method: Preferred capture method
        
    Returns:
        EnhancedCamera instance
    """
    return EnhancedCamera(width, height, fps, preferred_method)


def test_all_methods(width=640, height=480, fps=30):
    """
    Test all camera methods to see which ones work
    
    Args:
        width: Frame width
        height: Frame height
        fps: Frames per second
        
    Returns:
        List of working methods
    """
    methods = [
        'picamera2',
        'rpicam_raw',
        'gstreamer',
        'rpicam_tcp',
        'opencv_v4l2',
        'libcamera',
        'opencv'
    ]
    
    working_methods = []
    
    print("Testing all camera methods...")
    print("="*50)
    
    for method in methods:
        print(f"\nTesting {method}...")
        try:
            cam = EnhancedCamera(width, height, fps, preferred_method=method)
            if cam.camera is not None:
                ret, frame = cam.read()
                if ret and frame is not None:
                    print(f"✓ {method} works!")
                    working_methods.append(method)
                else:
                    print(f"✗ {method} failed to read frame")
            else:
                print(f"✗ {method} failed to initialize")
            cam.release()
        except Exception as e:
            print(f"✗ {method} error: {e}")
            
    print("\n" + "="*50)
    print(f"Working methods: {working_methods}")
    print("="*50)
    
    return working_methods


if __name__ == "__main__":
    # Test all camera methods
    working = test_all_methods()
    
    # Create camera with best available method
    if working:
        print(f"\nCreating camera with best method: {working[0]}")
        with create_enhanced_camera(preferred_method=working[0]) as cam:
            print(f"Camera info: {cam.get_camera_info()}")
            
            # Test reading frames
            print("\nTesting frame capture...")
            for i in range(10):
                ret, frame = cam.read()
                if ret:
                    print(f"Frame {i+1}: {frame.shape}")
                else:
                    print(f"Failed to read frame {i+1}")
                    
            print(f"\nFinal FPS: {cam.get_fps():.2f}")
    else:
        print("\nNo working camera methods found!")