import os
import platform

class Config:
    def __init__(self):
        # --- Platform Detection ---
        self.IS_RASPBERRY_PI = platform.machine() in ('armv7l', 'aarch64')
        self.IS_PI5 = self.IS_RASPBERRY_PI and platform.processor() == ''
        self.IS_WINDOWS = platform.system().lower() == 'windows'
        
        # --- Device Detection ---
        self.CUDA_AVAILABLE = self._check_cuda_available()
        self.HAILO_AVAILABLE = self._check_hailo_available()
        
        # --- YOLO Model Configuration ---
        # Platform-specific model selection
        if self.IS_PI5 and self.HAILO_AVAILABLE:
            # Raspberry Pi 5 with HAILO accelerator
            self.YOLO_MODEL = "yolov8n.hef"  # HAILO compiled model
            self.USE_HAILO = True
            self.DEVICE = "hailo"
        elif self.IS_WINDOWS and self.CUDA_AVAILABLE:
            # Windows with CUDA GPU
            self.YOLO_MODEL = "medium.pt"  # Regular PyTorch model
            self.USE_HAILO = False
            self.DEVICE = "cuda"
        elif self.CUDA_AVAILABLE:
            # Other platforms with CUDA GPU
            self.YOLO_MODEL = "medium.pt"  # Regular PyTorch model
            self.USE_HAILO = False
            self.DEVICE = "cuda"
        else:
            # Fallback to CPU
            self.YOLO_MODEL = "medium.pt"  # Regular PyTorch model
            self.USE_HAILO = False
            self.DEVICE = "cpu"
            
        # LOWERED: High confidence kills tracking in crowds.
        # 0.4 - 0.5 is usually the sweet spot for tracking.
        self.YOLO_CONFIDENCE = 0.5
        self.YOLO_IOU_THRESHOLD = 0.45
        self.YOLO_CLASSES = [0]  # Focus ONLY on Person (0) if you are doing passenger counting to save resources
        
        # --- BoTSORT Configuration ---
        self.BOTSORT_TRACKER = {
            # Must be slightly higher or equal to YOLO_CONFIDENCE
            'track_high_thresh': 0.65, 
            
            'track_low_thresh': 0.45,
            
            # LOWERED: Easier to start tracking a new person entering the frame
            'new_track_thresh': 0.65, 
            
            'track_buffer': 40,  # INCREASED: Keep "lost" tracks in memory for 2 seconds (at 30fps) to recover from occlusions
            
            # CRITICAL FIX: 0.9 is too strict. 0.7 allows for movement between frames.
            # If this is too high, you get "Ghosting" (tracker creates new ID for same person).
            'match_thresh': 0.5, 
            
            # Adjusted for standard BoTSORT behavior
            'proximity_thresh': 0.6, 
            
            'appearance_thresh': 0.20, 
            'with_reid': False,
            
            # sparseOptFlow is slow. If camera is fixed, use None. If moving (on bus), 'gmc' is often faster/better.
            'cmc_method': 'gmc', 
            'frame_rate': 30,
            'fuse_score': True
        }
        
        # --- ReID Configuration ---
        # If you have a good GPU, 'osnet_x1_0_msmt17.pt' is much better for multi-person distinction
        # than 'x0_25', though slightly slower.
        self.REID_MODEL = "osnet_x0_25_msmt17.pt"
        self.REID_DEVICE = self.DEVICE  # Use the same device as main detector
        
        # --- HAILO Configuration ---
        self.HAILO_DEVICE = "hailo0"  # Default HAILO device
        self.HAILO_BATCH_SIZE = 1  # Batch size for HAILO inference
        self.HAILO_INPUT_FORMAT = "RGB"  # Input format for HAILO
        
        # --- Video Processing ---
        # Focus on cam0 for Pi5, use default for development
        if self.IS_PI5:
            # CHANGE: Use integer 0 instead of string "/dev/video0" for better OpenCV compatibility
            self.VIDEO_SOURCE = 0
            self.CAMERA_TYPE = "cam_module"  # Changed from "usb" to "cam_module" for camera module
            
            # ADD THESE LINES (Crucial for Pi 5 Camera Module):
            self.CAM_WIDTH = 640
            self.CAM_HEIGHT = 480
            self.CAM_FPS = 30
        else:
            self.VIDEO_SOURCE = 0
            self.CAMERA_TYPE = "cam_module"  # Changed from "usb" to "cam_module" for camera module
            self.CAM_WIDTH = 1280
            self.CAM_HEIGHT = 720
            self.CAM_FPS = 30
            
        # --- Enhanced Camera Settings (from pi5_ai_test) ---
        # Camera access method priority - optimized for camera module
        self.CAMERA_METHODS = [
            'picamera2',      # Best option for Pi camera module
            'rpicam_raw',     # Lowest latency for Pi camera module
            'gstreamer',      # Efficient with libcamera for Pi camera
            'rpicam_tcp',     # Most reliable for Pi5 camera module
            'opencv_v4l2',    # Direct V4L2 access for camera module
            'libcamera',      # Pipe method for camera module
            'opencv'          # Regular OpenCV as fallback
        ]
        
        # Preferred camera method - prioritize picamera2 for camera module
        self.PREFERRED_CAMERA_METHOD = 'picamera2'
        
        # Low latency settings
        self.LOW_LATENCY_MODE = True  # Enable low latency camera mode
        self.SKIP_FRAMES = 0  # Skip frames to reduce latency (0 = no skip)
        
        # Camera buffer settings for low latency
        self.CAMERA_BUFFER_SIZE = 1  # Minimize buffer for lower latency
        
        # TCP settings for rpicam_tcp method
        self.TCP_PORT_START = 8888  # Starting port for TCP capture
            
        self.OUTPUT_VIDEO = "output_tracking.mp4"
        self.SAVE_VIDEO = True
        self.SHOW_VIDEO = True
        
        # --- Display Configuration ---
        # Optimize for Pi5 performance
        if self.IS_PI5:
            self.DISPLAY_SIZE = (1024, 768)  # Smaller for Pi5 performance
        else:
            self.DISPLAY_SIZE = (1280, 720)  # Full resolution for development
            
        self.FONT_SIZE = 0.6
        self.FONT_THICKNESS = 2
        self.LINE_THICKNESS = 2
        
        # --- Colors (BGR) ---
        self.COLORS = {
            'bbox': (0, 255, 0),
            'text': (255, 255, 255),
            'trail': (255, 0, 0),
            'center': (0, 0, 255)
        }
        
        # --- Bus Counting Configuration ---
        self.COUNTING_LINE_Y = None 
        self.COUNTING_DIRECTION = "both"
        self.MIN_TRACK_AGE = 3 # Lowered: Count faster
        
        # --- Performance ---
        # Optimize for Pi5 with HAILO 8L
        if self.IS_PI5:
            self.MAX_DETECTIONS = 50  # Reduced for Pi5
            self.PROCESS_EVERY_N_FRAMES = 2  # Skip frames for better performance
            self.DEBUG_FRAME_INTERVAL = 30  # Less frequent debug on Pi5
            self.ENABLE_REID = False  # Disable ReID for performance on Pi5
        else:
            self.MAX_DETECTIONS = 100
            self.PROCESS_EVERY_N_FRAMES = 1
            self.DEBUG_FRAME_INTERVAL = 10
            self.ENABLE_REID = True
    
    def _check_cuda_available(self):
        """Check if CUDA GPU is available"""
        try:
            import torch
            # Check if PyTorch is installed and CUDA is available
            if not torch.cuda.is_available():
                print("CUDA not available in PyTorch")
                return False
            
            # Check if we can actually access the GPU
            try:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"Found {gpu_count} CUDA GPU(s): {gpu_name}")
                    return True
                else:
                    print("No CUDA GPUs found")
                    return False
            except Exception as e:
                print(f"Error accessing CUDA GPU: {e}")
                return False
        except ImportError:
            print("PyTorch not available for CUDA detection")
            return False
    
    def _check_hailo_available(self):
        """Check if HAILO SDK is available"""
        try:
            from hailo_platform import HEF, VDevice
            return True
        except ImportError:
            return False
    
    def get_device_info(self):
        """Get information about the available device"""
        info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'device': self.DEVICE,
            'use_hailo': self.USE_HAILO,
            'model': self.YOLO_MODEL
        }
        
        if self.DEVICE == "cuda":
            try:
                import torch
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                if torch.cuda.is_available():
                    info['gpu_name'] = torch.cuda.get_device_name(0)
            except:
                pass
        elif self.DEVICE == "hailo":
            info['hailo_device'] = getattr(self, 'HAILO_DEVICE', 'hailo0')
        
        return info
    
    def print_device_info(self):
        """Print device information"""
        info = self.get_device_info()
        print("=" * 50)
        print("DEVICE CONFIGURATION")
        print("=" * 50)
        print(f"Platform: {info['platform']} ({info['machine']})")
        print(f"Device: {info['device']}")
        print(f"Model: {info['model']}")
        print(f"Use HAILO: {info['use_hailo']}")
        
        if info['device'] == "cuda":
            print(f"CUDA Version: {info.get('cuda_version', 'Unknown')}")
            print(f"GPU Count: {info.get('gpu_count', 0)}")
            if 'gpu_name' in info:
                print(f"GPU Name: {info['gpu_name']}")
        elif info['device'] == "hailo":
            print(f"HAILO Device: {info.get('hailo_device', 'hailo0')}")
        print("=" * 50)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)