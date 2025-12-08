import os

class Config:
    def __init__(self):
        # YOLO Model Configuration
        self.YOLO_MODEL = "yolo11s.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        self.YOLO_CONFIDENCE = 0.3
        self.YOLO_IOU_THRESHOLD = 0.45
        self.YOLO_CLASSES = [0, 2, 3, 5, 7]  # Person, car, motorcycle, bus, truck classes (COCO dataset)
        
        # BoTSORT Configuration
        self.BOTSORT_TRACKER = {
            'track_high_thresh': 0.6,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.7,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.25,
            'with_reid': True,
            'cmc_method': 'sparseOptFlow',
            'frame_rate': 30,
            'fuse_score': True
        }
        
        # ReID Configuration
        self.REID_MODEL = "osnet_x0_25_msmt17.pt"
        self.REID_DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
        
        # Video Processing
        self.VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
        self.OUTPUT_VIDEO = "output_tracking.mp4"
        self.SAVE_VIDEO = True
        self.SHOW_VIDEO = True
        
        # Display Configuration
        self.DISPLAY_SIZE = (1280, 720)
        self.FONT_SIZE = 0.6
        self.FONT_THICKNESS = 2
        self.LINE_THICKNESS = 2
        
        # Colors for visualization (BGR format)
        self.COLORS = {
            'bbox': (0, 255, 0),  # Green
            'text': (255, 255, 255),  # White
            'trail': (255, 0, 0),  # Blue
            'center': (0, 0, 255)  # Red
        }
        
        # Bus Counting Configuration
        self.COUNTING_LINE_Y = None  # Will be set based on video height
        self.COUNTING_DIRECTION = "both"  # "up", "down", or "both"
        self.MIN_TRACK_AGE = 5  # Minimum frames before counting
        
        # Performance
        self.MAX_DETECTIONS = 100
        self.PROCESS_EVERY_N_FRAMES = 1
    
    def __getitem__(self, key):
        """Allow dictionary-like access to attributes"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Allow dictionary-like setting of attributes"""
        setattr(self, key, value)