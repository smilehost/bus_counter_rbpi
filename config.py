import os

class Config:
    def __init__(self):
        # --- YOLO Model Configuration ---
        self.YOLO_MODEL = "yolo11s.pt"
        # LOWERED: High confidence kills tracking in crowds. 
        # 0.4 - 0.5 is usually the sweet spot for tracking.
        self.YOLO_CONFIDENCE = 0.45  
        self.YOLO_IOU_THRESHOLD = 0.5
        self.YOLO_CLASSES = [0]  # Focus ONLY on Person (0) if you are doing passenger counting to save resources
        
        # --- BoTSORT Configuration ---
        self.BOTSORT_TRACKER = {
            # Must be slightly higher or equal to YOLO_CONFIDENCE
            'track_high_thresh': 0.5, 
            
            'track_low_thresh': 0.1,
            
            # LOWERED: Easier to start tracking a new person entering the frame
            'new_track_thresh': 0.4, 
            
            'track_buffer': 60,  # INCREASED: Keep "lost" tracks in memory for 2 seconds (at 30fps) to recover from occlusions
            
            # CRITICAL FIX: 0.9 is too strict. 0.7 allows for movement between frames.
            # If this is too high, you get "Ghosting" (tracker creates new ID for same person).
            'match_thresh': 0.7, 
            
            # Adjusted for standard BoTSORT behavior
            'proximity_thresh': 0.5, 
            
            'appearance_thresh': 0.25, 
            'with_reid': True,
            
            # sparseOptFlow is slow. If camera is fixed, use None. If moving (on bus), 'gmc' is often faster/better.
            'cmc_method': 'sparseOptFlow', 
            'frame_rate': 30,
            'fuse_score': True
        }
        
        # --- ReID Configuration ---
        # If you have a good GPU, 'osnet_x1_0_msmt17.pt' is much better for multi-person distinction
        # than 'x0_25', though slightly slower.
        self.REID_MODEL = "osnet_x0_25_msmt17.pt" 
        self.REID_DEVICE = "cuda" if os.system("nvidia-smi") == 0 else "cpu"
        
        # --- Video Processing ---
        self.VIDEO_SOURCE = 0 
        self.OUTPUT_VIDEO = "output_tracking.mp4"
        self.SAVE_VIDEO = True
        self.SHOW_VIDEO = True
        
        # --- Display Configuration ---
        self.DISPLAY_SIZE = (1280, 720)
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
        self.MAX_DETECTIONS = 100
        self.PROCESS_EVERY_N_FRAMES = 1
        self.DEBUG_FRAME_INTERVAL = 10 
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)