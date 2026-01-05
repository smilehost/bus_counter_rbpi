from abc import ABC, abstractmethod
import numpy as np

class BaseTracker(ABC):
    """Abstract base class for all trackers
    
    This defines the common interface that all tracker implementations must follow.
    """
    
    def __init__(self, config):
        """Initialize the tracker with configuration
        
        Args:
            config: Dictionary containing tracker configuration parameters
        """
        self.config = config
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
        self.device = config.get('device', 'cpu')
    
    @abstractmethod
    def update(self, detections, frame):
        """Update tracker with new detections
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, confidence, class_id, ...]
            frame: Current frame (for ReID feature extraction if supported)
            
        Returns:
            List of active track objects
        """
        pass
    
    def _predict_tracks(self):
        """Predict next positions for all tracks
        
        Default implementation that can be overridden by subclasses
        """
        for track in self.tracks:
            if hasattr(track, 'predict'):
                track.predict()
    
    def extract_reid_features(self, frame, boxes):
        """Extract ReID features from detected bounding boxes
        
        Default implementation returns None (not supported).
        Subclasses that support ReID should override this method.
        
        Args:
            frame: Current frame
            boxes: Array of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            None or array of ReID features
        """
        return None
