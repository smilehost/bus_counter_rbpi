import numpy as np
from collections import defaultdict, deque
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from .base_tracker import BaseTracker

class ByteTrack(BaseTracker):
    """
    ByteTrack implementation - Multi-Object Tracking with simple IoU-based matching
    Uses a two-stage matching strategy for better handling of low-confidence detections
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # ByteTrack-specific parameters
        self.track_thresh = config.get('track_thresh', 0.5)
        self.track_buffer = config.get('track_buffer', 30)
        self.match_thresh = config.get('match_thresh', 0.8)
        self.frame_rate = config.get('frame_rate', 30)
        
        # High and low confidence thresholds for two-stage matching
        self.high_thresh = config.get('high_thresh', 0.6)
        self.low_thresh = config.get('low_thresh', 0.1)
        
        # Track history for management
        self.lost_tracks = []
        self.removed_tracks = []
        
        print(f"ByteTrack initialized with track_thresh={self.track_thresh}, match_thresh={self.match_thresh}")
    
    def update(self, detections, frame):
        """Update tracker with new detections using two-stage matching"""
        self.frame_count += 1
        
        # Handle empty detections
        if len(detections) == 0:
            self._manage_lost_tracks()
            return self._get_active_tracks()
        
        # Extract detection information
        boxes = detections[:, :4]  # x1, y1, x2, y2
        scores = detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections))
        class_ids = detections[:, 5] if detections.shape[1] > 5 else np.zeros(len(detections))
        
        # Predict track positions
        self._predict_tracks()
        
        # Split detections into high and low confidence
        high_dets_mask = scores >= self.high_thresh
        low_dets_mask = (scores >= self.low_thresh) & (scores < self.high_thresh)
        
        high_boxes = boxes[high_dets_mask]
        high_scores = scores[high_dets_mask]
        high_class_ids = class_ids[high_dets_mask]
        
        low_boxes = boxes[low_dets_mask]
        low_scores = scores[low_dets_mask]
        low_class_ids = class_ids[low_dets_mask]
        
        # First stage: match high-confidence detections
        matched_tracks, unmatched_high_dets, unmatched_trks = self._match_detections(
            high_boxes, high_scores
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(high_boxes[det_idx], high_scores[det_idx], 
                                          class_id=high_class_ids[det_idx])
        
        # Second stage: match low-confidence detections to unmatched tracks
        if len(unmatched_trks) > 0 and len(low_boxes) > 0:
            # Filter tracks to only unmatched ones
            unmatched_track_objects = [self.tracks[i] for i in unmatched_trks]
            
            # Match low-confidence detections
            low_matched, low_unmatched_dets, _ = self._match_detections(
                low_boxes, low_scores, track_indices=unmatched_trks
            )
            
            # Update matched tracks
            for track_idx, det_idx in low_matched:
                actual_track_idx = unmatched_trks[track_idx]
                self.tracks[actual_track_idx].update(low_boxes[det_idx], low_scores[det_idx],
                                                     class_id=low_class_ids[det_idx])
            
            # Update unmatched high detections with low unmatched detections
            unmatched_high_dets = list(unmatched_high_dets)
            unmatched_high_dets.extend([i for i in low_unmatched_dets])
        
        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_high_dets:
            if high_scores[det_idx] > self.track_thresh:
                self._create_new_track(high_boxes[det_idx], high_scores[det_idx],
                                      class_id=high_class_ids[det_idx])
        
        # Manage lost tracks
        self._manage_lost_tracks()
        
        return self._get_active_tracks()
    
    def _match_detections(self, boxes, scores, track_indices=None):
        """Match detections to existing tracks using IoU and Hungarian algorithm"""
        if len(self.tracks) == 0:
            return [], list(range(len(boxes))), []
        
        # Use specified track indices or all tracks
        if track_indices is None:
            track_indices = list(range(len(self.tracks)))
        
        if len(track_indices) == 0:
            return [], list(range(len(boxes))), []
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(boxes, track_indices)
        
        # Apply gating based on IoU threshold
        cost_matrix = 1.0 - iou_matrix
        gated_cost_matrix = np.where(iou_matrix < 0.1, cost_matrix, np.inf)
        
        # Hungarian algorithm for optimal assignment
        try:
            row_indices, col_indices = linear_sum_assignment(gated_cost_matrix)
        except Exception:
            return self._greedy_matching(iou_matrix, boxes, scores, track_indices)
        
        # Filter matches based on threshold
        matches = []
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(range(len(track_indices)))
        
        for r, c in zip(row_indices, col_indices):
            if gated_cost_matrix[r, c] < self.match_thresh or iou_matrix[r, c] > 0.2:
                matches.append((r, c))
                unmatched_dets.discard(c)
                unmatched_trks.discard(r)
        
        # Convert local indices back to global track indices
        global_unmatched_trks = [track_indices[i] for i in unmatched_trks]
        
        return matches, list(unmatched_dets), global_unmatched_trks
    
    def _greedy_matching(self, iou_matrix, boxes, scores, track_indices):
        """Fallback greedy matching when Hungarian algorithm fails"""
        matches = []
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(range(len(track_indices)))
        
        for track_idx in range(len(track_indices)):
            if track_idx in unmatched_trks:
                best_det_idx = None
                best_iou = 0
                
                for det_idx in unmatched_dets:
                    iou = iou_matrix[track_idx, det_idx]
                    if iou > best_iou and iou > 0.3:
                        best_iou = iou
                        best_det_idx = det_idx
                
                if best_det_idx is not None:
                    matches.append((track_idx, best_det_idx))
                    unmatched_dets.discard(best_det_idx)
                    unmatched_trks.discard(track_idx)
        
        global_unmatched_trks = [track_indices[i] for i in unmatched_trks]
        return matches, list(unmatched_dets), global_unmatched_trks
    
    def _calculate_iou_matrix(self, boxes, track_indices=None):
        """Calculate IoU matrix between tracks and detections"""
        if track_indices is None:
            track_indices = range(len(self.tracks))
        
        iou_matrix = np.zeros((len(track_indices), len(boxes)))
        
        for i, track_idx in enumerate(track_indices):
            track_box = self.tracks[track_idx].to_tlbr()
            for j, box in enumerate(boxes):
                iou_matrix[i, j] = self._calculate_iou(track_box, box)
        
        return iou_matrix
    
    def _calculate_iou(self, box1, box2):
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
    
    def _create_new_track(self, box, score, class_id=0):
        """Create a new track"""
        track = STrack(box, score, self.track_id_counter, self.frame_rate, class_id)
        self.tracks.append(track)
        self.track_id_counter += 1
    
    def _manage_lost_tracks(self):
        """Manage lost and removed tracks"""
        active_tracks = []
        lost_tracks = []
        
        for track in self.tracks:
            if track.state == STrack.Confirmed:
                if track.time_since_update <= self.track_buffer:
                    active_tracks.append(track)
                else:
                    lost_tracks.append(track)
            elif track.state == STrack.Tentative:
                if track.time_since_update <= 30:
                    active_tracks.append(track)
                else:
                    lost_tracks.append(track)
        
        self.tracks = active_tracks
        self.lost_tracks.extend(lost_tracks)
        
        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks
                           if t.time_since_update <= self.track_buffer * 5]
    
    def _get_active_tracks(self):
        """Get active tracks for output"""
        active_tracks = []
        for track in self.tracks:
            if track.state == STrack.Confirmed or track.state == STrack.Tentative:
                if track.time_since_update <= 10:
                    active_tracks.append(track)
        return active_tracks


class STrack:
    """Simple Track class for ByteTrack"""
    
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    
    def __init__(self, bbox, score, track_id, frame_rate=30, class_id=0):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.state = STrack.Tentative
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.frame_rate = frame_rate
        
        # Kalman filter for motion prediction
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self._init_kalman_filter()
    
    def _init_kalman_filter(self):
        """Initialize Kalman filter"""
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.kf.Q *= 0.01
        
        # Measurement noise covariance
        self.kf.R *= 10
        
        # Initial state
        cx = (self.bbox[0] + self.bbox[2]) / 2
        cy = (self.bbox[1] + self.bbox[3]) / 2
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        
        self.kf.x[:4] = np.array([cx, cy, w, h]).reshape(4, 1)
    
    def predict(self):
        """Predict next state using Kalman filter"""
        if self.state != STrack.Deleted:
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1
    
    def update(self, bbox, score, class_id=None):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        
        # Update state
        if self.state == STrack.Tentative and self.hits >= 2:
            self.state = STrack.Confirmed
        
        # Update Kalman filter
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.kf.update([cx, cy, w, h])
        
        # Update bbox
        self.bbox = bbox
        self.score = score
        
        # Update class ID if provided
        if class_id is not None:
            self.class_id = class_id
    
    def to_tlbr(self):
        """Convert bounding box to (top, left, bottom, right) format"""
        if hasattr(self.kf, 'x'):
            cx, cy, w, h = self.kf.x[:4].flatten()
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            return [x1, y1, x2, y2]
        else:
            return self.bbox
