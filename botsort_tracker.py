import numpy as np
from collections import defaultdict, deque
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from lap import lapjv
import torch

class BoTSORT:
    """
    BoTSORT (Bag of Tricks for Object Tracking) implementation
    Combines multiple tracking techniques for robust multi-object tracking
    """
    
    def __init__(self, config):
        self.config = config
        self.tracks = []
        self.track_id_counter = 0
        self.frame_count = 0
        
        # Tracking parameters
        self.track_high_thresh = config['track_high_thresh']
        self.track_low_thresh = config['track_low_thresh']
        self.new_track_thresh = config['new_track_thresh']
        self.track_buffer = config['track_buffer']
        self.match_thresh = config['match_thresh']
        self.proximity_thresh = config['proximity_thresh']
        self.appearance_thresh = config['appearance_thresh']
        self.with_reid = config['with_reid']
        self.frame_rate = config['frame_rate']
        self.fuse_score = config['fuse_score']
        
        # Track history for management
        self.lost_tracks = []
        self.removed_tracks = []
        
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Convert detections to format
        if len(detections) == 0:
            self._manage_lost_tracks()
            return self._get_active_tracks()
        
        # Extract detection information
        boxes = detections[:, :4]  # x1, y1, x2, y2
        scores = detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections))
        class_ids = detections[:, 5] if detections.shape[1] > 5 else np.zeros(len(detections))
        features = detections[:, 6:] if detections.shape[1] > 6 else None
        
        # Predict track positions
        self._predict_tracks()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_dets, unmatched_trks = self._match_detections(
            boxes, scores, features
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(boxes[det_idx], scores[det_idx],
                                        features[det_idx] if features is not None else None,
                                        class_ids[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            if scores[det_idx] > self.new_track_thresh:
                self._create_new_track(boxes[det_idx], scores[det_idx],
                                     features[det_idx] if features is not None else None,
                                     class_ids[det_idx])
        
        # Manage lost tracks
        self._manage_lost_tracks()
        
        return self._get_active_tracks()
    
    def _predict_tracks(self):
        """Predict next positions for all tracks"""
        for track in self.tracks:
            track.predict()
    
    def _match_detections(self, boxes, scores, features):
        """Match detections to existing tracks using Hungarian algorithm"""
        if len(self.tracks) == 0:
            return [], list(range(len(boxes))), []
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(boxes)
        
        # Calculate appearance similarity matrix if ReID is enabled
        if self.with_reid and features is not None:
            appearance_matrix = self._calculate_appearance_matrix(features)
            # Combine IoU and appearance scores
            cost_matrix = 1.0 - (0.6 * iou_matrix + 0.4 * appearance_matrix)
        else:
            cost_matrix = 1.0 - iou_matrix
        
        # Apply gating based on IoU threshold
        gated_cost_matrix = np.where(iou_matrix < self.proximity_thresh, 
                                    cost_matrix, np.inf)
        
        # Hungarian algorithm for optimal assignment
        try:
            row_indices, col_indices = linear_sum_assignment(gated_cost_matrix)
        except:
            return [], list(range(len(boxes))), []
        
        # Filter matches based on threshold
        matches = []
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(range(len(self.tracks)))
        
        for r, c in zip(row_indices, col_indices):
            if gated_cost_matrix[r, c] < self.match_thresh:
                matches.append((r, c))
                unmatched_dets.discard(c)
                unmatched_trks.discard(r)
        
        return matches, list(unmatched_dets), list(unmatched_trks)
    
    def _calculate_iou_matrix(self, boxes):
        """Calculate IoU matrix between tracks and detections"""
        iou_matrix = np.zeros((len(self.tracks), len(boxes)))
        
        for i, track in enumerate(self.tracks):
            track_box = track.to_tlbr()
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
    
    def _calculate_appearance_matrix(self, features):
        """Calculate appearance similarity matrix"""
        if not self.with_reid or len(self.tracks) == 0:
            return np.zeros((len(self.tracks), len(features)))
        
        appearance_matrix = np.zeros((len(self.tracks), len(features)))
        
        for i, track in enumerate(self.tracks):
            if track.features is not None:
                track_feat = track.get_feature()
                for j, det_feat in enumerate(features):
                    # Cosine similarity
                    similarity = np.dot(track_feat, det_feat) / (
                        np.linalg.norm(track_feat) * np.linalg.norm(det_feat) + 1e-8
                    )
                    appearance_matrix[i, j] = max(0, similarity)
        
        return appearance_matrix
    
    def _create_new_track(self, box, score, feature=None, class_id=0):
        """Create a new track"""
        track = Track(box, score, self.track_id_counter, feature, self.frame_rate, class_id)
        self.tracks.append(track)
        self.track_id_counter += 1
    
    def _manage_lost_tracks(self):
        """Manage lost and removed tracks"""
        active_tracks = []
        lost_tracks = []
        
        for track in self.tracks:
            if track.state == Track.Confirmed:
                if track.time_since_update <= self.track_buffer:
                    active_tracks.append(track)
                else:
                    lost_tracks.append(track)
            elif track.state == Track.Tentative:
                if track.time_since_update <= 1:
                    active_tracks.append(track)
                else:
                    lost_tracks.append(track)
        
        self.tracks = active_tracks
        self.lost_tracks.extend(lost_tracks)
        
        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks 
                           if t.time_since_update <= self.track_buffer * 2]
    
    def _get_active_tracks(self):
        """Get active tracks for output"""
        active_tracks = []
        for track in self.tracks:
            if track.state == Track.Confirmed and track.time_since_update == 0:
                active_tracks.append(track)
        return active_tracks


class Track:
    """Individual track class"""
    
    Tentative = 1
    Confirmed = 2
    Deleted = 3
    
    def __init__(self, bbox, score, track_id, feature=None, frame_rate=30, class_id=0):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.state = Track.Tentative
        self.features = deque(maxlen=100)
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.frame_rate = frame_rate
        
        # Kalman filter for motion prediction
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self._init_kalman_filter()
        
        if feature is not None:
            self.features.append(feature)
    
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
        if self.state != Track.Deleted:
            self.kf.predict()
            self.age += 1
            self.time_since_update += 1
    
    def update(self, bbox, score, feature=None, class_id=None):
        """Update track with new detection"""
        self.time_since_update = 0
        self.hits += 1
        
        # Update state
        if self.state == Track.Tentative and self.hits >= 3:
            self.state = Track.Confirmed
        
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
        
        # Update features
        if feature is not None:
            self.features.append(feature)
    
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
    
    def get_feature(self):
        """Get average feature from feature history"""
        if len(self.features) > 0:
            return np.mean(self.features, axis=0)
        return None