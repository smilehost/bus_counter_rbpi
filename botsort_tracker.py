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
        
        # Debug settings
        self.debug_frame_interval = getattr(config, 'DEBUG_FRAME_INTERVAL', 10)
        
    def update(self, detections, frame):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Log frame info only at debug intervals
        should_debug = self.frame_count % self.debug_frame_interval == 0
        if should_debug:
            print(f"\n=== FRAME {self.frame_count} ===")
            print(f"Active tracks before update: {len(self.tracks)}")
            track_ids_before = [t.track_id for t in self.tracks]
            print(f"Track IDs before update: {track_ids_before}")
            
            # DEBUG: Log track states and time_since_update
            for track in self.tracks:
                print(f"  Track {track.track_id}: state={track.state}, time_since_update={track.time_since_update}, hits={track.hits}, age={track.age}")
        
        # Convert detections to format
        if len(detections) == 0:
            if should_debug:
                print("No detections in this frame - checking for ghost tracks")
                # DEBUG: Check which tracks might be ghosting
                for track in self.tracks:
                    if track.time_since_update > 5:  # Tracks not updated for >5 frames
                        print(f"  POTENTIAL GHOST: Track {track.track_id} not updated for {track.time_since_update} frames")
                        print(f"    Last known bbox: {track.bbox}")
                        print(f"    Predicted bbox: {track.to_tlbr()}")
            self._manage_lost_tracks(should_debug)
            return self._get_active_tracks()
        
        if should_debug:
            print(f"Detections: {len(detections)}")
        
        # Extract detection information
        boxes = detections[:, :4]  # x1, y1, x2, y2
        scores = detections[:, 4] if detections.shape[1] > 4 else np.ones(len(detections))
        class_ids = detections[:, 5] if detections.shape[1] > 5 else np.zeros(len(detections))
        features = detections[:, 6:] if detections.shape[1] > 6 else None
        
        # Predict track positions
        self._predict_tracks()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_dets, unmatched_trks = self._match_detections(
            boxes, scores, features, should_debug
        )
        
        if should_debug:
            print(f"Matched tracks: {len(matched_tracks)}")
            print(f"Unmatched detections: {len(unmatched_dets)}")
            print(f"Unmatched tracks: {len(unmatched_trks)}")
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            track_id = self.tracks[track_idx].track_id
            if should_debug:
                print(f"Updating track {track_id} with detection {det_idx}")
            self.tracks[track_idx].update(boxes[det_idx], scores[det_idx],
                                        features[det_idx] if features is not None else None,
                                        class_ids[det_idx], should_debug)
        
        # Create new tracks for unmatched detections
        new_tracks_created = 0
        for det_idx in unmatched_dets:
            if scores[det_idx] > self.new_track_thresh:
                # Check if this detection overlaps significantly with existing tracks
                det_box = boxes[det_idx]
                should_create = True
                max_iou = 0
                
                for track in self.tracks:
                    if track.state == Track.Confirmed or track.state == Track.Tentative:
                        track_box = track.to_tlbr()
                        iou = self._calculate_iou(det_box, track_box)
                        max_iou = max(max_iou, iou)
                        if iou > 0.2:  # Even lower overlap threshold to prevent duplicate tracks
                            should_create = False
                            if should_debug:
                                print(f"  Detection {det_idx} overlaps with track {track.track_id} (IoU: {iou:.2f}), NOT creating new track")
                            break
                
                if should_create:
                    if should_debug:
                        print(f"  Creating new track for detection {det_idx} (score: {scores[det_idx]:.2f}, max IoU: {max_iou:.2f})")
                    self._create_new_track(boxes[det_idx], scores[det_idx],
                                         features[det_idx] if features is not None else None,
                                         class_ids[det_idx])
                    new_tracks_created += 1
        
        if should_debug:
            print(f"New tracks created: {new_tracks_created}")
        
        # Manage lost tracks
        self._manage_lost_tracks(should_debug)
        
        # Log final state
        active_tracks = self._get_active_tracks()
        if should_debug:
            track_ids_after = [t.track_id for t in active_tracks]
            print(f"Active tracks after update: {len(active_tracks)}")
            print(f"Track IDs after update: {track_ids_after}")
        
        return active_tracks
    
    def _predict_tracks(self):
        """Predict next positions for all tracks"""
        for track in self.tracks:
            track.predict()
    
    def _match_detections(self, boxes, scores, features, should_debug=True):
        """Match detections to existing tracks using Hungarian algorithm"""
        if len(self.tracks) == 0:
            if should_debug:
                print("    No tracks to match with")
            return [], list(range(len(boxes))), []
        
        if should_debug:
            print(f"    Matching {len(boxes)} detections with {len(self.tracks)} tracks")
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(boxes)
        
        # Calculate appearance similarity matrix if ReID is enabled
        if self.with_reid and features is not None:
            appearance_matrix = self._calculate_appearance_matrix(features)
            # Combine IoU and appearance scores with more weight on appearance
            cost_matrix = 1.0 - (0.4 * iou_matrix + 0.6 * appearance_matrix)
        else:
            cost_matrix = 1.0 - iou_matrix
        
        # Apply gating based on IoU threshold - be more permissive for matching
        gated_cost_matrix = np.where(iou_matrix < self.proximity_thresh,
                                    cost_matrix, np.inf)
        
        # Hungarian algorithm for optimal assignment
        try:
            row_indices, col_indices = linear_sum_assignment(gated_cost_matrix)
        except Exception as e:
            if should_debug:
                print(f"    Error in linear sum assignment: {e}")
                print(f"    Cost matrix shape: {gated_cost_matrix.shape}")
                print(f"    Cost matrix contains NaN: {np.isnan(gated_cost_matrix).any()}")
                print(f"    Cost matrix contains Inf: {np.isinf(gated_cost_matrix).any()}")
            # Fall back to greedy matching
            return self._greedy_matching(iou_matrix, boxes, scores, should_debug)
        
        # Filter matches based on threshold - be more permissive
        matches = []
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(range(len(self.tracks)))
        
        if should_debug:
            print(f"    Initial assignments: {len(row_indices)}")
        
        for r, c in zip(row_indices, col_indices):
            # Much more lenient matching criteria to prevent track loss
            if gated_cost_matrix[r, c] < self.match_thresh or iou_matrix[r, c] > 0.2:
                track_id = self.tracks[r].track_id
                if should_debug:
                    print(f"    Matched track {track_id} (idx {r}) with detection {c} (IoU: {iou_matrix[r, c]:.2f}, Cost: {gated_cost_matrix[r, c]:.2f})")
                    # DEBUG: Log track state before matching
                    track = self.tracks[r]
                    print(f"      Track {track_id} before update: time_since_update={track.time_since_update}, state={track.state}")
                matches.append((r, c))
                unmatched_dets.discard(c)
                unmatched_trks.discard(r)
            else:
                if should_debug:
                    print(f"    Rejected match track {r} with detection {c} (IoU: {iou_matrix[r, c]:.2f}, Cost: {gated_cost_matrix[r, c]:.2f})")
                    # DEBUG: Log why track wasn't matched
                    track = self.tracks[r]
                    print(f"      Unmatched track {track.track_id}: time_since_update={track.time_since_update}, state={track.state}")
        
        return matches, list(unmatched_dets), list(unmatched_trks)
    
    def _greedy_matching(self, iou_matrix, boxes, scores, should_debug=True):
        """Fallback greedy matching when Hungarian algorithm fails"""
        if should_debug:
            print("    Using greedy matching as fallback")
        matches = []
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(range(len(self.tracks)))
        
        # For each track, find the best detection
        for track_idx in range(len(self.tracks)):
            if track_idx in unmatched_trks:
                best_det_idx = None
                best_iou = 0
                
                for det_idx in unmatched_dets:
                    iou = iou_matrix[track_idx, det_idx]
                    if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                        best_iou = iou
                        best_det_idx = det_idx
                
                if best_det_idx is not None:
                    track_id = self.tracks[track_idx].track_id
                    if should_debug:
                        print(f"    Greedy matched track {track_id} (idx {track_idx}) with detection {best_det_idx} (IoU: {best_iou:.2f})")
                    matches.append((track_idx, best_det_idx))
                    unmatched_dets.discard(best_det_idx)
                    unmatched_trks.discard(track_idx)
        
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
    
    def _manage_lost_tracks(self, should_debug=True):
        """Manage lost and removed tracks"""
        active_tracks = []
        lost_tracks = []
        
        if should_debug:
            print(f"  Managing lost tracks ({len(self.tracks)} total)")
        
        for track in self.tracks:
            if track.state == Track.Confirmed:
                # DEBUG: More aggressive track removal to prevent ghosting
                # Reduced from track_buffer * 5 to track_buffer * 2
                if track.time_since_update <= self.track_buffer * 2:  # Reduced buffer
                    active_tracks.append(track)
                    if should_debug and track.time_since_update > 5:
                        print(f"    WARNING: Track {track.track_id} active but not updated for {track.time_since_update} frames (GHOSTING RISK)")
                else:
                    if should_debug:
                        print(f"    Moving confirmed track {track.track_id} to lost (time_since_update: {track.time_since_update} > {self.track_buffer * 2})")
                    lost_tracks.append(track)
            elif track.state == Track.Tentative:
                # DEBUG: More aggressive removal of tentative tracks
                # Reduced from 30 to 15 frames
                if track.time_since_update <= 15:  # Reduced buffer
                    active_tracks.append(track)
                    if should_debug and track.time_since_update > 5:
                        print(f"    WARNING: Tentative track {track.track_id} active but not updated for {track.time_since_update} frames")
                else:
                    if should_debug:
                        print(f"    Moving tentative track {track.track_id} to lost (time_since_update: {track.time_since_update} > 15)")
                    lost_tracks.append(track)
        
        self.tracks = active_tracks
        self.lost_tracks.extend(lost_tracks)
        
        # Remove old lost tracks
        old_lost_count = len(self.lost_tracks)
        self.lost_tracks = [t for t in self.lost_tracks
                           if t.time_since_update <= self.track_buffer * 5]  # Reduced from 10 to 5
        if should_debug:
            print(f"    Removed {old_lost_count - len(self.lost_tracks)} old lost tracks")
            print(f"    Active tracks after management: {len(self.tracks)}")
    
    def _get_active_tracks(self):
        """Get active tracks for output"""
        active_tracks = []
        for track in self.tracks:
            # Include both Confirmed and Tentative tracks that are still active
            # DEBUG: Add additional filtering to prevent ghosting
            if (track.state == Track.Confirmed or track.state == Track.Tentative):
                # DEBUG: Don't include tracks that haven't been updated for too long
                if track.time_since_update <= 10:  # Hard limit to prevent ghosting
                    active_tracks.append(track)
                else:
                    print(f"    FILTERING OUT: Track {track.track_id} (time_since_update: {track.time_since_update})")
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
            # DEBUG: More aggressive logging for ghosting detection
            if self.time_since_update % 3 == 0:  # Log more frequently
                print(f"      Track {self.track_id} predicted: time_since_update={self.time_since_update}, age={self.age}")
            
            # DEBUG: Mark tracks as potential ghosts if not updated for too long
            if self.time_since_update > 8:  # Earlier warning
                print(f"      WARNING: Track {self.track_id} potential ghost - not updated for {self.time_since_update} frames")
    
    def update(self, bbox, score, feature=None, class_id=None, should_debug=True):
        """Update track with new detection"""
        old_state = self.state
        old_time_since_update = self.time_since_update  # DEBUG: Track how long it was ghosting
        
        self.time_since_update = 0
        self.hits += 1
        
        # DEBUG: Log recovery from ghosting
        if old_time_since_update > 5:
            print(f"      RECOVERY: Track {self.track_id} recovered after {old_time_since_update} frames of ghosting")
        
        # Update state - confirm after first successful update to prevent ID changes
        if self.state == Track.Tentative and self.hits >= 1:  # Changed from 2 to 1
            self.state = Track.Confirmed
            if should_debug:
                print(f"      Track {self.track_id} confirmed (hits: {self.hits})")
        
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
        
        if should_debug:
            print(f"      Updated track {self.track_id}: hits={self.hits}, age={self.age}, state={old_state}->{self.state}")
    
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