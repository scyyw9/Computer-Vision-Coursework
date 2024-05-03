from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()  # Instantiate the Kalman filter
        self.tracks = []   # Store a series of trajectories
        self._next_id = 1  # The next assigned trajectory ID
 
    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run cascade matching
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set
        
        # case 1: For the matched results
        for track_idx, detection_idx in matches:
            # Update the corresponding detections in the tracks
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        # case 2: For the tracks that did not match, call mark_missed to mark them
        # If a track fails to match and is in the Tentative state, delete it
        # If it has been updated for a long time, also delete it
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # case 3: For the unmatched detections, as the detections have failed to match, initialize them
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # Obtain the latest list of tracks, which are the tracks marked as Confirmed and Tentative
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # Get all the track IDs that are in the Confirmed state
            if not track.is_confirmed():
                continue
            features += track.features  # Add the features of the tracks in the Confirmed state to the features list
            # Get the track ID corresponding to each feature
            targets += [track.track_id for _ in track.features]
            track.features = []
        # Update the feature set used in the distance metric
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # Compute the cost matrix through nearest neighbor calculation
            cost_matrix = self.metric.distance(features, targets)
            # Compute the cost matrix after gating
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features
        # Perform cascading matching for the confirmed tracks, and obtain the matched tracks, unmatched tracks,
        # and unmatched detections
        # The matching_cascade matches the detection boxes to the confirmed tracks based on the features
        # The gated cost matrix is passed in
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.        
        # Combine the unconfirmed tracks and the tracks that just did not match into iou_track_candidates
        # And perform IOU-based matching
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # The tracks that just did not match
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]  # Not the tracks that just did not match
        # Perform IOU matching again for the targets that did not match successfully in the cascading matching
        # min_cost_matching uses the Hungarian algorithm to solve the linear assignment problem
        # Pass in iou_cost and try to associate the remaining tracks with the unconfirmed tracks
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b  # Combine the two parts of the matching
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
