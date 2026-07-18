"""Multi-object tracker with CV/CA motion model fitting.

Tracks detected 3D objects across frames using Hungarian association
and fits constant-velocity / constant-acceleration models to estimate
yaw (from velocity), speed, and acceleration for each track.
"""

import math
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


class Track:
    """Single tracked object with position history."""

    def __init__(self, track_id, center, size, yaw, class_id, timestamp, class_name='?'):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.model_yaw = yaw       # geometric yaw from model (primary)
        self.history = [(timestamp, center.copy(), size.copy(), yaw)]
        self.fitted_yaw = yaw      # yaw from velocity direction
        self.fitted_vx = 0.0       # velocity x (m/s)
        self.fitted_vy = 0.0       # velocity y (m/s)
        self.fitted_v = 0.0        # speed (m/s)
        self.fitted_a = 0.0        # acceleration (m/s²)
        self.alive = True
        self.missed_frames = 0

    def update(self, timestamp, center, size, yaw):
        self.model_yaw = yaw  # keep latest model yaw
        self.history.append((timestamp, center.copy(), size.copy(), yaw))
        self.missed_frames = 0
        if len(self.history) > 30:
            self.history = self.history[-30:]

    def mark_missed(self):
        self.missed_frames += 1
        if self.missed_frames > 5:
            self.alive = False

    def fit_model(self):
        """Fit CV model to position history → yaw, velocity, acceleration."""
        if len(self.history) < 2:
            return
        times = np.array([t for t, _, _, _ in self.history]) * 1e-6  # µs → s
        centers = np.array([c[:2] for _, c, _, _ in self.history])    # (N, 2) XY

        # Use recent history (last 10 frames or 2 seconds)
        if len(times) > 10:
            times = times[-10:]
            centers = centers[-10:]

        dt = times[-1] - times[0]
        if dt < 0.05:
            return

        # Simple displacement-based velocity: (last_pos - first_pos) / dt
        # More robust to noise than regression for short windows
        first_center = centers[0]
        last_center = centers[-1]
        vx = (last_center[0] - first_center[0]) / dt
        vy = (last_center[1] - first_center[1]) / dt
        self.fitted_vx = vx
        self.fitted_vy = vy
        self.fitted_v = math.sqrt(vx**2 + vy**2)

        if self.fitted_v > 0.3:
            self.fitted_yaw = math.atan2(vy, vx)
            # Resolve model yaw 180° ambiguity: flip if closer to velocity direction
            yaw_diff = abs(self.fitted_yaw - self.model_yaw)
            yaw_diff_flipped = abs(self.fitted_yaw - (self.model_yaw + math.pi) % (2*math.pi))
            if yaw_diff_flipped < yaw_diff:
                self.model_yaw = (self.model_yaw + math.pi) % (2*math.pi)

        # Acceleration from velocity change over two halves
        if len(self.history) >= 4 and dt > 0.3:
            mid = len(times) // 2
            dt1 = times[mid-1] - times[0] + 1e-8
            dt2 = times[-1] - times[mid] + 1e-8
            v1x = (centers[mid-1, 0] - centers[0, 0]) / dt1
            v1y = (centers[mid-1, 1] - centers[0, 1]) / dt1
            v2x = (centers[-1, 0] - centers[mid, 0]) / dt2
            v2y = (centers[-1, 1] - centers[mid, 1]) / dt2
            dt_mid = (times[-1] - times[0]) / 2
            self.fitted_a = math.sqrt((v2x-v1x)**2 + (v2y-v1y)**2) / (dt_mid + 1e-8)


class Tracker:
    """Multi-object tracker using Hungarian association + CV model."""

    def __init__(self, max_dist=5.0, min_history=3):
        self.tracks = {}       # track_id → Track
        self.next_id = 0
        self.max_dist = max_dist
        self.min_history = min_history  # min frames before outputting motion

    def update(self, detections, timestamp):
        """
        Args:
            detections: list[dict] with 'center', 'size', 'yaw', 'class_id', 'class_name'
            timestamp: int (microseconds)
        Returns:
            list[dict]: updated tracks with fitted motion {track_id, center, size, yaw, v, a, class_id, class_name}
        """
        # Predict track positions (simple: last known position)
        active_tracks = [t for t in self.tracks.values() if t.alive]

        if not active_tracks and not detections:
            return []

        if not active_tracks:
            for d in detections:
                t = Track(self.next_id, d['center'], d['size'], d['yaw'],
                          d['class_id'], timestamp, d.get('class_name', '?'))
                self.tracks[self.next_id] = t
                self.next_id += 1
            return self._get_outputs()

        if not detections:
            for t in active_tracks:
                t.mark_missed()
            return self._get_outputs()

        # Build cost matrix (Hungarian)
        n_tracks = len(active_tracks)
        n_dets = len(detections)
        cost = np.zeros((n_tracks, n_dets))

        for i, t in enumerate(active_tracks):
            t_center = t.history[-1][1][:2]
            t_class = t.class_id
            for j, d in enumerate(detections):
                d_center = d['center'][:2]
                dist = np.linalg.norm(t_center - d_center)
                # Class mismatch penalty
                cls_penalty = 0.0 if d['class_id'] == t_class else 10.0
                cost[i, j] = dist + cls_penalty

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < self.max_dist:
                t = active_tracks[r]
                d = detections[c]
                t.update(timestamp, d['center'], d['size'], d['yaw'])
                assigned_tracks.add(t.track_id)
                assigned_dets.add(c)

        # Mark unmatched tracks as missed
        for t in active_tracks:
            if t.track_id not in assigned_tracks:
                t.mark_missed()

        # Create new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j not in assigned_dets:
                t = Track(self.next_id, d['center'], d['size'], d['yaw'],
                          d['class_id'], timestamp, d.get('class_name', '?'))
                self.tracks[self.next_id] = t
                self.next_id += 1

        # Fit models for all active tracks
        for t in self.tracks.values():
            if t.alive:
                t.fit_model()

        # Cleanup dead tracks
        dead = [tid for tid, t in self.tracks.items() if not t.alive]
        for tid in dead:
            del self.tracks[tid]

        return self._get_outputs()

    def _get_outputs(self):
        """Return list of active tracks with fitted motion."""
        results = []
        for t in self.tracks.values():
            if t.alive and len(t.history) >= self.min_history:
                last_center = t.history[-1][1]
                last_size = t.history[-1][2]
                results.append({
                    'track_id': t.track_id,
                    'center': last_center,
                    'size': last_size,
                    'yaw': t.fitted_yaw,          # motion yaw
                    'model_yaw': t.model_yaw,      # geometric yaw from model
                    'v': t.fitted_v,              # m/s
                    'vx': t.fitted_vx,            # m/s in LiDAR x
                    'vy': t.fitted_vy,            # m/s in LiDAR y
                    'a': t.fitted_a,              # m/s²
                    'class_id': t.class_id,
                    'class_name': t.class_name,
                    'history_len': len(t.history),
                })
        return results
