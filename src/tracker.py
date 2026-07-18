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

        # Weighted linear regression (recent frames weighted higher)
        weights = np.exp(np.linspace(-2, 0, len(times)))  # exponential weighting
        weights /= weights.sum()

        # Fit: center = v * t + c0  per axis
        t_mean = (times * weights).sum()
        cx_mean = (centers[:, 0] * weights).sum()
        cy_mean = (centers[:, 1] * weights).sum()

        t_var = (weights * (times - t_mean)**2).sum() + 1e-8
        vx = (weights * (times - t_mean) * (centers[:, 0] - cx_mean)).sum() / t_var
        vy = (weights * (times - t_mean) * (centers[:, 1] - cy_mean)).sum() / t_var

        self.fitted_v = math.sqrt(vx**2 + vy**2)

        # Yaw from velocity direction: atan2(vy, vx)
        if self.fitted_v > 0.3:  # only update yaw if moving
            self.fitted_yaw = math.atan2(vy, vx)
            if self.fitted_yaw < 0:
                self.fitted_yaw += math.pi  # [0, π)

        # Acceleration from velocity change (if enough history)
        if len(self.history) >= 4 and dt > 0.3:
            # Compute velocities in two halves
            mid = len(times) // 2
            t1 = times[:mid]; c1 = centers[:mid]
            t2 = times[mid:]; c2 = centers[mid:]
            w1 = np.exp(np.linspace(-2, 0, len(t1))); w1 /= w1.sum()
            w2 = np.exp(np.linspace(-2, 0, len(t2))); w2 /= w2.sum()

            def fit_vel(t, c, w):
                tm = (t * w).sum()
                cm0 = (c[:, 0] * w).sum()
                cm1 = (c[:, 1] * w).sum()
                tv = (w * (t - tm)**2).sum() + 1e-8
                v0 = (w * (t - tm) * (c[:, 0] - cm0)).sum() / tv
                v1 = (w * (t - tm) * (c[:, 1] - cm1)).sum() / tv
                return np.array([v0, v1])

            v1 = fit_vel(t1, c1, w1)
            v2 = fit_vel(t2, c2, w2)
            t_mid = (t2.mean() - t1.mean())
            self.fitted_a = np.linalg.norm(v2 - v1) / (t_mid + 1e-8)


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
                    'yaw': t.fitted_yaw,          # motion yaw (velocity direction)
                    'model_yaw': t.model_yaw,      # geometric yaw from model
                    'v': t.fitted_v,              # m/s
                    'a': t.fitted_a,              # m/s²
                    'class_id': t.class_id,
                    'class_name': t.class_name,
                    'history_len': len(t.history),
                })
        return results
