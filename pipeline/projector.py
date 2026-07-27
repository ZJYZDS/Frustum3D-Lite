"""LiDAR→Camera projection and quaternion utilities for nuScenes calibration."""

import json
import math
import os
from pathlib import Path

import numpy as np


def quaternion_to_mat(qw, qx, qy, qz):
    """Quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2],
    ], dtype=np.float32)


def quaternion_to_yaw(qw, qx, qy, qz):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotate_points_z(points, angle):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
    return points @ rot.T


class LiDARProjector:
    """Project LiDAR points to camera image plane using nuScenes calibration."""

    ALL_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self._load_calibrations()

    def _load_json(self, name):
        with open(os.path.join(self.data_root, "v1.0-mini", name)) as f:
            return json.load(f)

    def _load_calibrations(self):
        calib_list = self._load_json("calibrated_sensor.json")
        sensors = self._load_json("sensor.json")

        self.cam_front_token = None
        self.lidar_token = None
        for s in sensors:
            if s["channel"] == "CAM_FRONT":
                self.cam_front_token = s["token"]
            elif s["channel"] == "LIDAR_TOP":
                self.lidar_token = s["token"]

        self.calibs = {c["token"]: c for c in calib_list}

        sd_list = self._load_json("sample_data.json")
        self._sample_sensor_calib = {}
        for sd in sd_list:
            self._sample_sensor_calib.setdefault(sd["sample_token"], {})[sd["filename"].split("/")[1]] \
            = sd["calibrated_sensor_token"]

    def get_transform(self, sample_token, camera='CAM_FRONT'):
        """Get LiDAR-to-camera projection for a given sample + camera.

        Returns:
            K: (3, 3) camera intrinsic matrix
            T_lidar2cam: (3, 4) [R|t] from LiDAR to camera frame
            img_shape: (H, W)
        """
        cam_calib_token = self._sample_sensor_calib.get(sample_token, {}).get(camera)
        lidar_calib_token = self._sample_sensor_calib.get(sample_token, {}).get("LIDAR_TOP")

        if cam_calib_token is None or lidar_calib_token is None:
            return None, None, None

        cam_calib = self.calibs[cam_calib_token]
        lidar_calib = self.calibs[lidar_calib_token]

        K = np.array(cam_calib["camera_intrinsic"], dtype=np.float32)

        R_lidar = quaternion_to_mat(*lidar_calib["rotation"])
        t_lidar = np.array(lidar_calib["translation"], dtype=np.float32)

        R_cam = quaternion_to_mat(*cam_calib["rotation"])
        t_cam = np.array(cam_calib["translation"], dtype=np.float32)

        R = R_cam.T @ R_lidar
        t = R_cam.T @ (t_lidar - t_cam)
        T_lidar2cam = np.hstack([R, t.reshape(3, 1)]).astype(np.float32)

        img_shape = (900, 1600)

        return K, T_lidar2cam, img_shape

    def project(self, points_lidar, K, T_lidar2cam, img_shape):
        """Project LiDAR points to image pixel coordinates."""
        H, W = img_shape
        xyz = points_lidar[:, :3]
        pts_cam = (T_lidar2cam[:3, :3] @ xyz.T + T_lidar2cam[:3, 3:4]).T
        uv_hom = K @ pts_cam.T
        u = uv_hom[0] / uv_hom[2]
        v = uv_hom[1] / uv_hom[2]
        depth = pts_cam[:, 2]
        valid = (uv_hom[2] > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return np.stack([u, v], axis=1).astype(np.float32), depth, valid
