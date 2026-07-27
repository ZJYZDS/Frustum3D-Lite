"""Phase 1 dataset: YOLOv8-seg ONNX detections + LiDAR→image projection.

Replaces GT-bbox-based point extraction with real YOLO 2D detections,
projecting LiDAR to image plane and associating detections with GT 3D annotations.
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pipeline.detector import YOLOSegONNX, OBSTACLE_CLASS_IDS


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

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self._load_calibrations()

    def _load_json(self, name):
        with open(os.path.join(self.data_root, "v1.0-mini", name)) as f:
            return json.load(f)

    def _load_calibrations(self):
        calib_list = self._load_json("calibrated_sensor.json")
        sensors = self._load_json("sensor.json")

        # Find sensor tokens
        self.cam_front_token = None
        self.lidar_token = None
        for s in sensors:
            if s["channel"] == "CAM_FRONT":
                self.cam_front_token = s["token"]
            elif s["channel"] == "LIDAR_TOP":
                self.lidar_token = s["token"]

        # Build calibration dict
        self.calibs = {c["token"]: c for c in calib_list}

        # Build sensor -> calibrated_sensor_token lookup from sample_data
        sd_list = self._load_json("sample_data.json")
        self._sample_sensor_calib = {}
        for sd in sd_list:
            key = (sd["sample_token"],)  # will be matched later
            self._sample_sensor_calib.setdefault(sd["sample_token"], {})[sd["filename"].split("/")[1]] \
            = sd["calibrated_sensor_token"]

    ALL_CAMERAS = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']

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
            # Try to find closest keyframe
            return None, None, None

        cam_calib = self.calibs[cam_calib_token]
        lidar_calib = self.calibs[lidar_calib_token]

        # Camera intrinsics
        K = np.array(cam_calib["camera_intrinsic"], dtype=np.float32)

        # LiDAR -> ego: R_lidar @ pt + t_lidar
        R_lidar = quaternion_to_mat(*lidar_calib["rotation"])
        t_lidar = np.array(lidar_calib["translation"], dtype=np.float32)

        # Camera -> ego: R_cam @ pt + t_cam  =>  ego -> camera: R_cam^T @ (pt_ego - t_cam)
        R_cam = quaternion_to_mat(*cam_calib["rotation"])
        t_cam = np.array(cam_calib["translation"], dtype=np.float32)

        # LiDAR -> camera: pt_cam = R_cam^T @ (R_lidar @ pt_lidar + t_lidar - t_cam)
        R = R_cam.T @ R_lidar
        t = R_cam.T @ (t_lidar - t_cam)
        T_lidar2cam = np.hstack([R, t.reshape(3, 1)]).astype(np.float32)

        # Image dimensions (nuScenes CAM_FRONT)
        img_shape = (900, 1600)

        return K, T_lidar2cam, img_shape

    def project(self, points_lidar, K, T_lidar2cam, img_shape):
        """Project LiDAR points to image pixel coordinates.

        Args:
            points_lidar: (N, 3+) xyz in LiDAR frame
            K: (3, 3) intrinsics
            T_lidar2cam: (3, 4) [R|t]
            img_shape: (H, W)

        Returns:
            uv: (N, 2) pixel coordinates
            depth: (N,) depth in camera frame (>0 = in front)
            valid: (N,) boolean mask
        """
        H, W = img_shape
        xyz = points_lidar[:, :3]
        pts_cam = (T_lidar2cam[:3, :3] @ xyz.T + T_lidar2cam[:3, 3:4]).T   # (N, 3)
        uv_hom = K @ pts_cam.T                                               # (3, N)
        u = uv_hom[0] / uv_hom[2]
        v = uv_hom[1] / uv_hom[2]
        depth = pts_cam[:, 2]
        valid = (uv_hom[2] > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return np.stack([u, v], axis=1).astype(np.float32), depth, valid


class Phase1Dataset(Dataset):
    """Phase 1 dataset: YOLO detections + LiDAR projection + GT association.

    Each sample is extracted from a YOLO detection:
      - LiDAR points within the 2D detection bbox (projected to image)
      - Target: residual from noisy 3D bbox to matched GT annotation
    """

    MIN_LIDAR_PTS = 10     # lower threshold for sparse LiDAR
    NUM_POINTS = 256
    BBOX_MARGIN = 0.3
    NOISE_CENTER = 0.3
    NOISE_SIZE = 0.15
    NOISE_YAW_DEG = 5.0

    def __init__(self, data_root, split="train", cfg=None,
                 detector_path="models/yolov8s-seg.onnx"):
        self.data_root = Path(data_root)
        self.split = split
        cfg = cfg or {}

        self.num_points = cfg.get("num_points", self.NUM_POINTS)
        self.bbox_margin = cfg.get("bbox_margin", self.BBOX_MARGIN)
        val_scene_ids = cfg.get("val_scene_ids", 2)

        print(f"[Phase1] Loading metadata...")
        self._load_metadata(data_root)

        print(f"[Phase1] Building sensor lookup...")
        self._build_sensor_lookup()

        print(f"[Phase1] Loading detector: {detector_path}")
        self.detector = YOLOSegONNX(detector_path, conf_thresh=0.5, iou_thresh=0.65)

        print(f"[Phase1] Loading projector...")
        self.projector = LiDARProjector(data_root)

        self._build_mappings()
        self._build_sample_list(val_scene_ids)

    def _load_json(self, name):
        with open(os.path.join(self.data_root, "v1.0-mini", name)) as f:
            return json.load(f)

    def _load_metadata(self, data_root):
        self._categories = self._load_json("category.json")
        self._scenes = self._load_json("scene.json")
        self._samples = {s["token"]: s for s in self._load_json("sample.json")}
        self._annotations = self._load_json("sample_annotation.json")
        self._instances = {i["token"]: i for i in self._load_json("instance.json")}
        self._sample_data = {s["token"]: s for s in self._load_json("sample_data.json")}

    def _build_mappings(self):
        # Category name -> token
        self._cat_name2token = {c["name"]: c["token"] for c in self._categories}
        # instance -> category_token
        self._inst2cat = {token: inst["category_token"] for token, inst in self._instances.items()}

    def _build_sensor_lookup(self):
        """Build sample_token -> {sensor_name -> sample_data_token} for fast lookup."""
        self._frame_sensors = {}
        for sd_token, sd in self._sample_data.items():
            sensor = sd["filename"].split("/")[1]  # exact sensor name
            sample_token = sd["sample_token"]
            self._frame_sensors.setdefault(sample_token, {})[sensor] = sd_token

    def _build_sample_list(self, val_scene_ids):
        # Group samples by scene
        scene_samples = {}
        for sample_token, sample in self._samples.items():
            scene_samples.setdefault(sample["scene_token"], []).append(sample_token)

        sorted_scenes = sorted(self._scenes, key=lambda s: s["name"])
        if self.split == "train":
            split_scenes = sorted_scenes[:-val_scene_ids] if val_scene_ids > 0 else sorted_scenes
        else:
            split_scenes = sorted_scenes[-val_scene_ids:] if val_scene_ids > 0 else []

        # Build sample list: which frames have CAM_FRONT + LIDAR_TOP data
        self._frame_list = []
        for scene in split_scenes:
            for sample_token in scene_samples.get(scene["token"], []):
                sensors = self._frame_sensors.get(sample_token, {})
                if "CAM_FRONT" in sensors and "LIDAR_TOP" in sensors:
                    self._frame_list.append(sample_token)

        print(f"[Phase1][{self.split}] {len(split_scenes)} scenes, {len(self._frame_list)} frames")

    def __len__(self):
        return len(self._frame_list)

    def __getitem__(self, idx):
        """Returns (detections, matched_annotations) for a single frame.

        This is a frame-level dataset. The training loop handles per-object batching.
        """
        sample_token = self._frame_list[idx]
        sample = self._samples[sample_token]

        # Load image and run YOLO
        img = self._load_image(sample_token)
        if img is None:
            return []

        dets = self.detector.predict(img)
        dets = [d for d in dets if d["class_id"] in OBSTACLE_CLASS_IDS]
        if not dets:
            return []

        # Load LiDAR and project to image
        points = self._load_lidar(sample_token)
        if points is None:
            return []

        K, T_lidar2cam, img_shape = self.projector.get_transform(sample_token)
        if K is None:
            return []

        uv, depth, valid_proj = self.projector.project(points, K, T_lidar2cam, img_shape)

        # Get GT annotations for this frame
        gt_anns = self._get_gt_for_frame(sample_token)

        # For each detection, extract per-object data + find matching GT
        samples = []
        for det in dets:
            bbox = det["bbox"]  # (x1, y1, x2, y2)
            x1, y1, x2, y2 = bbox

            # Find LiDAR points inside the 2D bbox
            in_bbox = (
                valid_proj &
                (uv[:, 0] >= x1) & (uv[:, 0] <= x2) &
                (uv[:, 1] >= y1) & (uv[:, 1] <= y2) &
                (depth > 0.5)
            )
            obj_pts = points[in_bbox]
            if len(obj_pts) < self.MIN_LIDAR_PTS:
                continue

            # Find matching GT annotation (3D distance based)
            matched_gt = self._match_gt(det, gt_anns, uv, depth, valid_proj, K, T_lidar2cam, points)

            if matched_gt is not None:
                point_features, target = self._build_sample(obj_pts, matched_gt)
                samples.append((point_features, target, det["class_id"], det["is_person"]))

        return samples

    def _load_image(self, sample_token):
        sensors = self._frame_sensors.get(sample_token, {})
        sd_token = sensors.get("CAM_FRONT")
        if sd_token is None:
            return None
        path = os.path.join(self.data_root, self._sample_data[sd_token]["filename"])
        return cv2.imread(path)

    def _load_lidar(self, sample_token):
        sensors = self._frame_sensors.get(sample_token, {})
        sd_token = sensors.get("LIDAR_TOP")
        if sd_token is None:
            return None
        path = os.path.join(self.data_root, self._sample_data[sd_token]["filename"])
        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)

    def _get_gt_for_frame(self, sample_token):
        """Get GT annotations relevant to this frame."""
        # nuScenes links annotations to samples
        anns = []
        for ann in self._annotations:
            if ann["sample_token"] != sample_token:
                continue
            anns.append(ann)
        return anns

    def _match_gt(self, det, gt_anns, uv, depth, valid_proj, K, T_lidar2cam, points_lidar):
        """Match a YOLO detection to a GT annotation by 3D distance.

        Extracts LiDAR points in the 2D detection bbox, computes their mean 3D position,
        and finds the closest GT annotation center.
        """
        x1, y1, x2, y2 = det["bbox"]

        # Get LiDAR points within 2D bbox
        in_bbox = (
            valid_proj &
            (uv[:, 0] >= x1) & (uv[:, 0] <= x2) &
            (uv[:, 1] >= y1) & (uv[:, 1] <= y2) &
            (depth > 0.5)
        )
        pts_in = points_lidar[in_bbox]

        if len(pts_in) < 5:
            return None

        # Mean 3D position of points in bbox
        mean_3d = pts_in[:, :3].mean(axis=0)

        # Find nearest GT annotation
        best_match = None
        best_dist = 3.0  # 3m threshold

        for ann in gt_anns:
            gt_center = np.array(ann["translation"], dtype=np.float32)
            dist = np.linalg.norm(mean_3d - gt_center)
            if dist < best_dist:
                best_dist = dist
                best_match = ann

        return best_match

    def _build_sample(self, obj_points, ann):
        """Build per-object training sample (same format as NuScenesBboxDataset)."""
        rng = np.random.default_rng()

        gt_center = np.array(ann["translation"], dtype=np.float32)
        gt_size = np.array(ann["size"], dtype=np.float32)
        gt_yaw = quaternion_to_yaw(*ann["rotation"])

        # Add noise for refinement task
        noisy_center = gt_center + rng.normal(0, self.NOISE_CENTER, 3).astype(np.float32)
        noisy_size = gt_size + rng.normal(0, self.NOISE_SIZE, 3).astype(np.float32)
        noisy_size = np.clip(noisy_size, 0.3, 10.0)
        noisy_yaw = gt_yaw + math.radians(rng.normal(0, self.NOISE_YAW_DEG))

        # Normalize to local coordinates
        local_xyz = obj_points[:, :3] - noisy_center
        local_xyz = rotate_points_z(local_xyz, -noisy_yaw)
        local_xyz = self._resample(local_xyz, self.num_points)

        intensity = obj_points[:, 3:4] if obj_points.shape[1] >= 4 else np.zeros((len(obj_points), 1), dtype=np.float32)
        intensity = self._resample(intensity, self.num_points)

        point_features = np.concatenate([local_xyz, intensity], axis=1).astype(np.float32)

        delta_center = (gt_center - noisy_center).astype(np.float32)
        delta_size = (gt_size - noisy_size).astype(np.float32)
        delta_yaw = gt_yaw - noisy_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))

        target = np.array([
            delta_center[0], delta_center[1], delta_center[2],
            delta_size[0], delta_size[1], delta_size[2],
            math.sin(delta_yaw), math.cos(delta_yaw),
        ], dtype=np.float32)

        return torch.from_numpy(point_features), torch.from_numpy(target)

    def _resample(self, data, n):
        if len(data) >= n:
            return self._fps(data, n)
        repeats = n // len(data) + 1
        return np.tile(data, (repeats, 1))[:n]

    def _fps(self, xyz, n):
        idx = np.zeros(n, dtype=np.int64)
        dist = np.ones(len(xyz), dtype=np.float32) * 1e10
        farthest = 0
        for i in range(n):
            idx[i] = farthest
            d = np.sum((xyz - xyz[farthest]) ** 2, axis=1)
            dist = np.minimum(dist, d)
            farthest = np.argmax(dist)
        return xyz[idx]


def phase1_collate(batch):
    """Collate samples from potentially multiple frames into a flat batch."""
    all_samples = []
    for frame_samples in batch:
        all_samples.extend(frame_samples)
    if not all_samples:
        return torch.zeros(1, 256, 4), torch.zeros(1, 8)
    points, targets, *_ = zip(*all_samples)
    return torch.stack(points, dim=0), torch.stack(targets, dim=0)
