"""
Phase 2 数据集: YOLO-seg 检测 → RGB crop + mask-filtered LiDAR → 训练样本.

每个样本的生成流程:
  1. 对一帧图像运行 YOLO-seg → K 个检测 (bbox + mask)
  2. 将 LiDAR 点云投影到图像平面
  3. 对每个检测:
     a. bbox 内裁剪 RGB, resize 到 128×128
     b. YOLO-seg mask 过滤 LiDAR 点 (dilate=10 补偿投影对齐)
     c. 通过 2D 中心距离匹配 GT
     d. 构建 offset 目标: visible_centroid → volume_center

训练目标:
  - Center: 学习可见表面质心 → 体积中心 offset (0–1.5m)
  - Size/Yaw: GT + 小噪声残差
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.dataset_phase1 import (LiDARProjector, quaternion_to_yaw, quaternion_to_mat,
                                  rotate_points_z)
from src.detector import YOLOSegONNX, OBSTACLE_CLASS_IDS
from src.fusion import COCO_CLS_TO_GROUP
from src.init_estimator import filter_points_by_mask


# nuScenes 类别名 → COCO 类别索引 (用于一致性检查)
NUSCENES_CAT_TO_COCO = {
    "vehicle.car": 2,
    "vehicle.truck": 7,
    "vehicle.bus": 5,
    "vehicle.motorcycle": 3,
    "vehicle.bicycle": 1,
    "human.pedestrian": 0,
}
COCO_VEHICLE_IDS = {1, 2, 3, 5, 7}


class Phase2Dataset(Dataset):
    """Phase 2 数据集: 每帧返回多个 per-object 训练样本的列表.

    __getitem__ 返回 list[(rgb_crop, lidar_pts, target, class_group)], 由 phase2_collate
    展开为 (B, ...) 的 batch.
    """

    # 默认超参数
    MIN_LIDAR_PTS = 10        # mask 内最少 LiDAR 点数, 否则跳过该检测
    NUM_POINTS = 256           # FPS 重采样后的点数
    CROP_SIZE = 128            # RGB crop resize 尺寸
    NOISE_SIZE = 0.15          # size 噪声标准差 (米) — 好框
    NOISE_YAW_DEG = 5.0        # yaw 噪声标准差 (度) — 好框
    MAX_DELTA_CENTER = 2.0      # center offset 截断 (米): 可见表面→体积中心 ≤ 2m
    MAX_DELTA_SIZE = 1.0        # size 残差截断 (米)
    MAX_DELTA_YAW_DEG = 45.0    # yaw 残差截断 (度)
    MATCH_MAX_DIST_PX = 80     # GT 匹配的最大 2D 中心距离 (像素)

    # 各类别典型尺寸 (宽, 长, 高) — nuScenes 均值, 用于烂框默认值 + size_diff 计算
    CLS_AVG_SIZE = {
        "vehicle.car":        np.array([2.0, 4.5, 1.6], dtype=np.float32),
        "vehicle.truck":      np.array([2.8, 7.0, 2.5], dtype=np.float32),
        "vehicle.bus":        np.array([2.8, 10.0, 3.0], dtype=np.float32),
        "vehicle.motorcycle": np.array([0.8, 2.2, 1.5], dtype=np.float32),
        "vehicle.bicycle":    np.array([0.5, 1.8, 1.2], dtype=np.float32),
        "human.pedestrian":   np.array([0.7, 0.7, 1.75], dtype=np.float32),
    }

    def __init__(self, data_root, split="train", cfg=None,
                 detector_path="models/yolov8s-seg.onnx"):
        self.data_root = Path(data_root)
        self.split = split
        cfg = cfg or {}

        # 从 config 覆盖默认值
        self.num_points = cfg.get("num_points", self.NUM_POINTS)
        self.crop_size = cfg.get("crop_size", self.CROP_SIZE)
        self.match_max_dist = cfg.get("match_max_dist_px", self.MATCH_MAX_DIST_PX)
        val_scene_ids = cfg.get("val_scene_ids", 2)

        print(f"[Phase2] Loading metadata...")
        self._load_metadata(data_root)

        print(f"[Phase2] Loading detector: {detector_path}")
        self.detector = YOLOSegONNX(detector_path, conf_thresh=0.5)

        print(f"[Phase2] Loading projector...")
        self.projector = LiDARProjector(data_root)

        print(f"[Phase2] Building category lookup...")
        self._build_category_map()

        print(f"[Phase2] Building sample list...")
        self._build_sample_list(val_scene_ids)

    # ==========================================================================
    # 初始化: 加载 nuScenes 元数据
    # ==========================================================================

    def _load_json(self, name):
        with open(os.path.join(self.data_root, "v1.0-mini", name)) as f:
            return json.load(f)

    def _load_metadata(self, data_root):
        """加载 nuScenes v1.0-mini 的所有 JSON 表."""
        self._categories = {c["token"]: c for c in self._load_json("category.json")}
        self._scenes = self._load_json("scene.json")
        self._samples = {s["token"]: s for s in self._load_json("sample.json")}
        self._annotations = self._load_json("sample_annotation.json")
        self._instances = {i["token"]: i for i in self._load_json("instance.json")}
        self._sample_data = {s["token"]: s for s in self._load_json("sample_data.json")}
        self._ego_poses = {e["token"]: e for e in self._load_json("ego_pose.json")}

    # ==========================================================================
    # 初始化: 构建查找表
    # ==========================================================================

    def _build_category_map(self):
        """构建 3 个查找表:
          _inst2cat_name:  instance_token → category 名称 (如 "vehicle.car")
          _gt_by_sample:   sample_token → 该帧所有相关 GT annotations
          _frame_sensors:  sample_token → {sensor_name: sample_data_token}
          _sample_ego:     sample_token → ego_pose_token (通过 CAM_FRONT 关联)
        """
        # instance → category name
        self._inst2cat_name = {}
        for inst_token, inst in self._instances.items():
            cat = self._categories.get(inst["category_token"])
            self._inst2cat_name[inst_token] = cat["name"] if cat else "unknown"

        # GT annotations 按 sample 分组, 只保留 nuScenes 关注类别
        self._gt_by_sample = {}
        for ann in self._annotations:
            cat_name = self._inst2cat_name.get(ann["instance_token"], "unknown")
            if cat_name not in NUSCENES_CAT_TO_COCO:
                continue
            self._gt_by_sample.setdefault(ann["sample_token"], []).append(ann)

        # 传感器查找: sample_token → {sensor: sd_token}
        self._frame_sensors = {}
        for sd_token, sd in self._sample_data.items():
            sensor = sd["filename"].split("/")[1]   # 如 "CAM_FRONT", "LIDAR_TOP"
            sample_token = sd["sample_token"]
            self._frame_sensors.setdefault(sample_token, {})[sensor] = sd_token

        # ego pose 查找: 通过 CAM_FRONT 的 ego_pose_token 关联
        self._sample_ego = {}
        for sd_token, sd in self._sample_data.items():
            if "CAM_FRONT" in sd["filename"]:
                self._sample_ego[sd["sample_token"]] = sd["ego_pose_token"]

    # ==========================================================================
    # 坐标变换: global → ego → camera
    # ==========================================================================

    def _global_to_ego(self, point_global, sample_token):
        """global 坐标 → ego-vehicle 坐标.

        nuScenes 中 GT annotation translation 是 global 坐标系,
        而 LiDAR 在 ego 坐标系, 所以需要这个变换.
        """
        ego_pose_token = self._sample_ego.get(sample_token)
        if ego_pose_token is None:
            return None
        ego_pose = self._ego_poses[ego_pose_token]
        R_ego = quaternion_to_mat(*ego_pose["rotation"])
        t_ego = np.array(ego_pose["translation"], dtype=np.float32)
        # 逆变换: global → ego
        return R_ego.T @ (point_global.astype(np.float32) - t_ego)

    def _global_to_lidar(self, point_global, sample_token):
        """global → LiDAR 帧 (与 obj_points 统一坐标系, 消除训练/推理 domain gap)."""
        pt_ego = self._global_to_ego(point_global, sample_token)
        if pt_ego is None:
            return None
        lidar_calib_token = self.projector._sample_sensor_calib.get(sample_token, {}).get("LIDAR_TOP")
        if lidar_calib_token is None:
            return None
        lidar_calib = self.projector.calibs[lidar_calib_token]
        R_lidar = quaternion_to_mat(*lidar_calib["rotation"])
        t_lidar = np.array(lidar_calib["translation"], dtype=np.float32)
        return R_lidar.T @ (pt_ego - t_lidar)   # Ego → LiDAR

    def _global_to_camera(self, point_global, sample_token):
        """global 坐标 → camera 坐标 (用于 GT 投影到图像平面).

        变换链: global → ego → camera (CAM_FRONT).
        返回 None 如果点在相机后方 (z <= 0.5m).
        """
        pt_ego = self._global_to_ego(point_global, sample_token)
        if pt_ego is None:
            return None

        # 获取 CAM_FRONT 的标定参数
        cam_calib_token = self.projector._sample_sensor_calib.get(
            sample_token, {}).get("CAM_FRONT")
        if cam_calib_token is None:
            return None

        cam_calib = self.projector.calibs[cam_calib_token]
        R_cam = quaternion_to_mat(*cam_calib["rotation"])
        t_cam = np.array(cam_calib["translation"], dtype=np.float32)

        pt_cam = R_cam.T @ (pt_ego - t_cam)
        return pt_cam.astype(np.float32)

    # ==========================================================================
    # 初始化: 构建样本帧列表
    # ==========================================================================

    def _build_sample_list(self, val_scene_ids):
        """按场景切分 train/val, 构建有效帧列表.

        最后 val_scene_ids 个场景作为验证集, 其余作为训练集.
        只保留同时有 CAM_FRONT 和 LIDAR_TOP 的帧.
        """
        # 按场景分组 sample
        scene_samples = {}
        for sample_token, sample in self._samples.items():
            scene_samples.setdefault(sample["scene_token"], []).append(sample_token)

        # 按名称排序, 分配 train/val
        sorted_scenes = sorted(self._scenes, key=lambda s: s["name"])
        if self.split == "train":
            split_scenes = sorted_scenes[:-val_scene_ids] if val_scene_ids > 0 else sorted_scenes
        else:
            split_scenes = sorted_scenes[-val_scene_ids:] if val_scene_ids > 0 else []

        # 收集有效帧 (同时有 CAM_FRONT 和 LIDAR_TOP)
        self._frames = []
        for scene in split_scenes:
            for sample_token in scene_samples.get(scene["token"], []):
                sensors = self._frame_sensors.get(sample_token, {})
                if "CAM_FRONT" in sensors and "LIDAR_TOP" in sensors:
                    self._frames.append(sample_token)

        print(f"[Phase2][{self.split}] {len(split_scenes)} scenes, {len(self._frames)} frames")

    # ==========================================================================
    # Dataset 接口
    # ==========================================================================

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        """对一帧: YOLO-seg 检测 → 对每个检测生成样本 → 返回样本列表.

        Returns:
            list[(rgb_crop, lidar_pts, target, class_group)] 或空列表 (无有效检测时)
        """
        sample_token = self._frames[idx]

        # 加载图像
        img = self._load_image(sample_token)
        if img is None:
            return []

        # YOLO 检测, 只保留障碍物类别
        dets = self.detector.predict(img)
        dets = [d for d in dets if d["class_id"] in OBSTACLE_CLASS_IDS]
        if not dets:
            return []

        # 加载 LiDAR
        lidar = self._load_lidar(sample_token)
        if lidar is None:
            return []

        # 获取投影矩阵
        K, T_lidar2cam, img_shape = self.projector.get_transform(sample_token)
        if K is None:
            return []

        # LiDAR → 图像平面投影
        uv, depth, valid_proj = self.projector.project(lidar, K, T_lidar2cam, img_shape)

        # 该帧的 GT annotations
        gt_anns = self._gt_by_sample.get(sample_token, [])

        # 对每个检测生成样本
        samples = []
        for det in dets:
            sample = self._process_detection(det, img, lidar, uv, depth,
                                             valid_proj, gt_anns, K, T_lidar2cam,
                                             sample_token)
            if sample is not None:
                samples.append(sample)

        return samples

    # ==========================================================================
    # 核心: 单个检测的处理
    # ==========================================================================

    def _process_detection(self, det, img, lidar, uv, depth, valid_proj,
                           gt_anns, K, T_lidar2cam, sample_token):
        """处理一个 YOLO-seg 检测 → (rgb_crop, lidar_pts, target, class_group).

        步骤:
          1. RGB crop: bbox 内裁剪 + resize 到 128×128
          2. LiDAR 点提取: YOLO-seg mask 过滤 (dilate=10 补偿投影对齐)
          3. GT 匹配: 2D 中心距离最近匹配
          4. 目标构建: 可见表面质心 → volume center offset
        """
        x1, y1, x2, y2 = det["bbox"].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # ---- RGB crop ----
        rgb_crop = img[y1:y2, x1:x2]
        rgb_crop = cv2.resize(rgb_crop, (self.crop_size, self.crop_size),
                              interpolation=cv2.INTER_LINEAR)
        # BGR→RGB, HWC→CHW, uint8→float32 [0,1]
        rgb_crop = rgb_crop[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        rgb_crop = torch.from_numpy(rgb_crop)

        # ---- LiDAR 点: YOLO-seg mask 过滤 (替代 bbox margin 裁剪) ----
        obj_pts = filter_points_by_mask(uv, valid_proj, depth, det["mask"], lidar, dilate=10)
        if obj_pts is None or len(obj_pts) < self.MIN_LIDAR_PTS:
            return None

        # ---- GT 匹配 (2D 中心距离) ----
        matched_gt = self._match_gt_2d(det["bbox"], gt_anns, K, sample_token)
        if matched_gt is None:
            return None

        # ---- 构建训练目标 ----
        cat_name = self._inst2cat_name.get(matched_gt["instance_token"], "unknown")
        lidar_features, target = self._build_target(
            obj_pts, matched_gt, sample_token, cat_name)

        class_group = COCO_CLS_TO_GROUP.get(det["class_id"], "medium")
        return rgb_crop, lidar_features, target, class_group

    # ==========================================================================
    # GT 匹配: 2D 中心距离
    # ==========================================================================

    def _match_gt_2d(self, det_bbox, gt_anns, K, sample_token):
        """通过 2D bbox 中心距离匹配 GT.

        对每个 GT:
          1. 3D center (global) → camera 坐标
          2. camera → 图像平面投影 (u, v)
          3. 计算 (u, v) 到 YOLO bbox 中心的欧氏距离
          4. 取距离最近且在阈值内的 GT

        这比 Phase 1 的 3D LiDAR 点均值匹配可靠得多:
          - 2D 投影 + 距离匹配直接对应"这个检测框里是什么物体"
          - 不需要假设 GT center 与 LiDAR 点均值一致
        """
        det_cx = (det_bbox[0] + det_bbox[2]) / 2
        det_cy = (det_bbox[1] + det_bbox[3]) / 2

        best_match = None
        best_dist = self.match_max_dist

        for ann in gt_anns:
            gt_center = np.array(ann["translation"], dtype=np.float32)

            # global → camera (用于投影)
            pt_cam = self._global_to_camera(gt_center, sample_token)
            if pt_cam is None or pt_cam[2] <= 0.5:   # 相机后方或太近
                continue

            # 透视投影
            uv = K @ pt_cam
            u, v = uv[0] / uv[2], uv[1] / uv[2]

            # 检查是否在图像内 (nuScenes CAM_FRONT: 1600×900)
            if not (0 <= u < 1600 and 0 <= v < 900):
                continue

            dist = np.sqrt((det_cx - u) ** 2 + (det_cy - v) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_match = ann

        return best_match

    # ==========================================================================
    # 训练目标构建: GT + noise → residual
    # ==========================================================================

    def _build_target(self, obj_points, ann, sample_token, cat_name="unknown"):
        """构建训练目标: 可见表面质心 → 体积中心 offset.

        核心思路 (Spatial Mapper):
          - noisy_center = visible_centroid (z-bottom 35% mean): 传感器测到的真实位置
          - target = GT_center - visible_centroid: 网络学习"可见表面到体积中心的偏移"
          - 训练/推理使用完全相同的初始化, 无 distribution shift
          - Offset 范围 0–1.5m (正视: ~0.7m, 侧视: ~0.1m), 恰好落在 SA 半径内

        尺寸/yaw 保留 GT+小噪声 (网络同时学习精修这两个维度).
        """
        rng = np.random.default_rng()

        # GT global → LiDAR 帧
        gt_center_global = np.array(ann["translation"], dtype=np.float32)
        gt_center = self._global_to_lidar(gt_center_global, sample_token)
        gt_size = np.array(ann["size"], dtype=np.float32)
        gt_yaw = quaternion_to_yaw(*ann["rotation"])

        # yaw 从 global 调整到 LiDAR 坐标系
        ego_pose_token = self._sample_ego.get(sample_token)
        if ego_pose_token:
            ego_pose = self._ego_poses[ego_pose_token]
            ego_yaw = quaternion_to_yaw(*ego_pose["rotation"])
            gt_yaw = gt_yaw - ego_yaw
        lidar_calib_token = self.projector._sample_sensor_calib.get(sample_token, {}).get("LIDAR_TOP")
        if lidar_calib_token:
            lidar_calib = self.projector.calibs[lidar_calib_token]
            lidar_yaw = quaternion_to_yaw(*lidar_calib["rotation"])
            gt_yaw = gt_yaw - lidar_yaw

        # ---- Center: 可见表面质心 (z-bottom 35%) = 传感器真实测量 ----
        obj_xyz = obj_points[:, :3].astype(np.float32)
        z_vals = obj_xyz[:, 2]
        z_cut = z_vals.min() + (z_vals.max() - z_vals.min()) * 0.35
        bottom_mask = z_vals <= z_cut
        if bottom_mask.sum() >= 3:
            visible_centroid = obj_xyz[bottom_mask].mean(axis=0)
        else:
            visible_centroid = obj_xyz.mean(axis=0)
        noisy_center = visible_centroid

        # ---- Size/Yaw: GT + 小噪声 (与之前一致) ----
        noisy_size = gt_size + rng.normal(0, self.NOISE_SIZE, 3).astype(np.float32)
        noisy_size = np.clip(noisy_size, 0.3, 20.0)
        noisy_yaw = gt_yaw + math.radians(rng.normal(0, self.NOISE_YAW_DEG))

        # ---- 点云归一化: 以 visible_centroid 为原点, 对齐 noisy_yaw ----
        local_xyz = obj_points[:, :3] - noisy_center
        local_xyz = rotate_points_z(local_xyz, -noisy_yaw)

        extent = np.max(np.ptp(local_xyz, axis=0))
        extent = np.clip(extent, 0.3, 15.0)
        scale = extent / 2.0
        local_xyz = local_xyz / scale
        scale_feat = np.full((len(obj_points), 1), np.log(scale), dtype=np.float32)

        local_xyz = self._resample(local_xyz, self.num_points)

        intensity = (obj_points[:, 3:4]
                     if obj_points.shape[1] >= 4
                     else np.zeros((len(obj_points), 1), dtype=np.float32))
        intensity = self._resample(intensity, self.num_points)
        scale_feat = self._resample(scale_feat, self.num_points)

        point_features = np.concatenate([local_xyz, intensity, scale_feat], axis=1).astype(np.float32)

        # ---- Offset target: GT - visible_centroid (可见表面 → 体积中心) ----
        delta_center = (gt_center - noisy_center).astype(np.float32)
        delta_center = np.clip(delta_center, -self.MAX_DELTA_CENTER, self.MAX_DELTA_CENTER)
        delta_size = (gt_size - noisy_size).astype(np.float32)
        delta_size = np.clip(delta_size, -self.MAX_DELTA_SIZE, self.MAX_DELTA_SIZE)
        delta_yaw = gt_yaw - noisy_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))
        delta_yaw = np.clip(delta_yaw, -math.radians(self.MAX_DELTA_YAW_DEG),
                            math.radians(self.MAX_DELTA_YAW_DEG))

        target = np.array([
            delta_center[0], delta_center[1], delta_center[2],
            delta_size[0], delta_size[1], delta_size[2],
            math.sin(delta_yaw), math.cos(delta_yaw),
        ], dtype=np.float32)

        return (torch.from_numpy(point_features),
                torch.from_numpy(target))

    # ==========================================================================
    # 点云重采样: FPS (点数足够) 或 重复 (点数不足)
    # ==========================================================================

    def _resample(self, data, n):
        """将点云重采样到固定点数 n.

        点数 ≥ n: FPS (保留空间分布)
        点数 < n: 重复填充 (保证模型输入维度不变)
        """
        if len(data) >= n:
            return self._fps(data, n)
        repeats = n // len(data) + 1
        return np.tile(data, (repeats, 1))[:n]

    def _fps(self, xyz, n):
        """最远点采样 (FPS): 保证采样点空间均匀, 保留形状信息."""
        idx = np.zeros(n, dtype=np.int64)
        dist = np.ones(len(xyz), dtype=np.float32) * 1e10
        farthest = 0
        for i in range(n):
            idx[i] = farthest
            d = np.sum((xyz - xyz[farthest]) ** 2, axis=1)
            dist = np.minimum(dist, d)
            farthest = np.argmax(dist)
        return xyz[idx]

    # ==========================================================================
    # 数据加载辅助
    # ==========================================================================

    def _load_image(self, sample_token):
        """加载 CAM_FRONT 图像."""
        sensors = self._frame_sensors.get(sample_token, {})
        sd_token = sensors.get("CAM_FRONT")
        if sd_token is None:
            return None
        path = os.path.join(self.data_root, self._sample_data[sd_token]["filename"])
        return cv2.imread(path)

    def _load_lidar(self, sample_token):
        """加载 LIDAR_TOP 点云 (binary .bin 文件, float32, 每点 5 维: xyz+intensity+ring)."""
        sensors = self._frame_sensors.get(sample_token, {})
        sd_token = sensors.get("LIDAR_TOP")
        if sd_token is None:
            return None
        path = os.path.join(self.data_root, self._sample_data[sd_token]["filename"])
        return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


# ==============================================================================
# Collate: 将 per-frame 样本列表展开为 batch
# ==============================================================================

def phase2_collate(batch):
    """将 __getitem__ 返回的 frame-level 样本列表展开为 batch tensor.

    每个样本: (rgb_crop, lidar_pts, target, class_group)
    """
    all_samples = []
    for frame_samples in batch:
        all_samples.extend(frame_samples)

    if not all_samples:
        return (
            torch.zeros(1, 3, 128, 128),
            torch.zeros(1, 256, 5),
            torch.zeros(1, 8),
            ["medium"],
        )

    rgb_crops, lidar_pts, targets, class_groups = zip(*all_samples)
    return (
        torch.stack(rgb_crops, dim=0),
        torch.stack(lidar_pts, dim=0),
        torch.stack(targets, dim=0),
        list(class_groups),
    )
