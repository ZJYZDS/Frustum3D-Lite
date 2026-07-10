"""
Phase 2 数据集: YOLO 检测 → RGB crop + LiDAR 点云 → 训练样本.

每个样本的生成流程:
  1. 对一帧图像运行 YOLO → K 个 2D 检测框
  2. 将 LiDAR 点云投影到图像平面
  3. 对每个检测框:
     a. 框内裁剪 RGB, resize 到 128×128
     b. 框内提取 LiDAR 点 (带 5px margin)
     c. 通过 2D 中心距离匹配 GT (投影 GT 3D 中心 → 比较与 bbox 中心的像素距离)
     d. 对 GT 加噪声, 构建残差回归目标

GT 匹配的关键设计:
  - nuScenes 中 GT annotation 的 translation 是 global 坐标系
  - 需要 global → ego → camera → image 的完整坐标变换链
  - 用 2D 中心距离匹配 (而非 Phase 1 失败的 3D LiDAR 点均值匹配)

训练目标:
  - 模型学习从 noisy 初始值 → GT 的残差
  - noise: center=0.3m, size=0.15m, yaw=5°
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
from src.detector import YOLODetectONNX, OBSTACLE_CLASS_IDS


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

    __getitem__ 返回 list[(rgb_crop, lidar_pts, target)], 由 phase2_collate
    展开为 (B, ...) 的 batch.
    """

    # 默认超参数
    MIN_LIDAR_PTS = 10        # bbox 内最少 LiDAR 点数, 否则跳过该检测
    NUM_POINTS = 256           # FPS 重采样后的点数
    CROP_SIZE = 128            # RGB crop resize 尺寸
    BBOX_MARGIN = 0.3          # (未使用, 保留)
    NOISE_CENTER = 0.3         # center 噪声标准差 (米)
    NOISE_SIZE = 0.15          # size 噪声标准差 (米)
    NOISE_YAW_DEG = 5.0        # yaw 噪声标准差 (度)
    PC_INIT_PROB = 0.7          # 点云初始化概率 (否则 GT+noise)
    MATCH_MAX_DIST_PX = 80     # GT 匹配的最大 2D 中心距离 (像素)

    def __init__(self, data_root, split="train", cfg=None,
                 detector_path="models/yolo26s.onnx", return_category=False):
        self.data_root = Path(data_root)
        self.split = split
        cfg = cfg or {}

        # 从 config 覆盖默认值
        self.num_points = cfg.get("num_points", self.NUM_POINTS)
        self.crop_size = cfg.get("crop_size", self.CROP_SIZE)
        self.bbox_margin = cfg.get("bbox_margin", self.BBOX_MARGIN)
        self.match_max_dist = cfg.get("match_max_dist_px", self.MATCH_MAX_DIST_PX)
        self.return_category = cfg.get("return_category", return_category)
        val_scene_ids = cfg.get("val_scene_ids", 2)

        print(f"[Phase2] Loading metadata...")
        self._load_metadata(data_root)

        print(f"[Phase2] Loading detector: {detector_path}")
        self.detector = YOLODetectONNX(detector_path, conf_thresh=0.5)

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
        """对一帧: YOLO 检测 → 对每个检测生成样本 → 返回样本列表.

        Returns:
            list[(rgb_crop, lidar_pts, target)] 或空列表 (无有效检测时)
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
        """处理一个 YOLO 检测 → (rgb_crop, lidar_pts, target).

        步骤:
          1. RGB crop: bbox 内裁剪 + resize 到 128×128
          2. LiDAR 点提取: 投影在 bbox 内的 3D 点 (带 5px margin)
          3. GT 匹配: 2D 中心距离最近匹配
          4. 目标构建: 加噪声 → 残差
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

        # ---- LiDAR 点在 bbox 内 ----
        margin = 5   # 像素 margin: 补偿投影误差和 bbox 不精确
        in_bbox = (
            valid_proj &
            (uv[:, 0] >= x1 - margin) & (uv[:, 0] <= x2 + margin) &
            (uv[:, 1] >= y1 - margin) & (uv[:, 1] <= y2 + margin) &
            (depth > 0.5)   # 过滤近处噪声点 (相机后方或太近)
        )
        obj_pts = lidar[in_bbox]

        if len(obj_pts) < self.MIN_LIDAR_PTS:
            return None

        # ---- GT 匹配 (2D 中心距离) ----
        matched_gt = self._match_gt_2d(det["bbox"], gt_anns, K, sample_token)
        if matched_gt is None:
            return None

        # ---- 构建训练目标 ----
        lidar_features, target = self._build_target(obj_pts, matched_gt, sample_token)

        if self.return_category:
            cat_name = self._inst2cat_name.get(matched_gt["instance_token"], "unknown")
            return rgb_crop, lidar_features, target, cat_name
        return rgb_crop, lidar_features, target

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

    def _build_target(self, obj_points, ann, sample_token):
        """构建训练样本: 用点云统计量或 GT+噪声 作为初始框, 计算残差作为回归目标.

        两种模式 (随机切换, 消除训练/推理 domain gap):
          A. 点云初始化 (pc_init):  point cloud mean/PCA → 模拟推理时无 GT 的场景
          B. GT 加噪声 (gt_noise):   GT + small noise → 模拟有 GT 匹配的推理场景

        流程:
          1. GT 从 global → ego 坐标系 (与 LiDAR 一致)
          2. 生成 noisy center/size/yaw (pc_init 或 gt_noise)
          3. 点云去中心化 + 旋转对齐 noisy 朝向
          4. 按物体物理范围归一化坐标 (使 SA 半径自适应物体尺度)
          5. FPS 重采样到固定点数 (256)
          6. 目标 = [Δcenter, Δsize, sin(Δyaw), cos(Δyaw)]
        """
        rng = np.random.default_rng()
        use_pc_init = rng.random() < self.PC_INIT_PROB

        # GT global → ego
        gt_center_global = np.array(ann["translation"], dtype=np.float32)
        gt_center = self._global_to_ego(gt_center_global, sample_token)
        gt_size = np.array(ann["size"], dtype=np.float32)
        gt_yaw = quaternion_to_yaw(*ann["rotation"])

        # yaw 从 global 调整到 ego 坐标系
        ego_pose_token = self._sample_ego.get(sample_token)
        if ego_pose_token:
            ego_pose = self._ego_poses[ego_pose_token]
            ego_yaw = quaternion_to_yaw(*ego_pose["rotation"])
            gt_yaw = gt_yaw - ego_yaw

        if use_pc_init and len(obj_points) >= 20:
            # 点云初始化: 模拟推理时的真实输入分布
            obj_xyz = obj_points[:, :3].astype(np.float32)
            noisy_center = obj_xyz.mean(axis=0)

            # PCA 估计朝向
            centered = obj_xyz[:, :2] - noisy_center[:2]
            cov = centered.T @ centered / len(centered)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, -1]
            pca_yaw = math.atan2(principal[1], principal[0])
            pca_yaw = math.atan2(math.sin(pca_yaw), math.cos(pca_yaw))
            if abs(pca_yaw) > math.pi / 2:
                pca_yaw -= math.copysign(math.pi, pca_yaw)
            noisy_yaw = float(pca_yaw)

            # size: GT + 噪声 (部分可见点云无法估计完整尺寸)
            noisy_size = gt_size + rng.normal(0, self.NOISE_SIZE, 3).astype(np.float32)
            noisy_size = np.clip(noisy_size, 0.3, 20.0)
        else:
            # GT + 噪声: 保留 refiner 能力
            noisy_center = gt_center + rng.normal(0, self.NOISE_CENTER, 3).astype(np.float32)
            noisy_size = gt_size + rng.normal(0, self.NOISE_SIZE, 3).astype(np.float32)
            noisy_size = np.clip(noisy_size, 0.3, 20.0)
            noisy_yaw = gt_yaw + math.radians(rng.normal(0, self.NOISE_YAW_DEG))

        # 点云归一化: 去中心 + 旋转对齐 noisy 朝向
        local_xyz = obj_points[:, :3] - noisy_center
        local_xyz = rotate_points_z(local_xyz, -noisy_yaw)

        # 按物体物理范围归一化 → SA 半径自适应尺度
        extent = np.max(np.ptp(local_xyz, axis=0))                    # 点云跨度 (米)
        extent = np.clip(extent, 0.3, 15.0)                           # 限制在合理范围
        scale = extent / 2.0                                          # 半跨度 ≈ 典型物体半径
        local_xyz = local_xyz / scale                                 # 归一化到 0~1 范围
        scale_feat = np.full((len(obj_points), 1), np.log(scale), dtype=np.float32)

        local_xyz = self._resample(local_xyz, self.num_points)

        # intensity (如果有)
        intensity = (obj_points[:, 3:4]
                     if obj_points.shape[1] >= 4
                     else np.zeros((len(obj_points), 1), dtype=np.float32))
        intensity = self._resample(intensity, self.num_points)
        scale_feat = self._resample(scale_feat, self.num_points)

        point_features = np.concatenate([local_xyz, intensity, scale_feat], axis=1).astype(np.float32)

        # 残差目标
        delta_center = (gt_center - noisy_center).astype(np.float32)
        delta_size = (gt_size - noisy_size).astype(np.float32)
        delta_yaw = gt_yaw - noisy_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))  # 归一化到 [-π, π]

        target = np.array([
            delta_center[0], delta_center[1], delta_center[2],
            delta_size[0], delta_size[1], delta_size[2],
            math.sin(delta_yaw), math.cos(delta_yaw),
        ], dtype=np.float32)

        return torch.from_numpy(point_features), torch.from_numpy(target)

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
    """将 __getitem__ 返回的 frame-level 样本列表展开为 batch tensor."""
    all_samples = []
    for frame_samples in batch:
        all_samples.extend(frame_samples)

    if not all_samples:
        return (
            torch.zeros(1, 3, 128, 128),
            torch.zeros(1, 256, 4),
            torch.zeros(1, 8),
        )

    # 支持 4-tuple (with category) 和 3-tuple
    has_cat = len(all_samples[0]) == 4
    if has_cat:
        rgb_crops, lidar_pts, targets, categories = zip(*all_samples)
        return (
            torch.stack(rgb_crops, dim=0),
            torch.stack(lidar_pts, dim=0),
            torch.stack(targets, dim=0),
            list(categories),
        )
    else:
        rgb_crops, lidar_pts, targets = zip(*all_samples)
        return (
            torch.stack(rgb_crops, dim=0),
            torch.stack(lidar_pts, dim=0),
            torch.stack(targets, dim=0),
        )
