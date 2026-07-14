"""
Phase 3 Dataset: 绝对回归 + 多帧聚合 + 视锥裁剪 + 地面去除 + 点云净化.

管线:
  1. 加载 CAM_FRONT 图像 → YOLO 检测 (bbox only)
  2. 多帧聚合 LiDAR (nsweeps=10) → RANSAC 地面去除
  3. GT 投影中心 2D 匹配
  4. 视锥裁剪 (YOLO 4 条射线) → ROR 离群点剔除 → DBSCAN 最大簇
  5. 采样到固定点数 → 显式极值 + RGB Crop + 绝对回归标签

输入: 原始点云 (无归一化), RGB crop, xyz_min/max (显式特征)
输出: target = [cx, cy, cz, w, h, l, sin(yaw), cos(yaw)]
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.detector import YOLODetectONNX, YOLOPtDetector, OBSTACLE_CLASS_IDS
from src.dataset_phase1 import LiDARProjector, quaternion_to_mat, quaternion_to_yaw
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


# nuScenes 类别名 → 微调模型类别索引 (用于类别感知匹配)
NUSCENES_CAT_TO_CLASS = {
    "vehicle.car": 2,
    "vehicle.truck": 3,
    "vehicle.bus": 4,
    "vehicle.motorcycle": 6,
    "vehicle.bicycle": 7,
    "human.pedestrian": 0,   # rider(1) 也可能匹配到 pedestrian
}


# ==============================================================================
# 多帧聚合
# ==============================================================================

def aggregate_sweeps(nusc, sample_token, channel='LIDAR_TOP', nsweeps=10):
    """聚合当前帧及周围帧的 LiDAR 点云到当前帧坐标系.

    通过 ego_pose 将历史帧点云配准到当前帧, 增加点云密度.

    Returns:
        LidarPointCloud: 聚合后的点云 (points: 4×N)
    """
    from nuscenes.nuscenes import NuScenes
    sd_token = sample_token['data'][channel]
    ref_sd = nusc.get('sample_data', sd_token)
    ref_ego = nusc.get('ego_pose', ref_sd['ego_pose_token'])
    ref_to_world = np.eye(4)
    ref_to_world[:3, :3] = Quaternion(ref_ego['rotation']).rotation_matrix
    ref_to_world[:3, 3] = ref_ego['translation']

    # 向前追溯 nsweeps 帧 (仅历史, 不含未来 — 运动物体未来帧位置不同)
    tokens = []
    cur = sd_token
    while len(tokens) < nsweeps and cur != '':
        tokens.append(cur)
        cur = nusc.get('sample_data', cur)['prev']

    all_points = []
    for tk in tokens:
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(tk))
        sd = nusc.get('sample_data', tk)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
        pc.translate(np.array(cs['translation']))

        if tk != sd_token:
            ep = nusc.get('ego_pose', sd['ego_pose_token'])
            cur_to_world = np.eye(4)
            cur_to_world[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
            cur_to_world[:3, 3] = ep['translation']
            T = np.linalg.inv(ref_to_world) @ cur_to_world
            pc.rotate(T[:3, :3])
            pc.translate(T[:3, 3])

        all_points.append(pc.points)

    combined = np.hstack(all_points)

    # ---- ego → LiDAR: 与标签坐标系对齐 ----
    ref_cs = nusc.get('calibrated_sensor', ref_sd['calibrated_sensor_token'])
    R_lidar = Quaternion(ref_cs['rotation']).rotation_matrix
    t_lidar = np.array(ref_cs['translation'])
    combined[:3, :] = R_lidar.T @ (combined[:3, :] - t_lidar.reshape(3, 1))

    return LidarPointCloud(combined)


# ==============================================================================
# 地面去除 (RANSAC)
# ==============================================================================

def remove_ground_ransac(points, distance_threshold=0.25, num_iterations=200):
    """使用 RANSAC 平面拟合去除地面点.

    拟合最大平面 → 内点为地面, 外点保留 (非地面物体).

    Args:
        points: (N, 3) 点云 xyz
        distance_threshold: 点到平面距离阈值 (米), 默认 0.25
        num_iterations: RANSAC 迭代次数

    Returns:
        (M, 3) 非地面点
    """
    import open3d as o3d

    if len(points) < 10:
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )

    # 保留非地面点 (外点)
    inlier_set = set(inliers)
    mask = np.ones(len(points), dtype=bool)
    mask[list(inlier_set)] = False
    return points[mask]


# ==============================================================================
# 视锥裁剪
# ==============================================================================

def _compute_adaptive_margin(bbox, pts_lidar, K, T_lidar2cam):
    """根据 bbox 大小和深度动态计算 margin.

    规则:
      - bbox 面积 < 2500px² (小物体/远处): margin = 15-20px
      - bbox 面积 > 10000px² (大物体/近处): margin = 3-5px
      - 中间线性插值

    Returns:
        margin (int): 自适应像素扩展
    """
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    # bbox 面积 → margin: 大面积→小margin, 小面积→大margin
    if area <= 0:
        return 10
    margin = int(np.clip(20 - 17 * (area - 500) / 20000, 3, 20))
    return margin


def filter_points_by_frustum(pts_lidar, bbox, K, T_lidar2cam, margin=5):
    """用 YOLO BBox 的 4 条射线裁剪点云.

    Args:
        pts_lidar: (N, 3) LiDAR 坐标系下的点
        bbox: (x1, y1, x2, y2) 像素坐标
        K: (3, 3) 相机内参
        T_lidar2cam: (3, 4) LiDAR → Camera 变换
        margin: 像素边界扩展. 传 'auto' 时自适应.

    Returns:
        (M, 3) 视锥内的 LiDAR 点, 以及使用的 margin 值
    """
    x1, y1, x2, y2 = bbox.astype(int)

    if margin == 'auto':
        margin = _compute_adaptive_margin(bbox, pts_lidar, K, T_lidar2cam)

    # LiDAR → Camera
    pts_cam = (T_lidar2cam[:3, :3] @ pts_lidar.T).T + T_lidar2cam[:3, 3]
    valid_z = pts_cam[:, 2] > 0.5  # 相机前方

    # 透视投影 → 像素坐标
    u = (K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2]) + K[0, 2]
    v = (K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2]) + K[1, 2]

    mask_bbox = (
        (u > x1 - margin) & (u < x2 + margin) &
        (v > y1 - margin) & (v < y2 + margin)
    )
    return pts_lidar[valid_z & mask_bbox], margin


def filter_points_by_bbox_projection(pts_lidar, bbox, K, T_lidar2cam, depth_range=(0.5, 80.0)):
    """Fallback: 简单 bbox 投影 (不做视锥, 类似 Phase 1).

    当视锥裁剪点数不足时回退到此方法, 保证召回率.

    Args:
        pts_lidar: (N, 3) LiDAR 点
        bbox: (x1, y1, x2, y2)
        K, T_lidar2cam: 投影矩阵
        depth_range: (min, max) 深度范围

    Returns:
        (M, 3) bbox 内的 LiDAR 点
    """
    x1, y1, x2, y2 = bbox.astype(int)
    pts_cam = (T_lidar2cam[:3, :3] @ pts_lidar.T).T + T_lidar2cam[:3, 3]
    valid_z = (pts_cam[:, 2] > depth_range[0]) & (pts_cam[:, 2] < depth_range[1])

    u = (K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2]) + K[0, 2]
    v = (K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2]) + K[1, 2]

    mask = valid_z & (u > x1) & (u < x2) & (v > y1) & (v < y2)
    return pts_lidar[mask]


# ==============================================================================
# Face Coverage: 每面点云占对应 3D bbox 面积的比例
# ==============================================================================

def compute_face_coverage(pts_local, bbox_size):
    """计算点云在 3D bbox 各面的覆盖比例.

    将点云投影到 bbox 的 6 个面, 估算每个面的覆盖面积比例.

    Args:
        pts_local: (N, 3) bbox 局部坐标 (x=length, y=width, z=height)
        bbox_size: (3,) [w, l, h] nuScenes 尺寸

    Returns:
        face_cov: (6,) float32 每个面的覆盖比例 [0,1]
          [0]=+x(前), [1]=-x(后), [2]=+y(左), [3]=-y(右), [4]=+z(上), [5]=-z(下)
        max_face_idx: int 覆盖比例最大的面 (0-5)
    """
    if len(pts_local) < 5:
        return np.zeros(6, dtype=np.float32), 0

    w, l, h = bbox_size.astype(np.float32)
    half = np.array([l/2, w/2, h/2], dtype=np.float32)  # x, y, z half-sizes

    # 每面覆盖率: 用点云在该方向的 extent 相对 bbox 半尺寸的比例
    # face_+x = max(x) / (l/2)  → 点云到达 +x 面的程度
    # face_-x = -min(x) / (l/2) → 点云到达 -x 面的程度
    x, y, z = pts_local[:, 0], pts_local[:, 1], pts_local[:, 2]

    coverage = np.zeros(6, dtype=np.float32)
    coverage[0] = np.clip(np.max(x) / (half[0] + 1e-6), 0.0, 1.0)   # +x face
    coverage[1] = np.clip(-np.min(x) / (half[0] + 1e-6), 0.0, 1.0)  # -x face
    coverage[2] = np.clip(np.max(y) / (half[1] + 1e-6), 0.0, 1.0)   # +y face
    coverage[3] = np.clip(-np.min(y) / (half[1] + 1e-6), 0.0, 1.0)  # -y face
    coverage[4] = np.clip(np.max(z) / (half[2] + 1e-6), 0.0, 1.0)   # +z face
    coverage[5] = np.clip(-np.min(z) / (half[2] + 1e-6), 0.0, 1.0)  # -z face

    # 额外: 在该面附近点的密度 (距离面 < 0.15m 的点占总点数的比例)
    near_thresh = 0.15
    coverage[0] = 0.7 * coverage[0] + 0.3 * (np.sum(np.abs(x - half[0]) < near_thresh) / len(pts_local))
    coverage[1] = 0.7 * coverage[1] + 0.3 * (np.sum(np.abs(x + half[0]) < near_thresh) / len(pts_local))
    coverage[2] = 0.7 * coverage[2] + 0.3 * (np.sum(np.abs(y - half[1]) < near_thresh) / len(pts_local))
    coverage[3] = 0.7 * coverage[3] + 0.3 * (np.sum(np.abs(y + half[1]) < near_thresh) / len(pts_local))
    coverage[4] = 0.7 * coverage[4] + 0.3 * (np.sum(np.abs(z - half[2]) < near_thresh) / len(pts_local))
    coverage[5] = 0.7 * coverage[5] + 0.3 * (np.sum(np.abs(z + half[2]) < near_thresh) / len(pts_local))

    coverage = np.clip(coverage, 0.0, 1.0)
    max_face_idx = int(np.argmax(coverage))
    return coverage, max_face_idx


# ==============================================================================
# 点云净化
# ==============================================================================

def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """ROR (Radius Outlier Removal): 剔除孤立离群点.

    Args:
        points: (N, 3) 点云
        nb_neighbors: 邻居数
        std_ratio: 标准差倍数阈值

    Returns:
        (M, 3) 滤波后点云
    """
    import open3d as o3d
    if len(points) < 5:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return np.asarray(pcd.select_by_index(ind).points)


def extract_largest_cluster(points, eps=0.6, min_samples=8):
    """DBSCAN 取最大簇 — 解决多物体重叠/遮挡问题.

    仅在点数 > 30 时执行, 防止稀疏点被误删.

    Args:
        points: (N, 3) 点云
        eps: DBSCAN 邻域半径
        min_samples: 最小样本数

    Returns:
        (M, 3) 最大簇的点云
    """
    if len(points) <= 30:
        return points

    from sklearn.cluster import DBSCAN
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
    valid = labels != -1
    if np.sum(valid) == 0:
        return points
    unique, counts = np.unique(labels[valid], return_counts=True)
    largest_label = unique[np.argmax(counts)]
    return points[labels == largest_label]


# ==============================================================================
# Phase3Dataset
# ==============================================================================

class Phase3Dataset(Dataset):
    """Phase 3 数据集: 绝对回归 + 多帧聚合 + 视锥裁剪 + 点云净化.

    __getitem__ 返回 list[dict], 由 phase3_collate 展开为 batch.
    每个 dict 包含 points, rgb, xyz_min, xyz_max, target.
    """

    def __init__(self,
                 nusc_root: str,
                 version: str = 'v1.0-mini',
                 split: str = 'train',
                 detector_path: str = 'models/yolo26s.onnx',
                 nsweeps: int = 10,
                 num_points: int = 512,
                 crop_size: int = 128,
                 max_dist: float = 15.0,
                 min_points: int = 5,
                 match_threshold: float = 80.0,
                 val_scene_ids: int = 2,
                 test_ratio: float = 0.0,  # train→test 随机抽取比例 (仅 split='train'/'test' 有效)
                 remove_ground: bool = True,
                 use_augmentation: bool = True,
                 preprocess_dir: str = None,
                 frustum_mix_ratio: float = 0.0):  # 0=纯GT, 0.3=30%frustum, 1=纯frustum
        from nuscenes.nuscenes import NuScenes

        self.nusc = NuScenes(version=version, dataroot=nusc_root)
        self.split = split
        self.nsweeps = nsweeps
        self.num_points = num_points
        self.crop_size = crop_size
        self.max_dist = max_dist
        self.min_points = min_points
        self.match_threshold = match_threshold
        self.remove_ground = remove_ground
        self.use_augmentation = use_augmentation
        self.preprocess_dir = preprocess_dir
        self.frustum_mix_ratio = frustum_mix_ratio
        self.test_ratio = test_ratio

        # 加载 YOLO 检测器: .pt 用 ultralytics, .onnx 用 ONNX Runtime
        if detector_path.endswith('.pt'):
            self.detector = YOLOPtDetector(detector_path, conf_thresh=0.25)
        else:
            self.detector = YOLODetectONNX(detector_path, conf_thresh=0.5)

        # 投影器 (复用 Phase1)
        self.projector = LiDARProjector(nusc_root)

        # 构建帧列表
        self._build_frame_list(val_scene_ids, test_ratio)

        # 按 sample 分组 GT
        self._gt_by_sample = self._group_annotations()

        preprocess_info = f", preprocess={preprocess_dir}" if preprocess_dir else ""
        print(f"[Phase3] {split}: {len(self.frames)} frames, "
              f"nsweeps={nsweeps}, num_points={num_points}, "
              f"remove_ground={remove_ground}{preprocess_info}")

        # 预计算所有 per-detection 结果, 存入内存.
        # __getitem__ 只做旋转增强, 不再跑 YOLO/frustum/DBSCAN/匹配.
        self._build_cache()

        # 释放不再需要的重型对象
        self.detector = None
        self.projector = None

    def _build_frame_list(self, val_scene_ids, test_ratio=0.0):
        """构建有效帧列表: val 按场景切分, test 从 train 帧随机抽取.

        split='train': 前 N - val_scene_ids 个场景, 扣除 test 帧
        split='val':   最后 val_scene_ids 个场景
        split='test':  从 train 场景中随机抽 test_ratio 比例的帧
        """
        from nuscenes.nuscenes import NuScenes
        scene_samples = {}
        for sample in self.nusc.sample:
            scene_samples.setdefault(sample['scene_token'], []).append(sample['token'])

        scenes = self.nusc.scene
        sorted_scenes = sorted(scenes, key=lambda s: s['name'])

        # 构建完整帧列表 (有序)
        all_frames = []
        for scene in sorted_scenes:
            for sample_token in scene_samples.get(scene['token'], []):
                sample = self.nusc.get('sample', sample_token)
                if ('CAM_FRONT' in sample['data'] and
                        'LIDAR_TOP' in sample['data']):
                    all_frames.append(sample_token)

        if self.split == 'val':
            # val: 最后 val_scene_ids 个场景的所有帧
            if val_scene_ids <= 0:
                self.frames = []
                return
            val_scene_tokens = {s['token'] for s in sorted_scenes[-val_scene_ids:]}
            self.frames = [f for f in all_frames
                           if self._nusc_scene(self.nusc, f) in val_scene_tokens]
        elif self.split == 'test':
            # test: 从 train 场景中随机抽 test_ratio
            if test_ratio <= 0:
                self.frames = []
                return
            val_scene_tokens = {s['token'] for s in sorted_scenes[-val_scene_ids:]} if val_scene_ids > 0 else set()
            train_frames = [f for f in all_frames
                            if self._nusc_scene(self.nusc, f) not in val_scene_tokens]
            n_test = max(1, int(len(train_frames) * test_ratio))
            rng = np.random.RandomState(42)
            self.frames = list(rng.choice(train_frames, n_test, replace=False))
        else:
            # train: 前 N - val_scene_ids 个场景, 扣除 test
            if val_scene_ids > 0:
                val_scene_tokens = {s['token'] for s in sorted_scenes[-val_scene_ids:]}
            else:
                val_scene_tokens = set()
            train_frames = [f for f in all_frames
                            if self._nusc_scene(self.nusc, f) not in val_scene_tokens]
            if test_ratio > 0:
                rng = np.random.RandomState(42)
                n_test = max(1, int(len(train_frames) * test_ratio))
                test_frames = set(rng.choice(train_frames, n_test, replace=False))
                self.frames = [f for f in train_frames if f not in test_frames]
            else:
                self.frames = train_frames


    @staticmethod
    def _nusc_scene(nusc, sample_token):
        return nusc.get('sample', sample_token)['scene_token']

    def _group_annotations(self):
        """按 sample_token 分组 GT 标注, 同时建立 instance→frames 索引."""
        gt_by_sample = {}
        self._inst_anns = {}
        for ann in self.nusc.sample_annotation:
            gt_by_sample.setdefault(ann['sample_token'], []).append(ann)
            self._inst_anns.setdefault(ann['instance_token'], []).append(
                (ann['sample_token'], ann))
        return gt_by_sample

    # ==========================================================================
    # 坐标变换辅助
    # ==========================================================================

    # ==========================================================================
    # 坐标变换辅助
    # ==========================================================================

    def _get_ego_pose(self, sample):
        """从 sample dict 获取 ego_pose (通过 LIDAR_TOP 的 sample_data)."""
        lidar_sd_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_sd_token)
        return self.nusc.get('ego_pose', lidar_sd['ego_pose_token'])

    def _get_lidar_calib(self, sample):
        """从 sample dict 获取 LiDAR calibrated_sensor."""
        lidar_sd_token = sample['data']['LIDAR_TOP']
        lidar_sd = self.nusc.get('sample_data', lidar_sd_token)
        return self.nusc.get('calibrated_sensor', lidar_sd['calibrated_sensor_token'])

    def _global_to_lidar(self, point_global, sample):
        """global → LiDAR 帧."""
        ego_pose = self._get_ego_pose(sample)
        R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
        t_ego = np.array(ego_pose['translation'])
        pt_ego = R_ego.T @ (np.array(point_global) - t_ego)

        lidar_calib = self._get_lidar_calib(sample)
        R_lidar = Quaternion(lidar_calib['rotation']).rotation_matrix
        t_lidar = np.array(lidar_calib['translation'])
        return R_lidar.T @ (pt_ego - t_lidar)

    def _global_to_camera(self, point_global, sample):
        """global → camera 坐标 (仅用于投影匹配)."""
        ego_pose = self._get_ego_pose(sample)
        R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
        t_ego = np.array(ego_pose['translation'])
        pt_ego = R_ego.T @ (np.array(point_global) - t_ego)

        cam_sd_token = sample['data']['CAM_FRONT']
        cam_sd = self.nusc.get('sample_data', cam_sd_token)
        cam_calib = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        R_cam = Quaternion(cam_calib['rotation']).rotation_matrix
        t_cam = np.array(cam_calib['translation'])
        pt_cam = R_cam.T @ (pt_ego - t_cam)
        return pt_cam

    def _quaternion_to_yaw_lidar(self, q, sample):
        """将全局四元数转为 LiDAR 帧下的 yaw 角."""
        ego_pose = self._get_ego_pose(sample)
        R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
        R_global = Quaternion(q).rotation_matrix
        R_ego_frame = R_ego.T @ R_global

        lidar_calib = self._get_lidar_calib(sample)
        R_lidar = Quaternion(lidar_calib['rotation']).rotation_matrix
        R_lidar_frame = R_lidar.T @ R_ego_frame
        return math.atan2(R_lidar_frame[1, 0], R_lidar_frame[0, 0])

    # ==========================================================================
    # 内存缓存: 初始化时跑完所有重型 CPU 操作, 训练时只读缓存 + 增强
    # ==========================================================================

    def _build_cache(self):
        """初始化时预计算所有 per-detection 结果, 存入 self._cache.

        self._cache 结构:
          list[list[dict]]  — self._cache[frame_idx] = [sample_dict, ...]

        每个 sample_dict (原始, 无增强):
          {'points': np.float32 (512,3), 'rgb': np.float32 (3,128,128),
           'gt_center': np.float32 (3,), 'gt_size': np.float32 (3,),
           'gt_yaw': float32, 'class_id': int, 'bbox_feat': np.float32 (4,),
           'forward_label': float32 (1=朝运动方向, 0=反向, -1=无标签)}
        """
        import time
        t0 = time.time()
        self._cache = []
        total_samples = 0
        total_empty = 0

        for idx, sample_token in enumerate(self.frames):
            samples_raw = self._process_frame(sample_token)
            self._cache.append(samples_raw)
            total_samples += len(samples_raw)
            if not samples_raw:
                total_empty += 1

        elapsed = time.time() - t0
        print(f"[Phase3] {self.split}: cache built — {total_samples} samples "
              f"from {len(self.frames)} frames in {elapsed:.1f}s "
              f"({total_empty} empty frames)")

    def _process_frame(self, sample_token):
        """处理单帧: 跑完 YOLO→frustum→DBSCAN→匹配→采样, 返回原始样本列表 (无增强)."""
        sample = self.nusc.get('sample', sample_token)

        # 1. 加载图像
        img = self._load_image(sample)
        if img is None:
            return []

        # 2. YOLO 检测
        dets = self.detector.predict(img)
        dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]
        if not dets:
            return []

        # 3. 多帧聚合点云
        if self.preprocess_dir:
            cache_path = os.path.join(self.preprocess_dir, f"{sample_token}.npy")
            if os.path.exists(cache_path):
                pts = np.load(cache_path).astype(np.float32)
            else:
                pc = aggregate_sweeps(self.nusc, sample, nsweeps=self.nsweeps)
                pts = pc.points[:3, :].T
                if self.remove_ground:
                    pts = remove_ground_ransac(pts)
        else:
            pc = aggregate_sweeps(self.nusc, sample, nsweeps=self.nsweeps)
            pts = pc.points[:3, :].T
            if self.remove_ground:
                pts = remove_ground_ransac(pts)

        # 4. 投影矩阵
        K, T_lidar2cam, _ = self.projector.get_transform(sample_token)
        if K is None:
            return []

        # 5. GT 标注
        gt_anns = self._gt_by_sample.get(sample_token, [])
        if not gt_anns:
            return []

        # 预计算 GT 类别名
        gt_cat_names = []
        for ann in gt_anns:
            inst = self.nusc.get('instance', ann['instance_token'])
            cat = self.nusc.get('category', inst['category_token'])
            gt_cat_names.append(cat['name'])

        samples = []

        # ---- 多帧点云累积: 同一 instance 跨帧合并点云, 增加几何信息 ----
        # 1. 对每个 GT instance, 收集它出现过的帧的 GT-bbox-interior 点
        # 2. 变换到当前帧 LiDAR 坐标系
        # 3. 合并采样 → 多视角更完整的物体点云

        for ann_idx, ann in enumerate(gt_anns):
            # ---- GT 3D → 2D 投影, 匹配 YOLO ----
            gt_c_lidar = self._global_to_lidar(ann['translation'], sample)
            if gt_c_lidar is None:
                continue
            gt_c_cam = self._global_to_camera(ann['translation'], sample)
            if gt_c_cam is None or gt_c_cam[2] <= 0.5:
                continue
            u = (K[0, 0] * gt_c_cam[0] / gt_c_cam[2]) + K[0, 2]
            v = (K[1, 1] * gt_c_cam[1] / gt_c_cam[2]) + K[1, 2]
            if not (0 <= u < 1600 and 0 <= v < 900):
                continue

            gt_cls_id = NUSCENES_CAT_TO_CLASS.get(gt_cat_names[ann_idx], -1)
            best_det, best_dist, best_same_cls = None, 60.0, False
            for det in dets:
                bx1, by1, bx2, by2 = det['bbox']
                dcx, dcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                d2d = math.sqrt((u - dcx)**2 + (v - dcy)**2)
                same_cls = (gt_cls_id == det['class_id'])
                if not same_cls and det['class_id'] in (0, 1) and gt_cls_id in (0, 1):
                    same_cls = True
                cls_penalty = 0.0 if same_cls else 30.0
                if d2d + cls_penalty < best_dist:
                    best_dist = d2d + cls_penalty
                    best_det = det
                    best_same_cls = same_cls

            if best_det is None or (best_dist > 50 and not best_same_cls):
                continue

            det_class_id = best_det['class_id']
            w, length, h = ann['size']
            yaw = self._quaternion_to_yaw_lidar(ann['rotation'], sample)

            # ---- 选择点云来源: GT-bbox 或 frustum (混合训练) ----
            use_frustum = (self.frustum_mix_ratio > 0 and
                           np.random.rand() < self.frustum_mix_ratio)

            if use_frustum:
                # Frustum 管线: YOLO bbox → 视锥 → ROR → DBSCAN
                frustum_pts, _ = filter_points_by_frustum(
                    pts[:, :3], best_det['bbox'], K, T_lidar2cam, margin='auto')
                if len(frustum_pts) >= self.min_points:
                    f_ror = remove_statistical_outliers(frustum_pts)
                    f_cluster = extract_largest_cluster(
                        f_ror if len(f_ror) >= self.min_points else frustum_pts)
                    all_obj_pts = [f_cluster.astype(np.float32)]
                else:
                    # 点数不足, 回退到 GT-bbox
                    all_obj_pts = []
                    use_frustum = False

            if not use_frustum:
                # GT-bbox 内部点 (原有管线)
                local = pts[:, :3] - gt_c_lidar
                cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
                x = cos_y * local[:, 0] - sin_y * local[:, 1]
                y = sin_y * local[:, 0] + cos_y * local[:, 1]
                z = local[:, 2]
                inside = (abs(x) <= length/2) & (abs(y) <= w/2) & (abs(z) <= h/2)
                all_obj_pts = [pts[inside]]

            # ---- 跨帧累积: 从前后帧收集同一 instance 的点云 ----
            # frustum 模式也累积 (来自其他帧的 GT-bbox 点, 增加几何信息)
            inst_token = ann['instance_token']
            # 查 instance 在其他帧的位置
            other_anns = self._inst_anns.get(inst_token, [])
            n_extra = 0
            for other_sample_token, other_ann in other_anns:
                if n_extra >= 4:  # 最多 4 额外帧
                    break
                if other_sample_token == sample_token:
                    continue

                # 加载该帧的聚合点云
                other_sample = self.nusc.get('sample', other_sample_token)
                if self.preprocess_dir:
                    other_cache = os.path.join(self.preprocess_dir,
                                               f"{other_sample_token}.npy")
                    if os.path.exists(other_cache):
                        other_pts_full = np.load(other_cache).astype(np.float32)
                    else:
                        continue
                else:
                    other_pc = aggregate_sweeps(self.nusc, other_sample,
                                                nsweeps=self.nsweeps)
                    other_pts_full = other_pc.points[:3, :].T
                    if self.remove_ground:
                        other_pts_full = remove_ground_ransac(other_pts_full)

                # 提取该帧 GT bbox 内部点
                other_c = self._global_to_lidar(other_ann['translation'],
                                                 other_sample)
                if other_c is None:
                    continue
                other_yaw = self._quaternion_to_yaw_lidar(
                    other_ann['rotation'], other_sample)
                other_w, other_l, other_h = other_ann['size']

                # 变换到当前帧物体局部坐标系
                # Step 1: other frame LiDAR → object-local(other frame)
                o_local = other_pts_full[:, :3] - other_c
                cos_o, sin_o = math.cos(-other_yaw), math.sin(-other_yaw)
                ox = cos_o * o_local[:, 0] - sin_o * o_local[:, 1]
                oy = sin_o * o_local[:, 0] + cos_o * o_local[:, 1]
                oz = o_local[:, 2]
                o_inside = ((abs(ox) <= other_l/2) & (abs(oy) <= other_w/2) &
                            (abs(oz) <= other_h/2))
                o_pts_local = np.column_stack([ox[o_inside], oy[o_inside],
                                               oz[o_inside]])

                # Step 2: object-local(other frame) → current LiDAR
                cos_c, sin_c = math.cos(yaw), math.sin(yaw)
                pts_cur = np.zeros_like(o_pts_local)
                pts_cur[:, 0] = cos_c * o_pts_local[:, 0] - sin_c * o_pts_local[:, 1] + gt_c_lidar[0]
                pts_cur[:, 1] = sin_c * o_pts_local[:, 0] + cos_c * o_pts_local[:, 1] + gt_c_lidar[1]
                pts_cur[:, 2] = o_pts_local[:, 2] + gt_c_lidar[2]

                if len(pts_cur) >= self.min_points:
                    all_obj_pts.append(pts_cur.astype(np.float32))
                    n_extra += 1

            # 合并所有帧的点云
            obj_pts = np.vstack(all_obj_pts) if len(all_obj_pts) > 1 else all_obj_pts[0]
            if len(obj_pts) < self.min_points:
                continue

            # ---- Face Coverage: 每面点云占对应 bbox 面积的比例 ----
            # 将合并后的点云转到局部坐标计算
            obj_local_c = obj_pts[:, :3] - gt_c_lidar
            cos_y2, sin_y2 = math.cos(-yaw), math.sin(-yaw)
            obj_x = cos_y2 * obj_local_c[:, 0] - sin_y2 * obj_local_c[:, 1]
            obj_y = sin_y2 * obj_local_c[:, 0] + cos_y2 * obj_local_c[:, 1]
            obj_z = obj_local_c[:, 2]
            obj_local = np.column_stack([obj_x, obj_y, obj_z])
            face_cov, max_face_idx = compute_face_coverage(
                obj_local, np.array([w, length, h], dtype=np.float32))

            # ---- 距离过滤 ----
            avg_dist = np.linalg.norm(gt_c_lidar[:2])
            if avg_dist > self.max_dist:
                continue

            # ---- 采样 ----
            if len(obj_pts) > self.num_points:
                idx_pts = np.random.choice(len(obj_pts), self.num_points, replace=False)
            else:
                idx_pts = np.random.choice(len(obj_pts), self.num_points, replace=True)
            obj_pts = obj_pts[idx_pts].astype(np.float32)

            # ---- RGB Crop ----
            x1, y1, x2, y2 = best_det['bbox'].astype(int)
            rgb_crop = img[y1:y2, x1:x2]
            if rgb_crop.size == 0:
                continue
            rgb_crop = cv2.resize(rgb_crop, (self.crop_size, self.crop_size))
            rgb_crop = (rgb_crop[:, :, ::-1] / 255.0).astype(np.float32)
            rgb_crop = np.transpose(rgb_crop, (2, 0, 1))

            h_img, w_img = img.shape[:2]
            bw, bh = (x2 - x1) / w_img, (y2 - y1) / h_img
            aspect = np.clip(bw / (bh + 1e-6), 0.1, 10.0)
            bbox_feat = np.array([bw, bh, aspect, bw * bh], dtype=np.float32)

            samples.append({
                'points': obj_pts,
                'rgb': rgb_crop,
                'gt_center': gt_c_lidar.astype(np.float32),
                'gt_size': np.array([w, length, h], dtype=np.float32),
                'gt_yaw': np.float32(yaw),
                'class_id': det_class_id,
                'bbox_feat': bbox_feat,
                'forward_label': np.float32(-1.0),
                'face_cov': face_cov.astype(np.float32),      # (6,) 每面覆盖率
                'max_face_idx': np.int64(max_face_idx),       # 最大覆盖面的索引
            })

        return samples

    # ==========================================================================
    # Dataset 接口 (轻量: 只做增强 + 构建 tensor)
    # ==========================================================================

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        """从内存缓存加载, 仅应用随机 Z 轴旋转增强 + 构建 target tensor."""
        samples = []
        for raw in self._cache[idx]:
            pts = raw['points'].copy()
            center = raw['gt_center'].copy()
            yaw = float(raw['gt_yaw'])

            # ---- 数据增强 (随机 Z 轴旋转) ----
            if self.use_augmentation and np.random.rand() > 0.5:
                angle = np.random.uniform(-np.pi, np.pi)
                rot = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle),  np.cos(angle), 0],
                    [0, 0, 1],
                ], dtype=np.float32)
                pts = pts @ rot.T
                center = center @ rot.T
                yaw += angle

            # ---- 显式极值 ----
            xyz_min = pts.min(axis=0)
            xyz_max = pts.max(axis=0)

            # ---- 构建 target: [dx,dy,dz, δl,δw,δh, cos2θ,sin2θ] ----
            centroid = pts.mean(axis=0)
            d_center = (center - centroid) / 3.0                  # (3,)  center_scale=3
            # 尺寸对数残差 (避免 log(0))
            prior_size = np.array([1.9, 4.6, 1.5])  # fallback (car)
            cls_prior = {0: [0.70, 0.70, 1.70], 1: [0.70, 0.70, 1.70],
                         2: [1.90, 4.60, 1.50], 3: [2.50, 6.50, 2.80],
                         4: [2.80, 10.5, 3.20], 5: [3.00, 15.0, 4.00],
                         6: [0.70, 2.00, 1.50], 7: [0.60, 1.80, 1.30],
                         8: [0.30, 0.30, 0.80], 9: [0.30, 0.30, 0.80]}
            prior_size = np.array(cls_prior.get(raw['class_id'], [1.5, 2.0, 4.5]),
                                  dtype=np.float32)
            d_size = np.log(np.maximum(raw['gt_size'], 0.01) / prior_size)  # (3,)
            yaw_2theta = np.array([np.cos(2 * yaw), np.sin(2 * yaw)],
                                  dtype=np.float32)
            target = np.concatenate([d_center, d_size, yaw_2theta]).astype(np.float32)

            samples.append({
                'points': torch.from_numpy(pts).float(),
                'rgb': torch.from_numpy(raw['rgb']).float(),
                'xyz_min': torch.from_numpy(xyz_min).float(),
                'xyz_max': torch.from_numpy(xyz_max).float(),
                'target': torch.from_numpy(target).float(),
                'class_id': raw['class_id'],
                'bbox_feat': torch.from_numpy(raw['bbox_feat']).float(),
                'forward_label': torch.tensor(raw['forward_label']).float(),
                'face_cov': torch.from_numpy(raw['face_cov']).float(),       # (6,)
                'max_face_idx': torch.tensor(raw['max_face_idx']).long(),    # int
            })

        return samples

    def _load_image(self, sample):
        """加载 CAM_FRONT 图像."""
        sd_token = sample['data']['CAM_FRONT']
        filepath = self.nusc.get_sample_data_path(sd_token)
        return cv2.imread(filepath)


# ==============================================================================
# Collate Function
# ==============================================================================

def phase3_collate(batch):
    """展平 per-frame 样本列表并堆叠为 batch tensor.

    Args:
        batch: list[list[dict]] — 每个元素是一个帧的样本列表

    Returns:
        dict 或 None (所有帧都无有效样本时)
    """
    all_points, all_rgb, all_min, all_max, all_targets, all_class_ids, all_bbox, all_fwd = [], [], [], [], [], [], [], []
    all_face_cov, all_max_face = [], []
    for frame_samples in batch:
        for s in frame_samples:
            all_points.append(s['points'])
            all_rgb.append(s['rgb'])
            all_min.append(s['xyz_min'])
            all_max.append(s['xyz_max'])
            all_targets.append(s['target'])
            all_class_ids.append(s['class_id'])
            all_bbox.append(s['bbox_feat'])
            all_fwd.append(s['forward_label'])
            all_face_cov.append(s.get('face_cov', torch.zeros(6)))
            all_max_face.append(s.get('max_face_idx', torch.tensor(0)))

    if len(all_points) == 0:
        return None

    return {
        'points': torch.stack(all_points, 0),
        'rgb': torch.stack(all_rgb, 0),
        'xyz_min': torch.stack(all_min, 0),
        'xyz_max': torch.stack(all_max, 0),
        'target': torch.stack(all_targets, 0),
        'class_ids': torch.tensor(all_class_ids, dtype=torch.long),
        'bbox_feat': torch.stack(all_bbox, 0),
        'forward_label': torch.stack(all_fwd, 0),
        'face_cov': torch.stack(all_face_cov, 0),              # (B, 6)
        'max_face_idx': torch.stack(all_max_face, 0),           # (B,)
    }
