# Phase3 感知框架：基于双头 PointNet 的 3D 目标检测

**版本**: 1.0  
**日期**: 2026-07-12  
**状态**: 开发中

---

## 一、设计哲学

### 1.1 核心原则

| 原则 | 说明 |
|------|------|
| **传感器物理对齐** | 标签和输入始终在同一坐标系（LiDAR 帧），消除训练/推理分布偏移 |
| **分治策略** | 几何定位（Center/Size）归 3D，语义消歧（Yaw）归 2D，各司其职 |
| **绝对回归** | 直接输出绝对坐标，避免残差带来的 Domain Gap 问题 |
| **工程解耦** | 所有数据预处理在 Dataset 内部完成，训练脚本只负责 Forward/Loss/Backward |

### 1.2 架构总览

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              Phase3 感知 Pipeline                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │  CAM_FRONT   │    │  LIDAR_TOP   │    │  多帧聚合 (nsweeps=10)            │   │
│  │  (1600x900)  │    │  (N, 3)      │───▶│  + RANSAC 地面去除               │   │
│  └──────┬───────┘    └──────┬───────┘    └───────────────┬──────────────────┘   │
│         │                   │                            │                      │
│         ▼                   │                            ▼                      │
│  ┌──────────────┐           │    ┌──────────────────────────────────────────┐   │
│  │ YOLO-Detect  │           │    │  视锥裁剪 (YOLO 4条射线)                 │   │
│  │ (yolo26s)    │───▶ 2D BBox│    │  + ROR 离群点剔除                       │   │
│  └──────────────┘           │    │  + DBSCAN 最大簇提取                    │   │
│         │                   │    └─────────────────┬────────────────────────┘   │
│         │                   │                      │                          │
│         ▼                   ▼                      ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │ 2D 匹配 GT   │───▶│ 匹配的 GT    │    │  点云 (512, 3) + xyz_min/max     │   │
│  │ (投影中心)   │    │ 标签         │    │  + RGB Crop (3, 128, 128)        │   │
│  └──────────────┘    └──────┬───────┘    └─────────────────┬────────────────┘   │
│                             │                              │                    │
│                             ▼                              ▼                    │
│                    ┌─────────────────────────────────────────────────────┐      │
│                    │            Target (绝对回归)                        │      │
│                    │  [cx, cy, cz, w, h, l, sin(yaw), cos(yaw)]        │      │
│                    └─────────────────────────────────────────────────────┘      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 二、数据预处理管线

### 2.1 输入源

```python
dataset = Phase3Dataset(
    nusc_root="/path/to/nuscenes",      # nuScenes 数据集根目录
    version="v1.0-mini",                 # 数据集版本
    split="train",                       # "train" 或 "val"
    detector_path="models/yolo26s.onnx", # YOLO 检测模型
    nsweeps=10,                          # 多帧聚合帧数
    num_points=512,                      # 点云采样点数
    crop_size=128,                       # RGB Crop 尺寸
    max_dist=15.0,                       # 最大感知距离 (米)
    min_points=5,                        # 最少点数阈值
    match_threshold=80.0,                # 2D 匹配像素阈值
    remove_ground=True,                  # 是否去除地面
    use_augmentation=True,               # 是否启用数据增强
)
```

### 2.2 多帧聚合 (`aggregate_sweeps`)

**目的**: 利用时序信息增加点云密度，解决单帧 LiDAR 稀疏问题。

```
单帧点云: ~30,000 点
聚合 10 帧: ~300,000 点 (通过 ego_pose 配准到当前帧坐标系)
```

**关键操作**:
1. 从当前帧开始，通过 `prev` 指针向前追溯 `nsweeps` 帧
2. 每帧点云先转到全局坐标系，再转到当前帧坐标系
3. 拼接所有点云 → 稠密点云

### 2.3 地面去除 (`remove_ground_ransac`)

**目的**: 在视锥裁剪前去除地面点，防止底部污染和减少计算量。

```python
# 基于 RANSAC 平面拟合
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.25,  # 25cm 垂直误差
    ransac_n=3,
    num_iterations=200
)
# 保留非地面点 (外点)
```

### 2.4 2D→3D 数据关联 (2D 匹配 GT)

**目的**: 确定 YOLO 框对应的 3D GT 标签。

| 步骤 | 操作 |
|------|------|
| 1 | 遍历所有 GT 3D 中心 |
| 2 | 投影到图像平面 (global → ego → camera → pixel) |
| 3 | 计算与 YOLO 框中心的 2D 欧氏距离 |
| 4 | 选择距离最近且 < `match_threshold` (80px) 的 GT |

### 2.5 视锥裁剪 (`filter_points_by_frustum`)

**物理本质**: 利用 YOLO 2D 框的 4 条射线切割 3D 点云。

```
射线 1: 光心 → (x1, y1)    射线 2: 光心 → (x2, y1)
射线 3: 光心 → (x1, y2)    射线 4: 光心 → (x2, y2)
```

```python
# 保留满足以下条件的点:
#   x1 - margin < u < x2 + margin
#   y1 - margin < v < y2 + margin
#   z_cam > 0.5 (相机前方)
```

**效果**: 点云从 50,000 骤降至 100~500 个。

### 2.6 点云净化 (操作后)

| 操作 | 作用 | 参数 |
|------|------|------|
| **ROR** | 剔除孤立离群点 | `nb_neighbors=20, std_ratio=2.0` |
| **DBSCAN** | 提取最大簇 (解决重叠/遮挡) | `eps=0.6, min_samples=8` |

**执行条件**: 点数 > 30 时才执行 DBSCAN，防止稀疏点被误删。

### 2.7 采样与特征提取

```python
# 1. 重采样到固定点数 (512)
if len(obj_pts) > 512:
    idx = np.random.choice(len(obj_pts), 512, replace=False)
else:
    idx = np.random.choice(len(obj_pts), 512, replace=True)  # 重复采样
obj_pts = obj_pts[idx]

# 2. 显式极值 (输入 Head A)
xyz_min = np.min(obj_pts, axis=0)  # (3,)
xyz_max = np.max(obj_pts, axis=0)  # (3,)

# 3. RGB Crop (输入 Head B)
rgb_crop = cv2.resize(img[y1:y2, x1:x2], (128, 128))
rgb_crop = rgb_crop[:, :, ::-1] / 255.0  # BGR → RGB, 归一化
```

### 2.8 标签生成 (绝对回归)

| 分量 | 来源 | 坐标系 |
|------|------|--------|
| `cx, cy, cz` | GT `translation` → `_global_to_lidar()` | LiDAR 帧 |
| `w, h, l` | GT `size` (直接使用) | 物理单位 |
| `sin(yaw), cos(yaw)` | GT `rotation` → `_quaternion_to_yaw_lidar()` | LiDAR 帧 |

---

## 三、模型架构

### 3.1 双头设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DualHeadPointNet                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入点云 (512, 3)  输入 RGB (3,128,128)  输入 xyz_min/max (3,3)           │
│       │                    │                      │                         │
│       ▼                    ▼                      │                         │
│  ┌────────────────┐   ┌─────────────────┐         │                         │
│  │ PointNet       │   │ Lightweight2D   │         │                         │
│  │ Encoder        │   │ Head            │         │                         │
│  │ (共享 MLP      │   │ (2 Conv + GAP)  │         │                         │
│  │  + Max Pool)   │   │ → 32 维         │         │                         │
│  └───────┬────────┘   └────────┬────────┘         │                         │
│          │                     │                  │                         │
│          │  256 维             │ 32 维            │                         │
│          ▼                     ▼                  ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Head A (Center + Size)           Head B (Yaw)                     │   │
│  │  输入: 256 + 6 (极值)             输入: 256 + 32 (2D)               │   │
│  │  → [cx, cy, cz, w, h, l]          → [sin(yaw), cos(yaw)]           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  输出: (B, 8) = [cx, cy, cz, w, h, l, sin, cos]                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 PointNet Encoder

```python
class PointNetEncoder(nn.Module):
    def __init__(self, input_dim=3, feat_dim=256):
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
        )
    def forward(self, x):
        # x: (B, 512, 3)
        features = self.mlp(x)          # (B, 512, 256)
        global_feat = torch.max(features, dim=1)[0]  # (B, 256)
        return global_feat
```

### 3.3 Lightweight2D Head

```python
class Lightweight2DHead(nn.Module):
    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
    def forward(self, x):
        # x: (B, 3, 128, 128)
        return self.conv(x).view(x.size(0), -1)  # (B, 32)
```

### 3.4 回归头

```python
# Head A: Center + Size
self.head_A = nn.Sequential(
    nn.Linear(256 + 6, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Linear(128, 6)
)

# Head B: Yaw
self.head_B = nn.Sequential(
    nn.Linear(256 + 32, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Linear(256, 2)
)
```

### 3.5 损失函数

```python
def bbox_loss(pred, target):
    # Center (除以 50m 归一化)
    loss_center = F.smooth_l1_loss(pred[:, :3] / 50.0, target[:, :3] / 50.0)
    
    # Size (除以 10m 归一化)
    loss_size = F.smooth_l1_loss(pred[:, 3:6] / 10.0, target[:, 3:6] / 10.0)
    
    # Yaw (sin/cos 范围 [-1,1])
    loss_yaw = F.mse_loss(pred[:, 6:], target[:, 6:])
    
    return loss_center + loss_size + loss_yaw
```

---

## 四、文件结构

```
cross_atn_pointNet++/
├── src/
│   ├── dataset_phase3.py       # Phase3 数据集 (主要)
│   ├── dataset_phase1.py       # 基础投影器 (复用)
│   ├── detector.py             # YOLO 检测器
│   ├── fusion.py               # DualHeadPointNet 模型
│   ├── loss.py                 # 损失函数
│   ├── ground_removal.py       # RANSAC 地面去除
│   └── metrics.py              # 评估指标
├── scripts/
│   ├── train_phase3.py         # 训练脚本
│   ├── visualize_phase3.py     # 可视化脚本
│   └── eval_phase3.py          # 评估脚本
├── config/
│   └── phase3.yaml             # 配置文件
├── models/
│   └── yolo26s.onnx            # YOLO 模型
└── checkpoints_phase3/         # 训练保存目录
```

---

## 五、代码实现

### 5.1 核心导入与依赖

```python
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

from src.detector import YOLODetectONNX, OBSTACLE_CLASS_IDS
from src.dataset_phase1 import LiDARProjector
```

### 5.2 多帧聚合

```python
def aggregate_sweeps(nusc, sample_token, channel='LIDAR_TOP', nsweeps=10):
    """聚合多帧 LiDAR 点云到当前帧坐标系"""
    sd_token = sample_token['data'][channel]
    ref_sd = nusc.get('sample_data', sd_token)
    ref_ego = nusc.get('ego_pose', ref_sd['ego_pose_token'])
    ref_to_world = np.eye(4)
    ref_to_world[:3, :3] = Quaternion(ref_ego['rotation']).rotation_matrix
    ref_to_world[:3, 3] = ref_ego['translation']

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
    return LidarPointCloud(combined)
```

### 5.3 地面去除 (RANSAC)

```python
import open3d as o3d

def remove_ground_ransac(points, distance_threshold=0.25, num_iterations=200):
    """使用 RANSAC 去除地面点"""
    if len(points) < 10:
        return points
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations
    )
    
    # 保留非地面点 (外点)
    inlier_set = set(inliers)
    mask = np.ones(len(points), dtype=bool)
    mask[list(inlier_set)] = False
    return points[mask]
```

### 5.4 视锥裁剪

```python
def filter_points_by_frustum(pts, bbox, K, T_lidar2cam, margin=5):
    """用 YOLO BBox 的 4 条射线裁剪点云"""
    x1, y1, x2, y2 = bbox.astype(int)
    pts_cam = (T_lidar2cam[:3, :3] @ pts.T).T + T_lidar2cam[:3, 3]
    valid_z = pts_cam[:, 2] > 0.5
    
    u = (K[0, 0] * pts_cam[:, 0] / pts_cam[:, 2]) + K[0, 2]
    v = (K[1, 1] * pts_cam[:, 1] / pts_cam[:, 2]) + K[1, 2]
    
    mask_bbox = (u > x1 - margin) & (u < x2 + margin) & \
                (v > y1 - margin) & (v < y2 + margin)
    return pts[valid_z & mask_bbox]
```

### 5.5 点云净化

```python
def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """ROR 离群点剔除"""
    if len(points) < 5:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return np.asarray(pcd.select_by_index(ind).points)

def extract_largest_cluster(points, eps=0.6, min_samples=8):
    """DBSCAN 取最大簇 (仅当点数 > 30)"""
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
```

### 5.6 Phase3Dataset 完整实现

```python
class Phase3Dataset(Dataset):
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
                 remove_ground: bool = True,
                 use_augmentation: bool = True):
        
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
        
        self.detector = YOLODetectONNX(detector_path, conf_thresh=0.5)
        self.projector = LiDARProjector(nusc_root)
        
        self._build_frame_list(val_scene_ids)
        self._gt_by_sample = self._group_annotations()
        
        print(f"[Phase3] {split}: {len(self.frames)} frames, nsweeps={nsweeps}")

    def _build_frame_list(self, val_scene_ids):
        """构建有效帧列表"""
        scene_samples = {}
        for sample in self.nusc.sample:
            scene_samples.setdefault(sample['scene_token'], []).append(sample['token'])
        
        scenes = self.nusc.scene
        sorted_scenes = sorted(scenes, key=lambda s: s['name'])
        if self.split == 'train':
            split_scenes = sorted_scenes[:-val_scene_ids] if val_scene_ids > 0 else sorted_scenes
        else:
            split_scenes = sorted_scenes[-val_scene_ids:] if val_scene_ids > 0 else []
        
        self.frames = []
        for scene in split_scenes:
            for sample_token in scene_samples.get(scene['token'], []):
                if 'CAM_FRONT' in sample_token['data'] and 'LIDAR_TOP' in sample_token['data']:
                    self.frames.append(sample_token)

    def _group_annotations(self):
        """按 sample_token 分组 GT 标注"""
        gt_by_sample = {}
        for ann in self.nusc.sample_annotation:
            gt_by_sample.setdefault(ann['sample_token'], []).append(ann)
        return gt_by_sample

    # ---------- 坐标变换辅助 ----------
    def _global_to_lidar(self, point_global, sample_token):
        """global → LiDAR 帧"""
        ego_pose_token = self.nusc.get('sample', sample_token)['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
        t_ego = np.array(ego_pose['translation'])
        pt_ego = R_ego.T @ (np.array(point_global) - t_ego)

        lidar_calib_token = self.nusc.get('sample_data',
                                          sample_token['data']['LIDAR_TOP'])['calibrated_sensor_token']
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_calib_token)
        R_lidar = Quaternion(lidar_calib['rotation']).rotation_matrix
        t_lidar = np.array(lidar_calib['translation'])
        return R_lidar.T @ (pt_ego - t_lidar)

    def _global_to_camera(self, point_global, sample_token):
        """global → camera (用于投影匹配)"""
        ego_pose_token = self.nusc.get('sample', sample_token)['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
        t_ego = np.array(ego_pose['translation'])
        pt_ego = R_ego.T @ (np.array(point_global) - t_ego)

        cam_calib_token = self.projector._sample_sensor_calib.get(sample_token, {}).get('CAM_FRONT')
        if cam_calib_token is None:
            return None
        cam_calib = self.projector.calibs[cam_calib_token]
        R_cam = Quaternion(cam_calib['rotation']).rotation_matrix
        t_cam = np.array(cam_calib['translation'])
        pt_cam = R_cam.T @ (pt_ego - t_cam)
        return pt_cam

    def _quaternion_to_yaw_lidar(self, q, sample_token):
        """四元数 → LiDAR 帧下的 yaw 角"""
        # global → ego
        ego_pose_token = self.nusc.get('sample', sample_token)['ego_pose_token']
        ego_pose = self.nusc.get('ego_pose', ego_pose_token)
        R_ego = Quaternion(ego_pose['rotation']).rotation_matrix
        R_global = Quaternion(q).rotation_matrix
        R_ego_frame = R_ego.T @ R_global

        # ego → LiDAR
        lidar_calib_token = self.nusc.get('sample_data',
                                          sample_token['data']['LIDAR_TOP'])['calibrated_sensor_token']
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_calib_token)
        R_lidar = Quaternion(lidar_calib['rotation']).rotation_matrix
        R_lidar_frame = R_lidar.T @ R_ego_frame
        return math.atan2(R_lidar_frame[1, 0], R_lidar_frame[0, 0])

    # ---------- 核心 __getitem__ ----------
    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        sample_token = self.frames[idx]

        # 1. 加载图像
        img = self._load_image(sample_token)
        if img is None:
            return []

        # 2. YOLO 检测
        dets = self.detector.predict(img)
        dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]
        if not dets:
            return []

        # 3. 多帧聚合点云
        pc = aggregate_sweeps(self.nusc, sample_token, nsweeps=self.nsweeps)
        pts = pc.points[:3, :].T

        # 4. 地面去除 (操作前)
        if self.remove_ground:
            pts = remove_ground_ransac(pts)

        # 5. 投影矩阵
        K, T_lidar2cam, _ = self.projector.get_transform(sample_token)
        if K is None:
            return []

        # 6. GT 标注
        gt_anns = self._gt_by_sample.get(sample_token, [])
        if not gt_anns:
            return []

        # 预计算 GT 投影
        gt_uvs = []
        for ann in gt_anns:
            pt_cam = self._global_to_camera(ann['translation'], sample_token)
            if pt_cam is not None and pt_cam[2] > 0.5:
                u = (K[0,0]*pt_cam[0] / pt_cam[2]) + K[0,2]
                v = (K[1,1]*pt_cam[1] / pt_cam[2]) + K[1,2]
                gt_uvs.append((u, v))
            else:
                gt_uvs.append((None, None))

        samples = []
        for det in dets:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            cx_det, cy_det = (x1+x2)/2, (y1+y2)/2

            # ---- 2D 匹配 GT ----
            best_idx = -1
            best_dist = self.match_threshold
            for i, (u, v) in enumerate(gt_uvs):
                if u is None:
                    continue
                dist = np.sqrt((cx_det - u)**2 + (cy_det - v)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx == -1:
                continue
            ann = gt_anns[best_idx]

            # ---- 视锥裁剪 ----
            obj_pts = filter_points_by_frustum(pts, det['bbox'], K, T_lidar2cam, margin=5)
            if len(obj_pts) < self.min_points:
                continue

            # ---- 点云净化 (操作后) ----
            obj_pts = remove_statistical_outliers(obj_pts)
            if len(obj_pts) < 3:
                continue
            obj_pts = extract_largest_cluster(obj_pts)
            if len(obj_pts) < self.min_points:
                continue

            # ---- 距离过滤 ----
            avg_dist = np.linalg.norm(np.mean(obj_pts[:, :3], axis=0))
            if avg_dist > self.max_dist:
                continue

            # ---- 采样到固定点数 ----
            if len(obj_pts) > self.num_points:
                idx_pts = np.random.choice(len(obj_pts), self.num_points, replace=False)
            else:
                idx_pts = np.random.choice(len(obj_pts), self.num_points, replace=True)
            obj_pts = obj_pts[idx_pts]

            # ---- 显式极值 ----
            xyz_min = np.min(obj_pts, axis=0)
            xyz_max = np.max(obj_pts, axis=0)

            # ---- 标签 (绝对回归) ----
            gt_center_lidar = self._global_to_lidar(ann['translation'], sample_token)
            if gt_center_lidar is None:
                continue
            w, h, l = ann['size']  # [width, height, length]
            yaw = self._quaternion_to_yaw_lidar(ann['rotation'], sample_token)

            target = np.concatenate([
                gt_center_lidar,
                [w, h, l],
                [np.sin(yaw), np.cos(yaw)]
            ]).astype(np.float32)

            # ---- RGB Crop ----
            rgb_crop = img[y1:y2, x1:x2]
            if rgb_crop.size == 0:
                continue
            rgb_crop = cv2.resize(rgb_crop, (self.crop_size, self.crop_size))
            rgb_crop = rgb_crop[:, :, ::-1] / 255.0
            rgb_crop = torch.from_numpy(rgb_crop).permute(2, 0, 1).float()

            # ---- 数据增强 ----
            if self.use_augmentation and np.random.rand() > 0.5:
                angle = np.random.uniform(-np.pi, np.pi)
                rot = np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle),  np.cos(angle), 0],
                    [0, 0, 1]
                ], dtype=np.float32)
                obj_pts = (obj_pts @ rot.T).astype(np.float32)
                xyz_min = np.min(obj_pts, axis=0)
                xyz_max = np.max(obj_pts, axis=0)
                gt_center_lidar = gt_center_lidar @ rot.T
                target[:3] = gt_center_lidar
                yaw += angle
                target[6] = np.sin(yaw)
                target[7] = np.cos(yaw)

            samples.append({
                'points': torch.from_numpy(obj_pts).float(),
                'rgb': rgb_crop,
                'xyz_min': torch.from_numpy(xyz_min).float(),
                'xyz_max': torch.from_numpy(xyz_max).float(),
                'target': torch.from_numpy(target).float()
            })

        return samples

    def _load_image(self, sample_token):
        sd_token = sample_token['data']['CAM_FRONT']
        filepath = self.nusc.get_sample_data_path(sd_token)
        return cv2.imread(filepath)


# ---------- Collate Function ----------
def phase3_collate(batch):
    """展平并堆叠为 batch"""
    all_points, all_rgb, all_min, all_max, all_targets = [], [], [], [], []
    for frame_samples in batch:
        for s in frame_samples:
            all_points.append(s['points'])
            all_rgb.append(s['rgb'])
            all_min.append(s['xyz_min'])
            all_max.append(s['xyz_max'])
            all_targets.append(s['target'])
    
    if len(all_points) == 0:
        return None
    
    return {
        'points': torch.stack(all_points, 0),
        'rgb': torch.stack(all_rgb, 0),
        'xyz_min': torch.stack(all_min, 0),
        'xyz_max': torch.stack(all_max, 0),
        'target': torch.stack(all_targets, 0)
    }
```

### 5.7 训练脚本

```python
# scripts/train_phase3.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset_phase3 import Phase3Dataset, phase3_collate
from src.fusion import DualHeadPointNet
from src.loss import bbox_loss

def train():
    # 数据集
    train_set = Phase3Dataset(
        nusc_root="/path/to/nuscenes",
        split="train",
        detector_path="models/yolo26s.onnx",
        nsweeps=10,
        num_points=512,
        max_dist=15.0,
        remove_ground=True
    )
    val_set = Phase3Dataset(
        nusc_root="/path/to/nuscenes",
        split="val",
        detector_path="models/yolo26s.onnx",
        nsweeps=10,
        num_points=512,
        max_dist=15.0,
        remove_ground=True,
        use_augmentation=False
    )
    
    train_loader = DataLoader(
        train_set, batch_size=8, shuffle=True,
        collate_fn=phase3_collate, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=8, shuffle=False,
        collate_fn=phase3_collate, num_workers=0
    )
    
    # 模型
    model = DualHeadPointNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 训练循环
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in train_loader:
            if batch is None:
                continue
            pred = model(batch['points'], batch['rgb'], batch['xyz_min'], batch['xyz_max'])
            loss = bbox_loss(pred, batch['target'])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")
        scheduler.step()

if __name__ == "__main__":
    train()
```

---

## 六、配置文件

```yaml
# config/phase3.yaml

dataset:
  nsweeps: 10
  num_points: 512
  crop_size: 128
  max_dist: 15.0
  min_points: 5
  match_threshold: 80.0
  remove_ground: true
  use_augmentation: true

model:
  pointnet_dim: 256
  head_A_hidden: 128
  head_B_hidden: 256
  2d_head_channels: 32

training:
  batch_size: 8
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  scheduler: cosine
  warmup_epochs: 5

loss:
  center_scale: 50.0
  size_scale: 10.0
  yaw_scale: 1.0
```

---

## 七、关键参数速查表

| 参数 | 默认值 | 作用 | 调参建议 |
|------|--------|------|----------|
| `nsweeps` | 10 | 多帧聚合数 | 越大点云越密，但加载越慢 |
| `num_points` | 512 | 输入网络点数 | 512 在密度与速度间最优 |
| `max_dist` | 15.0 | 最大感知距离 (m) | 聚焦中近距离，提升精度 |
| `min_points` | 5 | 最少点数阈值 | 过低会引入噪声，过高会丢弃小物体 |
| `match_threshold` | 80 | 2D 匹配像素阈值 | nuScenes 1600x900 下经验值 |
| `remove_ground` | True | 是否去除地面 | 必须开启，否则 center 会漂移 |

---

## 八、验证与调试

### 8.1 快速验证

```python
# 单帧测试
dataset = Phase3Dataset(nusc_root="...", split="train")
for i in range(10):
    samples = dataset[i]
    print(f"Frame {i}: {len(samples)} samples")
    for s in samples:
        print(f"  points: {s['points'].shape}, target: {s['target']}")
```

### 8.2 可视化检查

```python
# 可视化点云与框
import open3d as o3d

def visualize_sample(sample):
    pts = sample['points'].numpy()
    target = sample['target'].numpy()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    
    # 绘制 GT 框
    center = target[:3]
    size = target[3:6]
    yaw = math.atan2(target[6], target[7])
    # ... 绘制 3D 框
    
    o3d.visualization.draw_geometries([pcd, bbox])
```

---

## 九、面试话术

当被问及数据预处理管线时，可简述：

> *"我设计了六阶段级联预处理管线：① 多帧聚合（nsweeps=10）解决 LiDAR 稀疏问题；② RANSAC 地面去除；③ YOLO 2D 框与 GT 投影中心的 2D 匹配；④ 视锥裁剪（4条射线）将点云从 5 万点降至 500 点；⑤ ROR + DBSCAN 双重净化；⑥ 绝对回归标签生成，确保所有数据均在 LiDAR 帧下，消除训练/推理分布偏移。"*