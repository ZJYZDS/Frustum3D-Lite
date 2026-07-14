"""
纯推理管线: YOLO检测 → frustum裁剪 → ROR去噪 → DBSCAN聚类 → 模型预测.

不依赖 GT bbox, 模拟真实部署场景.
"""

import math
import numpy as np


def pipeline_predict(model, pts_lidar, dets, K, T_lidar2cam, device,
                     num_points=512, min_points=5, class_names=None):
    """对一帧的所有 YOLO detection 执行完整的 frustum→净化→模型 管线.

    Args:
        model: PointNet3DDetector (eval mode)
        pts_lidar: (M, 3) LiDAR 点云 (已聚合+去地面, LiDAR 帧)
        dets: list[dict], YOLO 检测结果, 每个 dict 包含 bbox, class_id, conf
        K: (3,3) 相机内参
        T_lidar2cam: (3,4) LiDAR→Camera 变换
        device: torch.device
        num_points: 模型输入点数
        min_points: 最少点数阈值 (不足则跳过)
        class_names: dict (可选) class_id → name

    Returns:
        list[dict]: 每个有效检测的预测结果
          {center, size, yaw, class_id, class_name, conf, num_pts, yaw_norm}
    """
    from src.dataset_phase3 import (
        filter_points_by_frustum,
        remove_statistical_outliers,
        extract_largest_cluster,
    )
    import torch

    if class_names is None:
        class_names = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
                       4: 'bus', 6: 'motorcycle', 7: 'bicycle'}

    # 行人类别: 跳过模型 yaw, 用 PCA 兜底
    SKIP_YAW_CLASSES = {0, 1}

    CENTER_SCALE = 3.0
    SIZE_SCALE = 5.0
    SIZE_EXPAND = 1.12  # 膨胀 12% 确保 bbox 包住所有可见点云

    predictions = []

    for det in dets:
        bbox = det['bbox'].copy()  # (x1, y1, x2, y2)
        cls_id = det['class_id']
        cls_name = class_names.get(cls_id, f'cls_{cls_id}')

        # ---- Step 1: Frustum 裁剪 ----
        frustum_pts, margin = filter_points_by_frustum(
            pts_lidar, bbox, K, T_lidar2cam, margin='auto')

        if len(frustum_pts) < min_points:
            continue

        # ---- Step 2: ROR 去噪 ----
        ror_pts = remove_statistical_outliers(frustum_pts,
                                              nb_neighbors=20, std_ratio=2.0)

        if len(ror_pts) < min_points:
            ror_pts = frustum_pts  # 回退到未去噪的

        # ---- Step 3: DBSCAN 取最大簇 ----
        cluster_pts = extract_largest_cluster(ror_pts, eps=0.6, min_samples=8)

        if len(cluster_pts) < min_points:
            cluster_pts = ror_pts  # 回退

        # ---- Step 4: 采样 ----
        n_pts = len(cluster_pts)
        if n_pts > num_points:
            idx = np.random.choice(n_pts, num_points, replace=False)
        else:
            idx = np.random.choice(n_pts, num_points, replace=True)
        pts_sampled = cluster_pts[idx].astype(np.float32)

        # ---- Step 5: 模型推理 ----
        pts_tensor = torch.from_numpy(pts_sampled).unsqueeze(0).to(device)
        cid_tensor = torch.tensor([cls_id], dtype=torch.long).to(device)

        with torch.no_grad():
            out = model(points=pts_tensor, class_ids=cid_tensor)

        centroid = pts_sampled.mean(axis=0)
        prior = model.prior_table[cls_id].cpu().numpy()

        d_center = out[0, :3].cpu().numpy()
        d_size = out[0, 3:6].cpu().numpy()
        u, v = float(out[0, 6]), float(out[0, 7])
        yaw_norm = math.sqrt(u**2 + v**2 + 1e-8)

        # 解码 center / size
        center = centroid + d_center * CENTER_SCALE
        size = prior * np.exp(d_size) * SIZE_EXPAND

        # 解码 yaw
        if cls_id in SKIP_YAW_CLASSES:
            # 行人: PCA 兜底
            yaw, _ = _pca_yaw(pts_sampled)
        elif yaw_norm < 0.15:
            # 低置信 car: PCA 兜底
            yaw, conf = _pca_yaw(pts_sampled)
            if conf < 1.2:
                yaw = 0.5 * math.atan2(v / yaw_norm, u / yaw_norm)
                if yaw < 0:
                    yaw += math.pi
        else:
            yaw = 0.5 * math.atan2(v / yaw_norm, u / yaw_norm)
            if yaw < 0:
                yaw += math.pi

        predictions.append({
            'center': center,
            'size': size,
            'yaw': yaw,
            'yaw_norm': yaw_norm,
            'class_id': cls_id,
            'class_name': cls_name,
            'conf': det.get('conf', 1.0),
            'num_pts': n_pts,
            'centroid': centroid,
            'bbox': bbox,   # YOLO 2D detection box (x1,y1,x2,y2)
        })

    return predictions


def _pca_yaw(pts):
    """PCA 估计 XY 平面主方向 → yaw ∈ [0, π). 返回 (yaw, confidence)."""
    if len(pts) < 5:
        return 0.0, 0.0
    xy = pts[:, :2]
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    yaw = math.atan2(principal[1], principal[0])
    if yaw < 0:
        yaw += math.pi
    conf = eigvals[-1] / (eigvals[-2] + 1e-8)
    return yaw, conf


def pipeline_predict_with_gt(points_list, model, device,
                              num_points=512, class_names=None):
    """使用 GT-bbox 内部点做预测 (对比用, 同训练管线).

    Args:
        points_list: list[dict] 同 Phase3Dataset __getitem__ 返回格式
           每个 dict 含 points, class_id
        model: PointNet3DDetector (eval mode)
        device: torch.device
        num_points: 模型输入点数
        class_names: dict (可选)

    Returns:
        list[dict]: 预测结果
    """
    import torch
    if class_names is None:
        class_names = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck',
                       4: 'bus', 6: 'motorcycle', 7: 'bicycle'}

    SKIP_YAW_CLASSES = {0, 1}
    CENTER_SCALE = 3.0
    SIZE_EXPAND = 1.12

    predictions = []

    for s in points_list:
        pts = s['points']
        cls_id = s['class_id']

        pts_tensor = pts.unsqueeze(0).to(device) if isinstance(pts, torch.Tensor) \
            else torch.from_numpy(pts).unsqueeze(0).float().to(device)
        cid_tensor = torch.tensor([cls_id], dtype=torch.long).to(device)

        with torch.no_grad():
            out = model(points=pts_tensor, class_ids=cid_tensor)

        pts_np = pts if isinstance(pts, np.ndarray) else pts.cpu().numpy()
        centroid = pts_np.mean(axis=0)
        prior = model.prior_table[cls_id].cpu().numpy()

        d_center = out[0, :3].cpu().numpy()
        d_size = out[0, 3:6].cpu().numpy()
        u, v = float(out[0, 6]), float(out[0, 7])
        yaw_norm = math.sqrt(u**2 + v**2 + 1e-8)

        center = centroid + d_center * CENTER_SCALE
        size = prior * np.exp(d_size) * SIZE_EXPAND

        if cls_id in SKIP_YAW_CLASSES:
            yaw, _ = _pca_yaw(pts_np)
        elif yaw_norm < 0.15:
            yaw, conf = _pca_yaw(pts_np)
            if conf < 1.2:
                yaw = 0.5 * math.atan2(v / yaw_norm, u / yaw_norm)
                if yaw < 0: yaw += math.pi
        else:
            yaw = 0.5 * math.atan2(v / yaw_norm, u / yaw_norm)
            if yaw < 0: yaw += math.pi

        predictions.append({
            'center': center,
            'size': size,
            'yaw': yaw,
            'yaw_norm': yaw_norm,
            'class_id': cls_id,
            'class_name': class_names.get(cls_id, f'cls_{cls_id}'),
            'conf': 1.0,
            'num_pts': len(pts_np),
            'centroid': centroid,
        })

    return predictions
