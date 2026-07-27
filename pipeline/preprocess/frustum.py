"""Frustum-based point cloud cropping and face coverage computation."""

import numpy as np

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


def filter_points_by_bbox_projection(pts_lidar, bbox, K, T_lidar2cam, depth_range=(0.5, 55.0)):
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
"""
only 4 个 侧面face 的 pointcloud coverage (0,1) -> key []
"""
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