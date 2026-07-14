"""
2D-Guided 3D 初始估计器: 用 2D YOLO + 深度图 + 相机几何 直接算出高质量的 3D 框.

三个策略:
  1. Center: 2D bbox 底部中心 + 深度穿透 → 反投影到 3D (替代点云均值)
  2. Size:   2D 像素宽高 + 深度 → 物理尺寸 (替代类别默认值)
  3. Yaw:    PCA 主方向 + 2D 朝向分类器消歧义 (替代纯 PCA)

设计原则: 初始框误差应 < 0.5m / < 5°, 这样现有的 PointNet++ refiner
(训练时只见过 GT+noise) 可以直接用作精调器, 无需重新训练.
"""

import math
import numpy as np
import cv2


def estimate_center_from_2d(det_bbox, depth_map, uv_map, valid_proj, K,
                             T_lidar2cam=None, lidar_points_3d=None):
    """策略 1: 2D bbox 底部区域 LiDAR 点 → 3D center.

    用 bbox 底部 30% 区域的 LiDAR 点求均值. 物体底部 (轮胎) 的点最密集且可靠.
    比全框均值好 (排除背景/上部噪声), 比像素级深度反投影好 (LiDAR 太稀疏).

    Args:
        det_bbox: [x1, y1, x2, y2]
        lidar_points_3d: (N, 3) LiDAR 点云 3D 坐标 (LiDAR 帧). 如果提供, 直接取均值.
        uv_map, valid_proj, depth_map: 用于区域筛选
    Returns:
        (center_3d, depth, success)  — center 在 LiDAR 帧
    """
    x1, y1, x2, y2 = det_bbox.astype(int)

    # 底部 30% 区域
    v_cut = y2 - max(1, int((y2 - y1) * 0.35))
    in_bottom = (valid_proj &
                 (uv_map[:, 0] >= x1) & (uv_map[:, 0] <= x2) &
                 (uv_map[:, 1] >= v_cut) & (uv_map[:, 1] <= y2))

    if in_bottom.sum() < 5:
        # fallback: 整框底部 50%
        v_cut = y2 - max(1, int((y2 - y1) * 0.5))
        in_bottom = (valid_proj &
                     (uv_map[:, 0] >= x1) & (uv_map[:, 0] <= x2) &
                     (uv_map[:, 1] >= v_cut) & (uv_map[:, 1] <= y2))

    if in_bottom.sum() < 3:
        # last fallback: 整框
        in_bottom = (valid_proj &
                     (uv_map[:, 0] >= x1) & (uv_map[:, 0] <= x2) &
                     (uv_map[:, 1] >= y1) & (uv_map[:, 1] <= y2))

    if in_bottom.sum() < 1:
        return None, 0.0, False

    # 直接在 LiDAR 帧取均值 (避免坐标系转换误差)
    if lidar_points_3d is not None and len(lidar_points_3d) == len(in_bottom):
        center = lidar_points_3d[in_bottom].mean(axis=0).astype(np.float32)
        return center, float(depth_map[in_bottom].mean()), True

    # 如果有 T_lidar2cam, 反投影均值
    depths = depth_map[in_bottom]
    uv_bottom = uv_map[in_bottom]
    depth = float(np.median(depths))
    u_mean = uv_bottom[:, 0].mean()
    v_mean = uv_bottom[:, 1].mean()

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X_cam = (u_mean - cx) * depth / fx
    Y_cam = (v_mean - cy) * depth / fy
    center_cam = np.array([X_cam, Y_cam, depth], dtype=np.float32)

    if T_lidar2cam is not None:
        R = T_lidar2cam[:3, :3]
        t = T_lidar2cam[:3, 3]
        center_lidar = R.T @ (center_cam - t)
        return center_lidar.astype(np.float32), depth, True

    return center_cam, depth, True


def estimate_center_direct(obj_pts_lidar, uv_map, valid_proj, det_bbox):
    """策略 1 简化版: 直接在 LiDAR 帧取 bbox 底部区域点云均值.

    这是最稳健的方法: 不依赖深度图精度, 直接操作已有的 3D 点.
    """
    x1, y1, x2, y2 = det_bbox.astype(int)
    v_cut = y2 - max(1, int((y2 - y1) * 0.35))
    in_bottom = (valid_proj &
                 (uv_map[:, 0] >= x1) & (uv_map[:, 0] <= x2) &
                 (uv_map[:, 1] >= v_cut) & (uv_map[:, 1] <= y2))

    if in_bottom.sum() < 5:
        v_cut = y2 - max(1, int((y2 - y1) * 0.5))
        in_bottom = (valid_proj &
                     (uv_map[:, 0] >= x1) & (uv_map[:, 0] <= x2) &
                     (uv_map[:, 1] >= v_cut) & (uv_map[:, 1] <= y2))
    if in_bottom.sum() < 3:
        in_bottom = (valid_proj &
                     (uv_map[:, 0] >= x1) & (uv_map[:, 0] <= x2) &
                     (uv_map[:, 1] >= y1) & (uv_map[:, 1] <= y2))
    if in_bottom.sum() < 1:
        return None, False

    center = obj_pts_lidar[in_bottom].mean(axis=0).astype(np.float32)
    return center, True

    # 反投影: camera 帧 3D
    X_cam = (u_center - cx) * depth / fx
    Y_cam = (v_bottom - cy) * depth / fy
    Z_cam = depth
    center_cam = np.array([X_cam, Y_cam, Z_cam], dtype=np.float32)

    if T_lidar2cam is not None:
        # camera → LiDAR: pt_cam = R @ pt_lidar + t → pt_lidar = R.T @ (pt_cam - t)
        R = T_lidar2cam[:3, :3]
        t = T_lidar2cam[:3, 3]
        center_lidar = R.T @ (center_cam - t)
        return center_lidar.astype(np.float32), depth, True

    return center_cam, depth, True


def estimate_size_from_2d(det_bbox, depth, K, cls_id=None):
    """策略 3: 2D 像素宽高 + 深度 → 物理尺寸.

    width_physical  ≈ bbox_width_px  * depth / fx
    height_physical ≈ bbox_height_px * depth / fy
    length_physical ≈ width_physical (大多数车辆宽≈长, 或按类别比例调整)

    Args:
        det_bbox: [x1, y1, x2, y2]
        depth: float 物体深度 (m)
        K: (3,3) 相机内参
        cls_id: int 可选类别 ID, 用于宽/长比例调整

    Returns:
        size: (3,) [width, length, height] nuScenes 格式
    """
    x1, y1, x2, y2 = det_bbox
    bbox_w_px = x2 - x1
    bbox_h_px = y2 - y1
    fx, fy = K[0, 0], K[1, 1]

    # 物理宽度 (由像素宽度 + 深度反算)
    phys_w = bbox_w_px * depth / fx
    # 物理高度
    phys_h = bbox_h_px * depth / fy

    # 长度: 从 2D 无法直接观测 (深度方向被压缩), 按类别比例推断
    # 典型 w/l 比: car≈0.45, truck≈0.40, bus≈0.28, person≈1.0
    wl_ratios = {0: 1.0, 1: 0.28, 2: 0.45, 3: 0.36, 5: 0.28, 7: 0.40}
    wl_ratio = wl_ratios.get(cls_id, 0.45) if cls_id is not None else 0.45
    phys_l = phys_w / max(wl_ratio, 0.2)

    # 钳制在合理物理范围内
    phys_w = np.clip(phys_w, 0.3, 4.0)
    phys_l = np.clip(phys_l, 0.5, 15.0)
    phys_h = np.clip(phys_h, 0.3, 5.0)

    return np.array([phys_w, phys_l, phys_h], dtype=np.float32)


def estimate_yaw_pca(obj_xyz, cls_id=None):
    """PCA 主方向估计 (xy 平面), 返回 [0, π) 范围的角度.

    此函数只给出"轴线方向", 不区分正反 (180° 歧义).
    需要配合 2D 朝向分类器消歧义.

    Args:
        obj_xyz: (N, 3) LiDAR 点云
    Returns:
        yaw_rad: float PCA 主方向角 [0, π)
        eigval_ratio: float 特征值比 (越大=越细长=方向越可信)
    """
    centered = obj_xyz[:, :2] - obj_xyz[:, :2].mean(axis=0)
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]  # 最大特征值对应的向量
    angle = math.atan2(principal[1], principal[0])

    # 归一化到 [0, π)
    angle = angle % math.pi

    eigval_ratio = abs(eigvals[1] - eigvals[0]) / (eigvals[0] + eigvals[1] + 1e-6)

    return float(angle), float(eigval_ratio)


def estimate_yaw_plane_fitting(obj_xyz, pred_size=None, dist_thresh=0.15):
    """面拟合 → 匹配 bbox 面 → yaw 轴线 [0, π).

    1. RANSAC 拟合最多 2 个平面, 跳过地面
    2. 对每个垂直面: 测量面内 extent (宽×高)
    3. 匹配到 bbox 面:
        侧面:  extent ≈ (l_pred, h_pred) → yaw ⊥ normal
        前后面: extent ≈ (w_pred, h_pred) → yaw ∥ normal
    4. 若无 pred_size, fallback 用 extent ratio 判断
    5. 选点数最多的垂直面

    Args:
        obj_xyz: (N, 3) LiDAR 点云
        pred_size: (3,) 预测 bbox 尺寸 (w, l, h), 可选
        dist_thresh: RANSAC 距离阈值 (m)

    Returns:
        yaw_rad: float  [0, π)
        n_inliers: int
        face_center: (3,)  可见面中心点 (LiDAR 帧)
        face_normal: (3,)  面法向量 (指向物体外侧)
        dim_along_normal: float  法向量方向的 bbox 尺寸 (m), 用于 center 修正
    """
    import open3d as o3d

    remaining = obj_xyz[:, :3].copy()
    best_yaw, best_n_pts = None, 0
    best_face_normal = None
    best_dim_normal = None
    best_inlier_pts = None

    # 如果有预测尺寸, 用于面匹配
    w_pred, l_pred, h_pred = pred_size if pred_size is not None else (None, None, None)

    for _ in range(2):
        if len(remaining) < 15:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining.astype(np.float64))

        try:
            m, inliers = pcd.segment_plane(dist_thresh, 3, 200)
        except Exception:
            break

        inlier_set = set(inliers)
        if len(inlier_set) < 10:
            break

        a, b, c = float(m[0]), float(m[1]), float(m[2])
        n = np.array([a, b, c], dtype=np.float64)
        n /= np.linalg.norm(n) + 1e-10

        # 跳过地面
        if abs(c) > 0.65:
            remaining = np.delete(remaining, list(inlier_set), axis=0)
            continue

        # 垂直面 — 测量面内 extent
        inlier_pts = remaining[list(inlier_set)]
        n_xy = n[:2] / (np.linalg.norm(n[:2]) + 1e-10)

        # 面内两个方向: 水平(垂直于法向的水平方向), 垂直(z)
        horiz_dir = np.array([-n_xy[1], n_xy[0], 0.0])  # 水平面内方向
        vert_dir = np.array([0.0, 0.0, 1.0])              # 垂直方向

        centered = inlier_pts - inlier_pts.mean(0)
        ext_horiz = np.abs(centered @ horiz_dir).ptp()    # 面内水平 extent
        ext_vert = np.abs(centered @ vert_dir).ptp()       # 面内垂直 extent

        # ---- 匹配 bbox 面 ----
        if w_pred is not None and l_pred is not None and h_pred is not None:
            # 侧面误差: (ext_horiz - l)^2 + (ext_vert - h)^2
            err_side = (ext_horiz - l_pred)**2 + (ext_vert - h_pred)**2
            # 正面误差: (ext_horiz - w)^2 + (ext_vert - h)^2
            err_front = (ext_horiz - w_pred)**2 + (ext_vert - h_pred)**2

            if err_side < err_front:
                yaw = math.atan2(-n_xy[0], n_xy[1]) % math.pi   # yaw ⊥ normal
            else:
                yaw = math.atan2(n_xy[1], n_xy[0]) % math.pi    # yaw ∥ normal
        else:
            # Fallback: extent ratio
            along_n = np.abs(centered[:, :2] @ n_xy)
            perp_n = np.abs(centered[:, :2] @ np.array([-n_xy[1], n_xy[0]]))
            if perp_n.ptp() > along_n.ptp() * 1.2:
                yaw = math.atan2(-n_xy[0], n_xy[1]) % math.pi
            else:
                yaw = math.atan2(n_xy[1], n_xy[0]) % math.pi

        n_pts = len(inlier_set)
        if n_pts > best_n_pts:
            best_n_pts = n_pts
            best_yaw = yaw
            best_face_normal = n.copy()
            best_inlier_pts = inlier_pts.copy()
            if err_side < err_front:
                best_dim_normal = w_pred if w_pred is not None else ext_vert
            else:
                best_dim_normal = l_pred if l_pred is not None else ext_horiz

        remaining = np.delete(remaining, list(inlier_set), axis=0)

    if best_yaw is None:
        best_yaw, _ = estimate_yaw_pca(obj_xyz)
        best_n_pts = 0
        best_face_normal = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        best_dim_normal = 0.0
        best_inlier_pts = np.zeros((0, 3), dtype=np.float32)

    return (float(best_yaw), int(best_n_pts),
            best_face_normal.astype(np.float32),
            float(best_dim_normal),
            best_inlier_pts.astype(np.float32))


def compute_center_correction(pred_center, face_normal, dim_normal,
                               inlier_pts, total_pts):
    """让 bbox 面贴在点云表面: 沿法向量方向, 将面推到点云最外侧.

    点云 = 物体表面 → bbox 面应刚好包住点云.
    沿法向量方向: 面位置 = max(点云投影), 中心 = 面位置 - dim/2.

    Args:
        pred_center: (3,) 模型预测 center
        face_normal: (3,) 面法向量
        dim_normal: float  法向量方向 bbox 尺寸
        inlier_pts: (N,3)  面内点
        total_pts: int  总点数 (用于算权重)

    Returns:
        corrected_center: (3,) 修正后 center
        correction: (3,) 修正向量 (调试用)
    """
    n_inliers = len(inlier_pts)
    if n_inliers < 10 or dim_normal <= 0:
        return pred_center, np.zeros(3, dtype=np.float32)

    # 法向量指向传感器 (确保方向: face→origin)
    face_center = inlier_pts.mean(0)
    n = face_normal.copy()
    if np.dot(n, -face_center) < 0:
        n = -n

    # 点云沿法向量投影, 取 P95 (抗离群)
    proj = np.dot(inlier_pts, n)
    face_pos = np.percentile(proj, 95)

    # 中心沿法向量 = 面位置 - dim/2
    center_n = np.dot(pred_center, n)
    target_center_n = face_pos - dim_normal / 2.0
    correction_n = target_center_n - center_n

    # 钳制 + 加权
    max_shift = min(dim_normal * 0.3, 1.5)
    correction_n = np.clip(correction_n, -max_shift, max_shift)
    weight = min(n_inliers / max(total_pts, 1), 0.3)

    correction = n * correction_n * weight
    corrected = pred_center + correction
    return corrected.astype(np.float32), correction.astype(np.float32)


def disambiguate_yaw_2d(pca_yaw, bbox_aspect, cls_id=None):
    """用 2D bbox 宽高比消歧义 PCA 的 180° 翻转 (简化版, 无分类器).

    启发式规则:
      - 车辆侧面: bbox 宽>高 → yaw 接近 ±90° (垂直于相机视线)
      - 车辆前/后面: bbox 高>宽 → yaw 接近 0° 或 180° (沿相机视线)

    PCA 给出轴方向 α ∈ [0, π). 我们要决定: α 还是 α+π?
    对于 CAM_FRONT (相机看 +x):
      - yaw≈0°:  车尾朝向相机 (车向前开, 与 ego 同向)
      - yaw≈±90°: 车侧面朝向相机
      - yaw≈±180°: 车头朝向相机 (对向来车)

    规则:
      - 如果 bbox 明显宽 (w/h > 1.3) → 侧面视角 → yaw ≈ ±90°
        选择 α 或 α+π 中更接近 90° 的那个
      - 如果 bbox 明显窄 (w/h < 0.7) → 前/后面视角 → yaw ≈ 0° 或 180°
        选择 α 或 α+π 中更接近 0° 的那个

    Args:
        pca_yaw: float PCA 角度 [0, π)
        bbox_aspect: float 2D bbox 宽高比 (w/h)
    Returns:
        yaw_rad: float 消歧义后的 yaw [-π, π)
    """
    w_over_h = bbox_aspect

    # 两个候选方向
    yaw_a = pca_yaw            # [0, π)
    yaw_b = pca_yaw - math.pi  # ≈ yaw_a + π, 映射到 [-π, 0)

    if w_over_h > 1.2:
        # 侧面: yaw 应接近 ±90° (π/2 或 -π/2)
        target = math.pi / 2
        dist_a = min(abs(yaw_a - target), abs(yaw_a + target))
        dist_b = min(abs(yaw_b - target), abs(yaw_b + target))
        return yaw_a if dist_a <= dist_b else yaw_b
    elif w_over_h < 0.8:
        # 前/后面: yaw 应接近 0° 或 ±180° (即 0 或 π/-π)
        dist_a = min(abs(yaw_a), abs(yaw_a - math.pi), abs(yaw_a + math.pi))
        dist_b = min(abs(yaw_b), abs(yaw_b - math.pi), abs(yaw_b + math.pi))
        return yaw_a if dist_a <= dist_b else yaw_b
    else:
        # 接近正方形, 方向不确定, 保持 PCA 原始
        return yaw_a


# 各类别典型尺寸 (宽, 长, 高) — nuScenes 均值
CLS_AVG_SIZE = {
    0: np.array([0.7, 0.7, 1.75], dtype=np.float32),      # person
    1: np.array([0.5, 1.8, 1.2], dtype=np.float32),       # bicycle
    2: np.array([2.0, 4.5, 1.6], dtype=np.float32),       # car
    3: np.array([0.8, 2.2, 1.5], dtype=np.float32),       # motorcycle
    5: np.array([2.8, 10.0, 3.0], dtype=np.float32),      # bus
    7: np.array([2.8, 7.0, 2.5], dtype=np.float32),       # truck
}


def filter_points_by_mask(uv, valid_proj, depth, seg_mask, lidar_pts, min_depth=0.5,
                          dilate=3):
    """用 YOLO-seg 像素级 mask 过滤 LiDAR 点: 只保留落在物体轮廓内部的点.

    替代 bbox margin 裁剪: bbox 内包含地面/背景点 (~60% 噪声),
    而 mask 是物体精确轮廓, 过滤后的点云几乎全是前景点.

    Args:
        uv: (N, 2) LiDAR 点投影到图像的像素坐标 (float).
        valid_proj: (N,) 有效投影 mask.
        depth: (N,) LiDAR 点深度.
        seg_mask: (H, W) YOLO-seg 二值 mask (True=物体区域).
        lidar_pts: (N, 4+) LiDAR 点云 [x,y,z,intensity,...].
        min_depth: 最小深度过滤 (米).
        dilate: mask 膨胀像素数 (补偿投影对齐误差), 默认 3.

    Returns:
        filtered_pts: (M, 4+) 落在 mask 内的 LiDAR 点, 或 None (点数不足).
    """
    H, W = seg_mask.shape
    u_int = np.round(uv[:, 0]).astype(int)
    v_int = np.round(uv[:, 1]).astype(int)

    # 膨胀 mask 补偿投影对齐微小误差
    if dilate > 0:
        kernel = np.ones((dilate * 2 + 1, dilate * 2 + 1), dtype=np.uint8)
        seg_mask = cv2.dilate(seg_mask.astype(np.uint8), kernel).astype(bool)

    in_bounds = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    in_mask = in_bounds & valid_proj & (depth > min_depth)
    in_mask[in_bounds] &= seg_mask[v_int[in_bounds], u_int[in_bounds]]

    if in_mask.sum() < 10:
        return None

    return lidar_pts[in_mask]


def compute_init_quality(obj_pts, noisy_size, cls_id, cls_avg_sizes=None):
    """计算初始框质量指标 [reliability, size_diff, num_pts_norm]."""
    if cls_avg_sizes is None:
        cls_avg_sizes = CLS_AVG_SIZE
    obj_xyz = obj_pts[:, :3]
    centered = obj_xyz[:, :2] - obj_xyz[:, :2].mean(axis=0)
    cov = centered.T @ centered / max(len(centered), 1)
    eigvals, _ = np.linalg.eigh(cov)
    reliability = abs(eigvals[1] - eigvals[0]) / (eigvals[0] + eigvals[1] + 1e-6)

    cls_avg = cls_avg_sizes.get(cls_id, np.array([2.0, 4.5, 1.6], dtype=np.float32))
    size_diff = np.mean(np.abs(noisy_size - cls_avg) / (cls_avg + 1e-3))
    num_pts_norm = min(1.0, len(obj_pts) / 100.0)

    return np.array([reliability, size_diff, num_pts_norm], dtype=np.float32)
