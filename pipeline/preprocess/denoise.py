"""Point cloud denoising: ROR outlier removal, DBSCAN clustering, multi-sweep aggregation."""

import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

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

def remove_statistical_outliers(points, nb_neighbors=20, std_ratio=2.0):
    """ROR (Radius Outlier Removal): 剔除孤立离群点.

    纯 numpy/scipy 实现, 与 open3d.remove_statistical_outlier 等价:
      1. 对每个点, 找 k=nb_neighbors 个最近邻 (含自身)
      2. 计算每个点的平均邻近距离
      3. 距离 > μ + std_ratio·σ 的点视为离群点, 剔除

    Args:
        points: (N, 3) 点云
        nb_neighbors: 邻居数
        std_ratio: 标准差倍数阈值

    Returns:
        (M, 3) 滤波后点云
    """
    from scipy.spatial import cKDTree

    if len(points) < nb_neighbors + 1:
        return points

    tree = cKDTree(points)
    dists, _ = tree.query(points, k=nb_neighbors)
    mean_dists = dists.mean(axis=1)


    mean_dists = dists.mean(axis=1)

    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    keep = mean_dists <= threshold
    return points[keep]


def extract_largest_cluster(points, eps=0.6, min_samples=4):
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
# 地面去除 (RANSAC)
# ==============================================================================

def remove_ground_ransac(points, distance_threshold=0.25, num_iterations=200):
    """使用 RANSAC 平面拟合去除地面点.

    Args:
        points: (N, 3) 点云 xyz
        distance_threshold: 点到平面距离阈值 (米)
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

    inlier_set = set(inliers)
    mask = np.ones(len(points), dtype=bool)
    mask[list(inlier_set)] = False
    return points[mask]
