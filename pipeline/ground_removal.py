"""
RANSAC 地面分割: 聚合多帧 LiDAR 点云 → 直接 RANSAC 平面拟合 → 分割地面/非地面.

设计原则:
  - 不做 Z 轴粗筛: 避免"平坦世界"假设导致桥梁/坡道场景误删.
  - min_dist=0.0: 保留近处点, 不因距离过滤丢失贴车头的行人/自行车.
  - 超大点云 (>200k) 用 VoxelGrid 降采样替代 Z 轴截断.
"""

import numpy as np
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


def _multisweep_lidar(nusc, sample, channel='LIDAR_TOP', nsweeps=10, min_dist=0.0):
    """手动多帧 LiDAR 聚合 (不依赖 nuscenes-devkit 的 LidarMultiSweep).

    将所有帧变换到最新帧 (ref_token) 的 ego 坐标系.

    Args:
        nusc: NuScenes 实例.
        sample: 当前 sample dict.
        channel: LiDAR 传感器 channel, 默认 'LIDAR_TOP'.
        nsweeps: 聚合帧数 (含当前帧).
        min_dist: 最小距离过滤 (米). 设 0.0 保留所有点.

    Returns:
        LidarPointCloud: 聚合后的点云 (points: 4×N, 前三行 xyz, 第四行 intensity).
    """
    lidar_token = sample['data'][channel]
    tokens = [lidar_token]
    cur_tk = lidar_token
    while len(tokens) < nsweeps:
        sd_prev = nusc.get('sample_data', cur_tk)
        if sd_prev['prev'] == '':
            break
        tokens.insert(0, sd_prev['prev'])
        cur_tk = sd_prev['prev']

    # 以最新帧为参考帧
    ref_token = tokens[-1]
    sd_ref = nusc.get('sample_data', ref_token)
    ep_ref = nusc.get('ego_pose', sd_ref['ego_pose_token'])
    ref_to_world = np.eye(4)
    ref_to_world[:3, :3] = Quaternion(ep_ref['rotation']).rotation_matrix
    ref_to_world[:3, 3] = ep_ref['translation']

    all_points = []
    for tk in tokens:
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(tk))
        sd = nusc.get('sample_data', tk)
        cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

        # sensor → ego
        pc.rotate(Quaternion(cs['rotation']).rotation_matrix)
        pc.translate(np.array(cs['translation']))

        # 非参考帧: ego → ref_ego
        if tk != ref_token:
            ep = nusc.get('ego_pose', sd['ego_pose_token'])
            cur_to_world = np.eye(4)
            cur_to_world[:3, :3] = Quaternion(ep['rotation']).rotation_matrix
            cur_to_world[:3, 3] = ep['translation']
            T = np.linalg.inv(ref_to_world) @ cur_to_world
            pc.rotate(T[:3, :3])
            pc.translate(T[:3, 3])

        if min_dist > 0:
            pts = pc.points[:3, :]
            mask = np.linalg.norm(pts, axis=0) >= min_dist
            pc.points = pc.points[:, mask]

        all_points.append(pc.points)

    combined = np.hstack(all_points)
    return LidarPointCloud(combined)


def remove_ground(nusc, sample, nsweeps=10, ransac_thresh=0.25,
                  min_dist=0.0, voxel_size=None):
    """RANSAC 地面分割: 聚合多帧 → 直接拟合地平面 → 返回 (非地面点, 地面点).

    流程:
      1. 手动多帧聚合 (nsweeps 帧, min_dist=0.0 不过滤近点)
      2. 直接 RANSAC 平面拟合 (无 Z 轴粗筛, 避免"平坦世界"假设)
      3. 分割: 内点=地面, 外点=非地面

    Args:
        nusc: NuScenes 实例.
        sample: nuScenes sample dict.
        nsweeps: 聚合帧数, 默认 10.
        ransac_thresh: RANSAC 点到平面距离阈值 (米), 默认 0.25.
        min_dist: 最小距离过滤 (米). 默认 0.0 不过滤; 设置为 >0 可移除 ego 自身点.
        voxel_size: VoxelGrid 降采样尺寸 (米). 点云 > 200k 时推荐 0.1.
                    None 时不降采样.

    Returns:
        non_ground_pts: (N_ng, 4) 非地面点 [x, y, z, intensity].
        ground_pts: (N_g, 4) 地面点 [x, y, z, intensity].
    """
    # 1. 聚合点云
    pc = _multisweep_lidar(nusc, sample, nsweeps=nsweeps, min_dist=min_dist)
    pts = pc.points[:3, :].T.copy()          # (N, 3)
    intensity = pc.points[3, :].copy()        # (N,)

    # 2. 超大点云降采样 (替代 Z 轴截断)
    if voxel_size is not None and pts.shape[0] > 200000:
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(pts)
        pcd_down = pcd_full.voxel_down_sample(voxel_size)
        pts_ds = np.asarray(pcd_down.points)
        # 对降采样后的点做 RANSAC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_ds)

        plane_model, inliers_ds = pcd.segment_plane(
            distance_threshold=ransac_thresh,
            ransac_n=3,
            num_iterations=300
        )

        # 用降采样拟合的平面, 在全量点上做 inlier 判定
        a, b, c, d = plane_model
        all_dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)
        all_dists /= np.sqrt(a**2 + b**2 + c**2)
        ground_mask = all_dists <= ransac_thresh
    else:
        # 直接在原始点云上做 RANSAC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_thresh,
            ransac_n=3,
            num_iterations=300
        )

        inliers_set = set(inliers)
        ground_mask = np.zeros(pts.shape[0], dtype=bool)
        ground_mask[list(inliers_set)] = True

    # 3. 分割
    non_ground_mask = ~ground_mask
    non_ground_pts = np.column_stack([pts[non_ground_mask], intensity[non_ground_mask]])
    ground_pts = np.column_stack([pts[ground_mask], intensity[ground_mask]])

    return non_ground_pts, ground_pts


def remove_ground_from_points(pts_xyz, intensity=None, ransac_thresh=0.25,
                               voxel_size=None):
    """纯点云输入的地面分割 (无需 nuScenes API, 适用于已聚合的点云).

    Args:
        pts_xyz: (N, 3) 点云 xyz 坐标.
        intensity: (N,) 可选强度值.
        ransac_thresh: RANSAC 距离阈值 (米).
        voxel_size: VoxelGrid 降采样尺寸, None 则不降采样.

    Returns:
        non_ground_pts: (M, 3) 或 (M, 4) 非地面点.
        ground_pts: (K, 3) 或 (K, 4) 地面点.
    """
    pts = pts_xyz.astype(np.float64).copy()

    if voxel_size is not None and pts.shape[0] > 200000:
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(pts)
        pcd_down = pcd_full.voxel_down_sample(voxel_size)
        pts_ds = np.asarray(pcd_down.points)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_ds)
        plane_model, inliers_ds = pcd.segment_plane(
            distance_threshold=ransac_thresh,
            ransac_n=3,
            num_iterations=300
        )

        a, b, c, d = plane_model
        all_dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d)
        all_dists /= np.sqrt(a**2 + b**2 + c**2)
        ground_mask = all_dists <= ransac_thresh
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_thresh,
            ransac_n=3,
            num_iterations=300
        )
        inliers_set = set(inliers)
        ground_mask = np.zeros(pts.shape[0], dtype=bool)
        ground_mask[list(inliers_set)] = True

    non_ground_xyz = pts[~ground_mask]
    ground_xyz = pts[ground_mask]

    if intensity is not None:
        non_ground_pts = np.column_stack([non_ground_xyz, intensity[~ground_mask]])
        ground_pts = np.column_stack([ground_xyz, intensity[ground_mask]])
    else:
        non_ground_pts = non_ground_xyz
        ground_pts = ground_xyz

    return non_ground_pts, ground_pts
