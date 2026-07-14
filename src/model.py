"""
PointNet++ 核心模块: FPS 降采样 + Ball Query 分组 + Mini-PointNet 特征提取.

Set Abstraction (SA) 是 PointNet++ 的基本计算单元:
  1. FPS (最远点采样): 从 N 个点中选出 M 个代表点 (centroids)
  2. Ball Query: 以每个 centroid 为中心, 在半径 r 内找 k 个邻居
  3. Mini-PointNet: 对每个邻域用 MLP + MaxPool 提取局部特征

重复多层 SA 形成层次化特征: 局部细节 → 部件结构 → 全局语义.
"""

import torch
import torch.nn as nn


# ==============================================================================
# FPS: 最远点采样 (Farthest Point Sampling)
# ==============================================================================

def farthest_point_sample(xyz, npoint):
    """从 N 个点中采样 npoint 个点, 保证空间均匀覆盖.

    算法: 贪心迭代, 每次选离已选点集最远的点.
    时间复杂度 O(B × npoint × N), 相比随机采样能更好地保留物体形状.

    Args:
        xyz:    (B, N, 3)  输入点云坐标
        npoint: int         采样点数
    Returns:
        centroids: (B, npoint, 3)  采样点坐标
        idx:       (B, npoint)     采样点在原始点云中的索引
    """
    B, N, _ = xyz.shape
    device = xyz.device

    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.ones(B, N, device=device) * 1e10                     # 每个点到已选点集的最近距离

    # 第一个点随机选 (增加多样性)
    farthest = torch.randint(0, N, (B,), device=device, dtype=torch.long)
    batch_indices = torch.arange(B, device=device)

    for i in range(npoint):
        idx[:, i] = farthest                                             # 记录当前选中的点
        centroid = xyz[batch_indices, farthest].unsqueeze(1)            # (B, 1, 3)
        d = ((xyz - centroid) ** 2).sum(-1)                             # (B, N)  — 欧氏距离平方
        dist = torch.minimum(dist, d)                                    # 更新最近距离
        farthest = dist.argmax(-1)                                      # 选最远的点

    centroids = torch.stack([xyz[b, idx[b]] for b in range(B)], dim=0)
    return centroids, idx


# ==============================================================================
# Ball Query: 球查询邻域搜索
# ==============================================================================

def ball_query(centroids, xyz, radius, nsample):
    """为每个 centroid 在半径 radius 内找 nsample 个邻居.

    如果邻域内点不足 nsample, 复制最近的点补齐 (保证固定输出大小).
    如果邻域内没有任何点 (radius 太小), 用第一个点填充并后续被 mask 处理.

    Args:
        centroids: (B, M, 3)  查询中心点
        xyz:       (B, N, 3)  原始点云
        radius:    float      搜索半径 (米)
        nsample:   int        每组采样点数
    Returns:
        idx: (B, M, nsample)  邻居点的索引
    """
    B, M, _ = centroids.shape
    N = xyz.shape[1]
    device = xyz.device

    # 计算所有 centroid-点 对的距离: (B, M, N)
    c = centroids.unsqueeze(2)          # (B, M, 1, 3)
    p = xyz.unsqueeze(1)                # (B, 1, N, 3)
    dist = ((c - p) ** 2).sum(-1)       # (B, M, N)

    # 超出半径的点距离设为无穷大 (被过滤)
    idx = torch.arange(N, device=device).view(1, 1, N).expand(B, M, N)
    dist[dist > radius * radius] = 1e10

    # 按距离排序, 取最近的 nsample 个
    _, sort_idx = dist.sort(-1)         # (B, M, N)
    sort_idx = sort_idx[:, :, :nsample] # (B, M, nsample)
    idx = idx.gather(-1, sort_idx)

    # 对于没有足够邻居的点, 用最近点填充 (first point replication)
    first = idx[:, :, 0].unsqueeze(-1).expand(-1, -1, nsample)
    mask = dist.gather(-1, sort_idx) > 1e9   # 标记无效邻居
    idx[mask] = first[mask]                  # 用最近点替代

    return idx


# ==============================================================================
# Set Abstraction: FPS + Ball Query + Mini-PointNet
# ==============================================================================

class SetAbstraction(nn.Module):
    """Set Abstraction 层: 从点云中提取局部特征, 降采样.

    流程:
      1. FPS 选 centroids (空间均匀降采样)
      2. Ball Query 为每个 centroid 找 k 个邻居
      3. 计算邻居相对 centroid 的坐标 (去中心化)
      4. 拼接 (相对坐标, 原始特征) → MLP → MaxPool → 局部特征

    Args:
        npoint:   降采样后的点数
        radius:   Ball Query 搜索半径 (米). 越大感受野越大
        nsample:  每个邻域的采样点数. 越大局部信息越丰富
        in_dim:   输入特征维度 (包含相对坐标 xyz 的 3 维)
        mlp_dims: MLP 各层输出维度, 如 [64, 128, 256]
    """

    def __init__(self, npoint, radius, nsample, in_dim, mlp_dims):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        # Mini-PointNet: 1×1 Conv (等效于 per-point MLP) + BN + ReLU
        layers = []
        in_c = in_dim
        for out_c in mlp_dims:
            layers.append(nn.Conv2d(in_c, out_c, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU(inplace=True))
            in_c = out_c
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz, features, radius_scale=1.0):
        """
        Args:
            xyz:      (B, N, 3)  点云坐标
            features: (B, C, N)  点云特征 (初始为 xyz+intensity)
            radius_scale: float 半径缩放因子, 用于类自适应 SA (e.g. car=1.0, bus=2.0)
        Returns:
            centroids:    (B, npoint, 3)  降采样后的坐标
            new_features: (B, D_out, npoint)  降采样后的特征
        """
        B, N, _ = xyz.shape

        # 1. FPS 降采样
        centroids, _ = farthest_point_sample(xyz, self.npoint)

        # 2. Ball Query 邻域搜索 (radius 按类缩放)
        effective_radius = self.radius * radius_scale
        group_idx = ball_query(centroids, xyz, effective_radius, self.nsample)

        # 3. 收集邻域点的 xyz 坐标, 去中心化 (相对 centroid)
        batch_indices = torch.arange(B, device=xyz.device).view(
            B, 1, 1).expand(-1, self.npoint, self.nsample)
        grouped_xyz = xyz[batch_indices, group_idx]             # (B, npoint, nsample, 3)

        centroids_expand = centroids.unsqueeze(2)                # (B, npoint, 1, 3)
        grouped_xyz = grouped_xyz - centroids_expand             # 去中心化: 相对坐标

        # 4. 收集邻域点的特征, 与相对坐标拼接
        # PyTorch advanced indexing: 输出 shape = (B, npoint, nsample, C)
        if features is not None:
            grouped_feats = features[batch_indices, :, group_idx]   # (B, npoint, nsample, C)
            grouped = torch.cat([grouped_xyz, grouped_feats], dim=-1)
        else:
            grouped = grouped_xyz

        # 5. Mini-PointNet: (B, *, *, C) → (B, C, *, *) → Conv2d → MaxPool
        grouped = grouped.permute(0, 3, 1, 2)   # (B, C, npoint, nsample)
        grouped = self.mlp(grouped)              # (B, D_out, npoint, nsample)
        new_features, _ = grouped.max(-1)        # (B, D_out, npoint)  — 对称函数 max pooling

        return centroids, new_features


# ==============================================================================
# 完整 PointNet++ 回归模型 (Phase 1 遗留, Phase 2 不再使用)
# ==============================================================================

class PointNet2Refiner(nn.Module):
    """Phase 1 的完整 PointNet++ 回归模型 (3 层 SA + MLP head).

    Phase 2 中已被 fusion.py 中的 PointNet2Encoder + CrossModalFusion 替代.
    保留用于可能的回退测试.
    """

    def __init__(self, sa1_cfg, sa2_cfg, sa3_cfg, fc_dims, dropout=0.3):
        super().__init__()
        self.sa1 = SetAbstraction(**sa1_cfg)
        self.sa2 = SetAbstraction(**sa2_cfg)
        self.sa3 = SetAbstraction(**sa3_cfg)

        layers = []
        in_dim = fc_dims[0]
        for out_dim in fc_dims[1:-1]:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, fc_dims[-1]))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        xyz = x[..., :3]
        feats = x.permute(0, 2, 1)    # (B, 4, N): xyz + intensity

        c1, f1 = self.sa1(xyz, feats)
        c2, f2 = self.sa2(c1, f1)
        _, f3 = self.sa3(c2, f2)

        global_feat = f3.squeeze(-1)  # (B, D_out)
        return self.fc(global_feat)
