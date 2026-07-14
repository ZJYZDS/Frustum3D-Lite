"""
PointNet 3D 检测: 共享骨干 + 3 并行回归头.

架构:
  PointNetBackbone(3→64→128 + MaxPool) → g(128)
  PriorEncoder(3→16) → s(16)
  CentroidEncoder(3→16) → e(16)  ← 空间上下文 (LiDAR 坐标, /50m 归一化)
  ExtentEncoder(3→16) → x(16)    ← 可见范围 / prior
  ViewDirEncoder(3→16) → v(16)   ← 物体→LiDAR 单位方向
  Concat → [g|s|e|x|v](192)
    ├─ Center head: 192→64→3  (dx,dy,dz 相对质心)
    ├─ Size head:   192→64→3  (δl,δw,δh 对数残差)
    └─ Yaw head:    192→64→2  (cos2θ,sin2θ, 消歧180°)

输出 (B,7): [dx, dy, dz, δl, δw, δh, cos2θ, sin2θ]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# PointNet 骨干
# ==============================================================================

class PointNetBackbone(nn.Module):
    """per-point MLP + MaxPool → 全局几何特征.

    输入: (B, N, 3)  质心归一化点云
    输出: (B, 128)    全局特征
    """

    def __init__(self, input_dim=3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.transpose(1, 2)      # (B, 3, N)
        x = self.mlp(x)             # (B, 128, N)
        return x.max(dim=-1)[0]     # (B, 128)


# ==============================================================================
# 3D 检测器: 共享骨干 + 3 并行头
# ==============================================================================

class PointNet3DDetector(nn.Module):
    """PointNet 3D 检测: center 残差 + size 对数残差 + yaw 2θ.

    输出 (B, 7): [dx, dy, dz, δl, δw, δh, cos2θ, sin2θ]
    """

    # nuScenes size = (width, length, height)
    # Typical values from nuScenes statistics
    CLASS_SIZE_PRIOR = {
        0: [0.70, 0.70, 1.70],   # pedestrian (w, l, h)
        1: [0.70, 0.70, 1.70],   # rider
        2: [1.90, 4.60, 1.50],   # car
        3: [2.50, 6.50, 2.80],   # truck
        4: [2.80, 10.5, 3.20],   # bus
        6: [0.70, 2.00, 1.50],   # motorcycle
        7: [0.60, 1.80, 1.30],   # bicycle
    }
    DEFAULT_PRIOR = [1.90, 4.60, 1.50]

    def __init__(self, hidden_dim=64, num_classes=10,
                 **_unused_kwargs):
        super().__init__()

        self.backbone = PointNetBackbone(input_dim=3)

        self.register_buffer('prior_table', torch.tensor([
            self.CLASS_SIZE_PRIOR.get(i, self.DEFAULT_PRIOR)
            for i in range(num_classes)
        ], dtype=torch.float32))

        self.prior_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
        )

        # 空间上下文: LiDAR 质心坐标 → 16 维
        self.centroid_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
        )

        # 可见范围: extent / prior → 16 维
        self.extent_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
        )

        # 视角方向: 物体→LiDAR 单位向量 → 16 维
        self.viewdir_encoder = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(inplace=True),
        )

        # Face coverage: 6 面覆盖率 + max_face_idx(one-hot 6) → 32 维
        # 最大值面信息告诉模型"朝哪个方向的面最完整 → 中心在对面"
        self.face_encoder = nn.Sequential(
            nn.Linear(12, 24),  # 6 cov + 6 max_face one-hot
            nn.ReLU(inplace=True),
            nn.Linear(24, 16),
            nn.ReLU(inplace=True),
        )

        fusion_dim = 128 + 16 + 16 + 16 + 16 + 16  # backbone + prior + centroid + extent + viewdir + face

        self.center_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim*2), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.size_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )
        self.yaw_head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

        # 零初始化 center/size 最后一层 (残差→0 合理)
        for head in [self.center_head, self.size_head]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        # yaw head 需要单位范数输出，用标准初始化
        nn.init.xavier_uniform_(self.yaw_head[-1].weight)
        nn.init.zeros_(self.yaw_head[-1].bias)

    def forward(self, points, class_ids=None, centroids=None,
                face_cov=None, max_face_idx=None, **_unused_kwargs):
        """
        Args:
            points: (B, N, 3)  LiDAR 点云
            class_ids: (B,)    类别 ID, 用于查先验尺寸
            centroids: (B, 3)  点云在 LiDAR 帧的质心, 提供空间上下文
            face_cov: (B, 6)   6 面覆盖率 [0,1]
            max_face_idx: (B,) 最大覆盖面的索引 (0-5)
        Returns:
            (B, 7) [dx, dy, dz, δl, δw, δh, cos2θ, sin2θ]
        """
        B = points.shape[0]

        # 质心归一化
        centroid = points.mean(dim=1, keepdim=True)          # (B, 1, 3)
        points_norm = points - centroid

        # 共享特征
        feat = self.backbone(points_norm)                     # (B, 128)

        # 尺寸先验
        if class_ids is not None:
            prior = self.prior_table[class_ids]                # (B, 3)
        else:
            prior = self.prior_table[2].unsqueeze(0).expand(B, -1)
        prior_feat = self.prior_encoder(prior)                 # (B, 16)

        # 空间上下文: LiDAR 质心 / 50m
        if centroids is not None:
            cent_raw = centroids                                # (B, 3)
        else:
            cent_raw = centroid.squeeze(1)                      # (B, 3)
        cent_feat = self.centroid_encoder(cent_raw / 50.0)     # (B, 16)

        # 可见范围: extent / prior
        pts_max = points.max(dim=1)[0]                          # (B, 3)
        pts_min = points.min(dim=1)[0]                          # (B, 3)
        extent = (pts_max - pts_min).clamp(min=0.01)            # (B, 3)
        extent_norm = extent / prior.clamp(min=0.01)            # (B, 3)
        extent_feat = self.extent_encoder(extent_norm)          # (B, 16)

        # 视角方向: 物体→LiDAR 单位向量
        view_dir = -cent_raw / (cent_raw.norm(dim=1, keepdim=True).clamp(min=1e-6))  # (B, 3)
        viewdir_feat = self.viewdir_encoder(view_dir)           # (B, 16)

        # Face coverage: 每面覆盖率 + 最大面指示
        if face_cov is not None and max_face_idx is not None:
            max_face_onehot = torch.zeros(B, 6, device=points.device, dtype=torch.float32)
            max_face_idx_gpu = max_face_idx.to(points.device)
            max_face_onehot.scatter_(1, max_face_idx_gpu.unsqueeze(1).long(), 1.0)
            face_input = torch.cat([face_cov.to(points.device), max_face_onehot], dim=-1)
            face_feat = self.face_encoder(face_input)                     # (B, 16)
        else:
            face_feat = torch.zeros(B, 16, device=points.device)

        fused = torch.cat([feat, prior_feat, cent_feat,
                           extent_feat, viewdir_feat, face_feat], dim=-1)  # (B, 208)

        d_center = self.center_head(fused)                     # (B, 3)
        d_size = self.size_head(fused)                         # (B, 3)
        yaw_2theta = self.yaw_head(fused)                      # (B, 2)

        return torch.cat([d_center, d_size, yaw_2theta], dim=-1)


# 兼容旧代码
DualHeadPointNet = PointNet3DDetector
