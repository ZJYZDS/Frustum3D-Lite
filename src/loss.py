"""
损失函数: PointNet 3D 检测 — center 残差 + size 对数残差 + yaw 2θ.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet3DLoss(nn.Module):
    """三头联合损失.

    - Center: SmoothL1(dx, dy, dz)  — 相对质心的残差
    - Size:   MSE(δl, δw, δh)       — 相对先验的对数残差
    - Yaw:    1 - cos(2Δθ)          — 2θ 表示自然消歧 180°
    """

    def __init__(self, center_w=2.0, size_w=0.3, yaw_w=0.3,
                 center_scale=3.0, size_scale=5.0):
        super().__init__()
        self.center_w = center_w
        self.size_w = size_w
        self.yaw_w = yaw_w
        self.center_scale = center_scale
        self.size_scale = size_scale

    def forward(self, pred, target, class_ids=None):
        """
        Args:
            pred:   (B, 7) [dx, dy, dz, δl, δw, δh, cos2θ, sin2θ]
            target: (B, 7) same format
            class_ids: (B,) optional, pedestrian (0) / rider (1) skip yaw loss
        Returns:
            total_loss, {"loss": float, "center": float, "size": float, "yaw": float}
        """
        # Center: SmoothL1 on residual
        loss_center = F.smooth_l1_loss(pred[:, :3], target[:, :3])

        # Size: MSE on log-residual
        loss_size = F.mse_loss(pred[:, 3:6], target[:, 3:6])

        # Yaw: 1 - cos(2Δθ), 先归一化到单位圆
        u, v = pred[:, 6], pred[:, 7]
        norm = torch.sqrt(u**2 + v**2 + 1e-8)
        u, v = u / norm, v / norm
        cos_2diff = u * target[:, 6] + v * target[:, 7]
        loss_per_sample = 1.0 - cos_2diff  # (B,)

        # 行人和骑行者几何上无方向信息, 跳过 yaw loss
        if class_ids is not None:
            mask = (class_ids != 0) & (class_ids != 1)  # skip pedestrian & rider
            mask = mask.float()
            if mask.sum() > 0:
                loss_yaw = (loss_per_sample * mask).sum() / mask.sum()
            else:
                loss_yaw = loss_per_sample.mean() * 0.0  # all pedestrians → 0
        else:
            loss_yaw = loss_per_sample.mean()

        total = (self.center_w * loss_center +
                 self.size_w * loss_size +
                 self.yaw_w * loss_yaw)

        return total, {
            "loss": total.item(),
            "center": loss_center.item(),
            "size": loss_size.item(),
            "yaw": loss_yaw.item(),
        }

    def denormalize(self, pred, centroid, priors):
        center = pred[:, :3] * self.center_scale + centroid
        size = priors * torch.exp(pred[:, 3:6])
        u, v = pred[:, 6], pred[:, 7]
        norm = torch.sqrt(u**2 + v**2 + 1e-8)
        yaw = 0.5 * torch.atan2(v / norm, u / norm)
        yaw = torch.where(yaw < 0, yaw + math.pi, yaw)  # [0, π)
        return center, size, yaw


# 兼容
BboxCenterSizeLoss = PointNet3DLoss
BboxAbsoluteLoss = PointNet3DLoss
