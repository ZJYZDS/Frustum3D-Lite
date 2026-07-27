"""
评估指标: 将模型输出转换为物理量 (米、度), 方便人工理解.

注意: 这些指标在训练时仅用于监控, 不参与反向传播.
"""

import numpy as np
import torch


def compute_metrics(pred, target):
    """计算预测与 GT 之间的物理误差 (Phase 2 残差回归).

    Args:
        pred:   (B, 8)  [dx,dy,dz, dw,dh,dl, sin(dθ), cos(dθ)]
        target: (B, 8)  同上
    Returns:
        {"center_err": m, "size_err": m, "yaw_deg": degrees}
    """
    with torch.no_grad():
        # center: 欧氏距离 (米)
        center_err = torch.norm(pred[:, :3] - target[:, :3], dim=1).mean()

        # size: 各维度绝对误差的均值 (米)
        size_err = torch.abs(pred[:, 3:6] - target[:, 3:6]).mean()

        # yaw: atan2 恢复角度 → 差值 (度), 处理周期性
        pred_yaw = torch.atan2(pred[:, 6], pred[:, 7])
        target_yaw = torch.atan2(target[:, 6], target[:, 7])
        # 标准化到 [-π, π]
        yaw_err = (pred_yaw - target_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_err = yaw_err.abs().mean() * (180.0 / np.pi)

    return {
        "center_err": center_err.item(),
        "size_err": size_err.item(),
        "yaw_deg": yaw_err.item(),
    }


def compute_metrics_absolute(pred, target, center_scale=25.0, size_scale=5.0):
    """Phase 3 物理指标 — 反归一化后计算.

    Args:
        pred:   (B, 8)  归一化 [cx/25, cy/25, cz/25, w/5, h/5, l/5, sin, cos]
        target: (B, 8)  归一化 同上
    Returns:
        {"center_err": m, "size_err": m, "yaw_deg": degrees}
    """
    with torch.no_grad():
        # Center: 反归一化 ×25m → 欧氏距离
        center_err = torch.norm(
            (pred[:, :3] - target[:, :3]) * center_scale, dim=1
        ).mean()

        # Size: 反归一化 ×5m → 各维绝对误差均值
        size_err = torch.abs(
            (pred[:, 3:6] - target[:, 3:6]) * size_scale
        ).mean()

        # Yaw: atan2 → 角度差 (度)
        pred_yaw = torch.atan2(pred[:, 6], pred[:, 7])
        target_yaw = torch.atan2(target[:, 6], target[:, 7])
        yaw_err = (pred_yaw - target_yaw + np.pi) % (2 * np.pi) - np.pi
        yaw_err = yaw_err.abs().mean() * (180.0 / np.pi)

    return {
        "center_err": center_err.item(),
        "size_err": size_err.item(),
        "yaw_deg": yaw_err.item(),
    }
