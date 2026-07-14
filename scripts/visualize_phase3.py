"""
Phase 3 3D BBox 可视化: 预测 vs GT, 点云叠加.

用法:
  python scripts/visualize_phase3.py [--num_samples 8] [--checkpoint checkpoints_phase3/best_model.pt]
"""

import argparse
import os
import sys
import math
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_phase3 import Phase3Dataset, phase3_collate
from src.fusion import DualHeadPointNet
from src.init_estimator import estimate_yaw_plane_fitting


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def denormalize_pred(pred):
    """将模型输出反归一化到物理单位.
    Args:
        pred: (B, 9) or (B, 8)  [cx/25, cy/25, cz/25, w/5, h/5, l/5, sin, cos, (forward_logit)]
    Returns:
        center: (B, 3) 物理米
        size:   (B, 3) 物理米 (w, l, h) LiDAR frame
        yaw:    (B,)   弧度
    """
    center = pred[:, :3].cpu().numpy() * 25.0
    size = pred[:, 3:6].cpu().numpy() * 5.0   # (w, l, h) from nuScenes
    yaw = np.arctan2(pred[:, 6].cpu().numpy(), pred[:, 7].cpu().numpy())
    return center, size, yaw


def get_bbox_corners(center, size, yaw):
    """生成 3D bbox 的 8 个角点.

    Args:
        center: (3,) [cx, cy, cz] LiDAR frame (米)
        size:   (3,) [w, l, h] 物体尺寸 (米): width, length, height
        yaw:    float  绕 z 轴旋转角 (弧度)
    Returns:
        (8, 3) 角点坐标
    """
    w, length, h = size  # nuScenes ordering: width, length, height

    # 物体局部帧 (yaw=0, 车头朝 +x): x=length, y=width, z=height
    dx = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * length / 2
    dy = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * w / 2
    dz = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * h / 2

    # 绕 z 轴旋转
    cos, sin = np.cos(yaw), np.sin(yaw)
    R = np.array([[cos, -sin, 0],
                  [sin,  cos, 0],
                  [0,    0,   1]])
    corners = R @ np.stack([dx, dy, dz])  # (3, 8)
    corners = corners.T + center.reshape(1, 3)  # (8, 3)
    return corners


def get_bbox_edges():
    """12 条边的顶点索引对."""
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 竖边
    ]


def draw_bbox_3d(ax, corners, color='r', linewidth=1.5, alpha=0.8, label=None):
    """在 3D 坐标系中画 bbox 线框."""
    edges = get_bbox_edges()
    for i, (e1, e2) in enumerate(edges):
        ax.plot3D([corners[e1, 0], corners[e2, 0]],
                  [corners[e1, 1], corners[e2, 1]],
                  [corners[e1, 2], corners[e2, 2]],
                  color=color, linewidth=linewidth, alpha=alpha,
                  label=label if i == 0 else None)


def draw_bbox_faces(ax, corners, color='r', alpha=0.15):
    """画半透明 bbox 面."""
    faces = [
        [corners[0], corners[1], corners[5], corners[4]],  # +y face
        [corners[2], corners[3], corners[7], corners[6]],  # -y face
        [corners[1], corners[2], corners[6], corners[5]],  # +x face
        [corners[0], corners[3], corners[7], corners[4]],  # -x face
        [corners[4], corners[5], corners[6], corners[7]],  # +z face
        [corners[0], corners[1], corners[2], corners[3]],  # -z face
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor='none')
    ax.add_collection3d(poly)


def visualize_sample(ax, points, gt_center, gt_size, gt_yaw,
                     pred_center, pred_size, pred_yaw,
                     title="", max_dist=30):
    """在单个 subplot 中画点云 + 预测 bbox + GT bbox.

    三个视角: 侧视 (xz), 俯视 (xy), 前视 (yz)
    """
    # 生成角点
    gt_corners = get_bbox_corners(gt_center, gt_size, gt_yaw)
    pred_corners = get_bbox_corners(pred_center, pred_size, pred_yaw)

    # 过滤远处点
    mask = np.linalg.norm(points[:, :2], axis=1) < max_dist

    # ---- 俯视图 (xy 平面) ----
    ax[0].scatter(points[mask, 0], points[mask, 1], c='blue', s=0.3, alpha=0.5, label='LiDAR')

    # GT bbox 在 xy 平面的投影
    for e1, e2 in get_bbox_edges():
        ax[0].plot([gt_corners[e1, 0], gt_corners[e2, 0]],
                   [gt_corners[e1, 1], gt_corners[e2, 1]],
                   'g-', linewidth=2, alpha=0.8, label='GT' if (e1, e2) == (0, 1) else None)
    for e1, e2 in get_bbox_edges():
        ax[0].plot([pred_corners[e1, 0], pred_corners[e2, 0]],
                   [pred_corners[e1, 1], pred_corners[e2, 1]],
                   'r-', linewidth=2, alpha=0.8, label='Pred' if (e1, e2) == (0, 1) else None)

    # 画朝向箭头
    arrow_len = gt_size[1] * 0.8  # length
    ax[0].arrow(gt_center[0], gt_center[1],
                arrow_len * np.cos(gt_yaw), arrow_len * np.sin(gt_yaw),
                head_width=0.3, head_length=0.3, fc='green', ec='green', alpha=0.7)
    ax[0].arrow(pred_center[0], pred_center[1],
                arrow_len * np.cos(pred_yaw), arrow_len * np.sin(pred_yaw),
                head_width=0.3, head_length=0.3, fc='red', ec='red', alpha=0.7)

    ax[0].set_xlabel('X (forward)'); ax[0].set_ylabel('Y (left)')
    ax[0].set_title('Top View (XY)')
    ax[0].legend(loc='upper right', fontsize=6)
    ax[0].set_aspect('equal')
    ax[0].grid(True, alpha=0.3)

    # ---- 前视图 (XZ 平面) ----
    ax[1].scatter(points[mask, 0], points[mask, 2], c='blue', s=0.3, alpha=0.5)
    for e1, e2 in get_bbox_edges():
        ax[1].plot([pred_corners[e1, 0], pred_corners[e2, 0]],
                   [pred_corners[e1, 2], pred_corners[e2, 2]], 'r-', linewidth=2, alpha=0.8)
    for e1, e2 in get_bbox_edges():
        ax[1].plot([gt_corners[e1, 0], gt_corners[e2, 0]],
                   [gt_corners[e1, 2], gt_corners[e2, 2]], 'g-', linewidth=2, alpha=0.8)

    ax[1].set_xlabel('X (forward)'); ax[1].set_ylabel('Z (up)')
    ax[1].set_title('Front View (XZ)')
    ax[1].set_aspect('equal')
    ax[1].grid(True, alpha=0.3)

    # ---- 侧视图 (YZ 平面) ----
    ax[2].scatter(points[mask, 1], points[mask, 2], c='blue', s=0.3, alpha=0.5)
    for e1, e2 in get_bbox_edges():
        ax[2].plot([pred_corners[e1, 1], pred_corners[e2, 1]],
                   [pred_corners[e1, 2], pred_corners[e2, 2]], 'r-', linewidth=2, alpha=0.8)
    for e1, e2 in get_bbox_edges():
        ax[2].plot([gt_corners[e1, 1], gt_corners[e2, 1]],
                   [gt_corners[e1, 2], gt_corners[e2, 2]], 'g-', linewidth=2, alpha=0.8)

    ax[2].set_xlabel('Y (left)'); ax[2].set_ylabel('Z (up)')
    ax[2].set_title('Side View (YZ)')
    ax[2].set_aspect('equal')
    ax[2].grid(True, alpha=0.3)

    # 标题
    center_err = np.linalg.norm(pred_center - gt_center)
    size_err = np.abs(pred_size - gt_size).mean()
    yaw_err = abs(((pred_yaw - gt_yaw + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi
    fig = ax[0].figure
    fig.suptitle(f'{title}  |  Center err={center_err:.2f}m  Size err={size_err:.2f}m  Yaw err={yaw_err:.1f}°',
                 fontsize=10, fontweight='bold')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints_phase3/best_model.pt')
    parser.add_argument('--config', type=str, default='config/phase3.yaml')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='display/phase3_viz')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 加载配置 ----
    cfg = load_config(args.config)
    data_cfg = cfg.get('dataset', {})

    # ---- 加载模型 ----
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_cfg = cfg.get('model', {})
    model = DualHeadPointNet(
        pointnet_dim=model_cfg.get('pointnet_dim', 256),
        head_A_hidden=model_cfg.get('head_A_hidden', 128),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    # 兼容旧 checkpoint (可能包含已删除的 Head B 参数)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', 'N/A'):.4f}")
    print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} matching params")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ---- 加载数据集 (val split, 无增强) ----
    dataset = Phase3Dataset(
        nusc_root=data_cfg.get('nusc_root', 'data/nuscenes'),
        version=data_cfg.get('version', 'v1.0-mini'),
        split='val',
        detector_path=cfg.get('detector_path', 'models/yolo26s.onnx'),
        nsweeps=data_cfg.get('nsweeps', 5),
        num_points=data_cfg.get('num_points', 512),
        crop_size=data_cfg.get('crop_size', 128),
        max_dist=data_cfg.get('max_dist', 25.0),
        min_points=data_cfg.get('min_points', 5),
        match_threshold=data_cfg.get('match_threshold', 80.0),
        val_scene_ids=data_cfg.get('val_scene_ids', 2),
        remove_ground=data_cfg.get('remove_ground', True),
        use_augmentation=False,
        preprocess_dir=data_cfg.get('preprocess_dir', None),
    )

    # ---- 逐个样本可视化 ----
    all_center_errs = []
    all_size_errs = []
    all_yaw_errs = []

    sample_idx = 0
    viz_count = 0
    while viz_count < args.num_samples and sample_idx < len(dataset):
        frame_samples = dataset[sample_idx]
        sample_idx += 1
        if not frame_samples:
            continue

        for s in frame_samples:
            if viz_count >= args.num_samples:
                break

            # 准备输入
            points = s['points'].unsqueeze(0).to(device)
            xyz_min = s['xyz_min'].unsqueeze(0).to(device)
            xyz_max = s['xyz_max'].unsqueeze(0).to(device)
            class_id = torch.tensor([s['class_id']], dtype=torch.long).to(device)

            # 推理 (Head A: center+size)
            with torch.no_grad():
                pred_6d = model(points=points, xyz_min=xyz_min, xyz_max=xyz_max,
                                class_ids=class_id)

            # Plane-fitting yaw with predicted size
            plane_yaw, _ = estimate_yaw_plane_fitting(points[0].cpu().numpy(), pred_size=pred_size)

            # 反归一化
            pred_center = pred_6d[0, :3].cpu().numpy() * 25.0
            pred_size = pred_6d[0, 3:6].cpu().numpy() * 5.0
            pred_yaw = np.array([plane_yaw])
            gt_center = s['target'][:3].numpy() * 25.0
            gt_size = s['target'][3:6].numpy() * 5.0
            gt_yaw = np.arctan2(s['target'][6].item(), s['target'][7].item())

            # 指标
            center_err = np.linalg.norm(pred_center - gt_center)
            size_err = np.abs(pred_size - gt_size).mean()
            yaw_err = float(abs(((pred_yaw[0] - gt_yaw + np.pi) % (2 * np.pi)) - np.pi) * 180 / np.pi)

            all_center_errs.append(center_err)
            all_size_errs.append(size_err)
            all_yaw_errs.append(yaw_err)

            # ---- 绘制 ----
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            class_names = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 6: 'motorcycle', 7: 'bicycle'}
            cls_name = class_names.get(s['class_id'], f'cls{s["class_id"]}')

            title = f"#{viz_count + 1} {cls_name}"
            visualize_sample(
                axes,
                points=s['points'].numpy(),
                gt_center=gt_center, gt_size=gt_size, gt_yaw=gt_yaw,
                pred_center=pred_center, pred_size=pred_size, pred_yaw=pred_yaw[0],
                title=title,
            )

            save_path = os.path.join(args.output_dir, f'sample_{viz_count:02d}_{cls_name}.png')
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
            print(f"  #{viz_count + 1} {cls_name}: center_err={center_err:.2f}m "
                  f"size_err={size_err:.2f}m yaw_err={yaw_err:.1f}° -> {save_path}")
            viz_count += 1

    # ---- 总结 ----
    print(f"\n{'='*60}")
    print(f"Visualized {viz_count} samples, saved to {args.output_dir}/")
    if all_center_errs:
        print(f"Summary: center={np.mean(all_center_errs):.3f}±{np.std(all_center_errs):.3f}m "
              f"size={np.mean(all_size_errs):.3f}±{np.std(all_size_errs):.3f}m "
              f"yaw={np.mean(all_yaw_errs):.1f}±{np.std(all_yaw_errs):.1f}°")


if __name__ == '__main__':
    main()
