"""
C1/C2/C3 消融对比脚本: 每种模型训练 5 epochs, 输出对比表.

用法:
  python scripts/compare_c1c2c3.py

输出:
  每个 epoch 的 train/val 指标 + 最终对比表格 (Val Loss, Center Err, Size Err, Yaw Err)

目的:
  快速验证融合是否有效: C3 (Dual-Attn) 应该显著优于 C1 (2D-only) 和 C2 (3D-only).
  如果 C3 ≈ C1 或 C3 ≈ C2, 说明融合没有带来收益, 需要检查架构或数据.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from src.dataset_phase2 import Phase2Dataset, phase2_collate
from src.fusion import CrossModalFusion, ImageOnlyRefiner, LidarOnlyRefiner
from src.loss import BboxRefinementLoss
from src.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = {"loss": 0.0, "center": 0.0, "size": 0.0, "yaw": 0.0}
    for rgb, lidar, target in loader:
        rgb, lidar, target = rgb.to(device), lidar.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model(rgb, lidar)
        loss, loss_dict = criterion(pred, target)
        loss.backward()
        optimizer.step()
        for k in total:
            total[k] += loss_dict[k]
    n = max(len(loader), 1)
    return {k: v / n for k, v in total.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = {"loss": 0.0, "center": 0.0, "size": 0.0, "yaw": 0.0}
    total_metrics = {"center_err": 0.0, "size_err": 0.0, "yaw_deg": 0.0}
    for rgb, lidar, target in loader:
        rgb, lidar, target = rgb.to(device), lidar.to(device), target.to(device)
        pred = model(rgb, lidar)
        _, loss_dict = criterion(pred, target)
        metrics = compute_metrics(pred, target)
        for k in total_loss:
            total_loss[k] += loss_dict[k]
        for k in total_metrics:
            total_metrics[k] += metrics[k]
    n = max(len(loader), 1)
    return {k: v / n for k, v in total_loss.items()}, {k: v / n for k, v in total_metrics.items()}


def run_experiment(name, model, epochs, device):
    """训练一个模型并返回最佳 epoch 的指标."""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    criterion = BboxRefinementLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    best_val_loss = float("inf")
    results = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_avg = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_avg, val_metrics = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:2d} | "
              f"train_loss={train_avg['loss']:.4f} | "
              f"val_loss={val_avg['loss']:.4f} | "
              f"c_err={val_metrics['center_err']:.3f}m "
              f"s_err={val_metrics['size_err']:.3f}m "
              f"yaw={val_metrics['yaw_deg']:.1f}deg | "
              f"{elapsed:.0f}s")

        if val_avg["loss"] < best_val_loss:
            best_val_loss = val_avg["loss"]
        results.append({"epoch": epoch, "val": val_avg, "val_metrics": val_metrics})

    # 返回 val loss 最低的 epoch 的指标
    best = min(results, key=lambda r: r["val"]["loss"])
    return {
        "name": name,
        "best_val_loss": best["val"]["loss"],
        "best_center_err": best["val_metrics"]["center_err"],
        "best_size_err": best["val_metrics"]["size_err"],
        "best_yaw_deg": best["val_metrics"]["yaw_deg"],
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5
    batch_size = 32
    print(f"Device: {device}, Epochs: {epochs}, Batch size: {batch_size}")

    # ---- 加载数据 (会运行 YOLO ONNX 推理, 耗时约 1 分钟) ----
    print("Loading datasets (runs YOLO ONNX inference on all frames)...")
    t0 = time.time()
    train_set = Phase2Dataset("data/nuscenes", split="train",
                               detector_path="models/yolo26s.onnx")
    val_set = Phase2Dataset("data/nuscenes", split="val",
                             detector_path="models/yolo26s.onnx")
    print(f"Data loading: {time.time() - t0:.0f}s")
    print(f"Train frames: {len(train_set)}, Val frames: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               collate_fn=phase2_collate, num_workers=0,
                               drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             collate_fn=phase2_collate, num_workers=0,
                             drop_last=True)

    # ---- 依次训练三种模型 ----
    summary = []

    # C1: 纯 2D — 只用 ResNet18, 输入 rgb_crop
    model_c1 = ImageOnlyRefiner().to(device)
    summary.append(run_experiment("C1 (ImageOnly)", model_c1, epochs, device))

    # C2: 纯 3D — 只用 PointNet++, 输入 lidar_pts
    model_c2 = LidarOnlyRefiner().to(device)
    summary.append(run_experiment("C2 (LidarOnly)", model_c2, epochs, device))

    # C3: 跨模态融合 — 同时使用 RGB + LiDAR
    model_c3 = CrossModalFusion().to(device)
    summary.append(run_experiment("C3 (Fusion)", model_c3, epochs, device))

    # ---- 打印对比表 ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20s} {'Val Loss':>10s} {'Center Err':>12s} {'Size Err':>10s} {'Yaw Err':>10s}")
    print("-" * 62)
    for r in summary:
        print(f"{r['name']:<20s} {r['best_val_loss']:>10.4f} {r['best_center_err']:>10.4f}m {r['best_size_err']:>8.4f}m {r['best_yaw_deg']:>8.2f}deg")
