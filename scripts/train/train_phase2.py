"""
Phase 2 训练脚本: 支持 C1/C2/C3 三种模型的消融对比和完整训练.

用法:
  # 训练 C3 (默认)
  python scripts/train_phase2.py

  # 训练 C1 (2D baseline)
  python scripts/train_phase2.py --model_type c1_2d

  # 自定义配置
  python scripts/train_phase2.py --config config/phase2.yaml --epochs 50 --batch_size 32

CLI 参数优先级高于 YAML 配置文件.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dataset_phase2 import Phase2Dataset, phase2_collate
from src.fusion import CrossModalFusion, ImageOnlyRefiner, LidarOnlyRefiner
from src.loss import BboxRefinementLoss
from src.metrics import compute_metrics


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(cfg):
    """根据 config 的 model_type 构建对应模型.

    model_type:
      c1_2d      → ImageOnlyRefiner    (纯 2D, 不用点云)
      c2_3d      → LidarOnlyRefiner    (纯 3D PointNet++, 不用图像)
      cross_attn → CrossModalFusion    (跨模态融合, 默认)
    """
    model_type = cfg.get("model_type", "cross_attn")

    if model_type == "c1_2d":
        print("[Model] C1: ImageOnlyRefiner (2D only)")
        return ImageOnlyRefiner(
            d_model=cfg.get("d_model", 256),
            dropout=cfg.get("dropout", 0.3),
        )

    elif model_type == "c2_3d":
        print("[Model] C2: LidarOnlyRefiner (3D only)")
        return LidarOnlyRefiner(
            sa_configs=cfg.get("sa_configs"),
            dropout=cfg.get("dropout", 0.3),
        )

    else:  # cross_attn / C3
        print("[Model] C3: CrossModalFusion")
        return CrossModalFusion(
            sa_configs=cfg.get("sa_configs"),
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 8),
            num_layers=cfg.get("num_layers", 1),
            dropout=cfg.get("dropout", 0.1),
        )


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """训练一个 epoch.

    每 10 个 batch 打印一次当前指标, 方便监控训练状态.
    """
    model.train()
    total = {"loss": 0.0, "center": 0.0, "size": 0.0, "yaw": 0.0}
    start = time.time()

    for batch_idx, batch_data in enumerate(loader):
        rgb, lidar, target, class_groups = batch_data
        rgb = rgb.to(device); lidar = lidar.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        if isinstance(model, LidarOnlyRefiner):
            pred = model(rgb, lidar, class_groups=class_groups)
        else:
            pred = model(rgb, lidar)
        loss, loss_dict = criterion(pred, target)
        loss.backward()
        optimizer.step()

        for k in total:
            total[k] += loss_dict[k]

        if batch_idx % 10 == 0:
            metrics = compute_metrics(pred, target)
            print(f"  Epoch {epoch} | Batch {batch_idx:4d}/{len(loader)} | "
                  f"loss={loss_dict['loss']:.4f} | "
                  f"c={loss_dict['center']:.4f} s={loss_dict['size']:.4f} y={loss_dict['yaw']:.4f} | "
                  f"c_err={metrics['center_err']:.3f}m s_err={metrics['size_err']:.3f}m yaw={metrics['yaw_deg']:.2f}deg")

    elapsed = time.time() - start
    n = max(len(loader), 1)
    avg = {k: v / n for k, v in total.items()}
    print(f"  [Train] Epoch {epoch:3d} | loss={avg['loss']:.4f} c={avg['center']:.4f} "
          f"s={avg['size']:.4f} y={avg['yaw']:.4f} | {elapsed:.1f}s")
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """验证: 计算 loss + 物理指标 (米/度)."""
    model.eval()
    total_loss = {"loss": 0.0, "center": 0.0, "size": 0.0, "yaw": 0.0}
    total_metrics = {"center_err": 0.0, "size_err": 0.0, "yaw_deg": 0.0}

    for batch_data in loader:
        rgb, lidar, target, class_groups = batch_data
        rgb = rgb.to(device); lidar = lidar.to(device)
        target = target.to(device)
        if isinstance(model, LidarOnlyRefiner):
            pred = model(rgb, lidar, class_groups=class_groups)
        else:
            pred = model(rgb, lidar)
        _, loss_dict = criterion(pred, target)
        metrics = compute_metrics(pred, target)
        for k in total_loss:
            total_loss[k] += loss_dict[k]
        for k in total_metrics:
            total_metrics[k] += metrics[k]

    n = max(len(loader), 1)
    avg_loss = {k: v / n for k, v in total_loss.items()}
    avg_metrics = {k: v / n for k, v in total_metrics.items()}
    print(f"  [Val]   Epoch {epoch:3d} | loss={avg_loss['loss']:.4f} c={avg_loss['center']:.4f} "
          f"s={avg_loss['size']:.4f} y={avg_loss['yaw']:.4f} | "
          f"c_err={avg_metrics['center_err']:.3f}m s_err={avg_metrics['size_err']:.3f}m yaw={avg_metrics['yaw_deg']:.2f}deg")
    return avg_loss, avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 cross-attention fusion")
    parser.add_argument("--config", type=str, default="config/phase2.yaml")
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["c1_2d", "c2_3d", "cross_attn"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    args = parser.parse_args()

    # ---- 加载配置 ----
    cfg = load_config(args.config)

    # CLI 覆盖 YAML 配置
    if args.model_type is not None:
        cfg["model_type"] = args.model_type
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.device is not None:
        cfg["training"]["device"] = args.device

    # ---- 设备 ----
    device_str = cfg["training"].get("device", "auto")
    device = torch.device(
        "cuda" if device_str == "auto" and torch.cuda.is_available()
        else (device_str if device_str != "auto" else "cpu")
    )
    print(f"Device: {device}")

    # ---- 数据 ----
    data_root = args.data_root or cfg["data"]["root"]
    data_cfg = cfg.get("data", {})

    train_set = Phase2Dataset(
        data_root, split="train", cfg=data_cfg,
        detector_path=cfg.get("detector_path", "models/yolo26s.onnx"),
    )
    val_set = Phase2Dataset(
        data_root, split="val", cfg=data_cfg,
        detector_path=cfg.get("detector_path", "models/yolo26s.onnx"),
    )

    train_loader = DataLoader(
        train_set, batch_size=cfg["training"]["batch_size"], shuffle=True,
        collate_fn=phase2_collate,
        num_workers=min(cfg["training"].get("num_workers", 2), os.cpu_count() or 1),
        drop_last=True,   # 避免 batch_size=1 导致 BatchNorm crash
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg["training"]["batch_size"], shuffle=False,
        collate_fn=phase2_collate,
        num_workers=min(2, os.cpu_count() or 1),
        drop_last=True,
    )

    # ---- 模型 ----
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # ---- 优化器 & 调度器 ----
    loss_cfg = cfg.get("loss", {})
    criterion = BboxRefinementLoss(
        center_weight=loss_cfg.get("center_weight", 1.0),
        size_weight=loss_cfg.get("size_weight", 1.0),
        yaw_weight=loss_cfg.get("yaw_weight", 0.5),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"],
    )

    # ---- Checkpoint ----
    save_dir = Path(cfg["training"].get("save_dir", "checkpoints_phase2"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history = []
    start_epoch = 1

    # ---- Resume ----
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"  Resumed: epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', 'N/A'):.4f}" if isinstance(ckpt.get('val_loss'), float) else f"  Resumed: epoch {ckpt['epoch']}")

        # 剩余的 cosine 退火
        remaining = cfg["training"]["epochs"] - ckpt["epoch"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining,
        )

    print(f"\n{'='*60}")
    print(f"Training: epochs {start_epoch}→{cfg['training']['epochs']}, "
          f"{len(train_set)} train frames / {len(val_set)} val frames")
    print(f"Model type: {cfg.get('model_type', 'cross_attn')}")
    print(f"{'='*60}\n")

    # ---- 训练循环 ----
    for epoch in range(start_epoch, cfg["training"]["epochs"] + 1):
        train_avg = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_avg, val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        history.append({
            "epoch": epoch, "train": train_avg,
            "val": val_avg, "val_metrics": val_metrics,
        })

        # 保存最佳模型
        if val_avg["loss"] < best_loss:
            best_loss = val_avg["loss"]
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_loss, "history": history,
            }, save_dir / "best_model.pt")
            print(f"  -> Saved best (val_loss={best_loss:.4f})")

        # 保存最新模型 (用于断点续训)
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "history": history,
        }, save_dir / "latest.pt")

    # ---- 总结 ----
    print(f"\n{'='*60}")
    if history:
        best = min(history, key=lambda h: h["val"]["loss"])
        print(f"Best: epoch={best['epoch']}, val_loss={best['val']['loss']:.4f}")
    print(f"Checkpoints: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
