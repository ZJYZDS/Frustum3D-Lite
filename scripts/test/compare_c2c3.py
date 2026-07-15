"""
C2 vs C3 逐帧对比: 加载两个模型, 在验证集上推理, 对比物理误差.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset_phase2 import Phase2Dataset, phase2_collate
from src.fusion import LidarOnlyRefiner, CrossModalFusion
from src.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(cls, ckpt_path, **kwargs):
    model = cls(**kwargs).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def main():
    # 加载数据
    ds = Phase2Dataset("data/nuscenes", split="val", cfg={
        "min_lidar_pts": 10, "num_points": 256, "crop_size": 128,
        "bbox_margin": 0.3, "noise_center": 0.3, "noise_size": 0.15,
        "noise_yaw_deg": 5.0, "match_max_dist_px": 80, "val_scene_ids": 2,
    }, detector_path="models/yolo26s.onnx")
    loader = DataLoader(ds, batch_size=16, collate_fn=phase2_collate, shuffle=False)

    # 加载模型
    c2 = load_model(LidarOnlyRefiner, "checkpoints_phase2/lidar_only.pt")
    c3 = load_model(CrossModalFusion, "checkpoints_phase2/fusion.pt")

    # 统计
    c2_all, c3_all = [], []

    print(f"{'Frame':>5s} | {'C2 center':>9s} {'C2 size':>8s} {'C2 yaw':>7s} | "
          f"{'C3 center':>9s} {'C3 size':>8s} {'C3 yaw':>7s} | {'Win':>4s}")
    print("-" * 78)

    obj_idx = 0
    with torch.no_grad():
        for rgb, lidar, target in loader:
            rgb, lidar, target = rgb.to(DEVICE), lidar.to(DEVICE), target.to(DEVICE)

            pred_c2 = c2(None, lidar)
            pred_c3 = c3(rgb, lidar)

            for i in range(len(target)):
                m2 = compute_metrics(pred_c2[i:i+1], target[i:i+1])
                m3 = compute_metrics(pred_c3[i:i+1], target[i:i+1])

                c2_all.append(m2)
                c3_all.append(m3)

                if obj_idx < 30:  # 打印前 30 个物体
                    winner = "C2" if m2["center_err"] + m2["size_err"] < \
                                      m3["center_err"] + m3["size_err"] else "C3"
                    print(f"{obj_idx:5d} | {m2['center_err']:8.3f}m {m2['size_err']:7.3f}m "
                          f"{m2['yaw_deg']:6.2f}d | {m3['center_err']:8.3f}m {m3['size_err']:7.3f}m "
                          f"{m3['yaw_deg']:6.2f}d | {winner:>4s}")
                obj_idx += 1

    # 汇总
    print(f"\n{'='*60}")
    print(f"Total objects: {obj_idx}")
    for name, results in [("C2", c2_all), ("C3", c3_all)]:
        avg = {k: np.mean([r[k] for r in results]) for k in results[0]}
        print(f"\n{name}:")
        print(f"  center_err: {avg['center_err']:.3f}m")
        print(f"  size_err:   {avg['size_err']:.3f}m")
        print(f"  yaw_err:    {avg['yaw_deg']:.2f}deg")

    # 统计胜负
    c2_wins = sum(1 for a, b in zip(c2_all, c3_all)
                  if a["center_err"] + a["size_err"] < b["center_err"] + b["size_err"])
    c3_wins = len(c2_all) - c2_wins
    print(f"\nHead-to-head (center+size): C2 wins {c2_wins}, C3 wins {c3_wins}")

if __name__ == "__main__":
    main()
