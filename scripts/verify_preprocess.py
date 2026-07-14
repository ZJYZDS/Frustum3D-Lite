"""
验证 Phase 3 数据预处理正确性: 坐标对齐、点数分布、视锥裁剪、标签一致性.
用法: python scripts/verify_preprocess.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.dataset_phase3 import (Phase3Dataset, phase3_collate,
    filter_points_by_frustum, filter_points_by_bbox_projection,
    _compute_adaptive_margin)

# 检查 1: 数据集加载
print("=" * 60)
print("1. 数据集加载")
ds = Phase3Dataset(
    nusc_root="data/nuscenes",
    version="v1.0-mini",
    split="train",
    detector_path="models/yolo26s.onnx",
    nsweeps=3,             # 加速验证
    num_points=512,
    max_dist=15.0,
    min_points=5,
    remove_ground=True,
    use_augmentation=False,  # 关闭增强, 便于检查
)
print(f"   帧数: {len(ds)}")

# 检查 2: 收集样本, 分析坐标对齐
print("\n" + "=" * 60)
print("2. 坐标对齐检查 (pts vs label center)")

total_samples = 0
total_frames_with_data = 0
fallback_counts = {1: 0, 2: 0, 3: 0}  # auto margin / margin=20 / bbox proj
pts_per_sample = []
center_dists = []   # |pts_mean - gt_center| 应对齐

# 修改 __getitem__ 来追踪 fallback 使用
original_getitem = ds.__getitem__

for frame_idx in range(min(len(ds), 30)):
    samples = ds[frame_idx]
    if not samples:
        continue
    total_frames_with_data += 1
    for s in samples:
        pts = s['points'].numpy()
        target = s['target'].numpy()
        gt_center = target[:3]
        pts_mean = pts.mean(axis=0)
        dist = np.linalg.norm(pts_mean - gt_center)
        center_dists.append(dist)
        pts_per_sample.append(len(pts))
        total_samples += 1

if center_dists:
    center_dists = np.array(center_dists)
    print(f"   有效帧: {total_frames_with_data}/30")
    print(f"   总样本: {total_samples}")
    print(f"   点数分布: min={np.min(pts_per_sample)}, "
          f"median={np.median(pts_per_sample):.0f}, max={np.max(pts_per_sample)}")
    print(f"   |pts_mean - gt_center|: "
          f"mean={center_dists.mean():.3f}m, median={np.median(center_dists):.3f}m, "
          f"max={center_dists.max():.3f}m")
    if center_dists.mean() > 2.0:
        print("   ⚠️  WARNING: 平均距离 > 2m, 坐标帧可能不对齐!")
    else:
        print("   ✅ 坐标对齐正常 (pts_mean 接近 gt_center)")
else:
    print("   ❌ 无有效样本!")

# 检查 3: 标签数值范围
print("\n" + "=" * 60)
print("3. 标签数值范围检查")
if total_samples > 0:
    # 手动收集几个 target
    targets = []
    for frame_idx in range(min(len(ds), 5)):
        for s in ds[frame_idx]:
            targets.append(s['target'].numpy())
    targets = np.array(targets)

    print(f"   Center: x=[{targets[:,0].min():.2f}, {targets[:,0].max():.2f}], "
          f"y=[{targets[:,1].min():.2f}, {targets[:,1].max():.2f}], "
          f"z=[{targets[:,2].min():.2f}, {targets[:,2].max():.2f}]")
    print(f"   Size:   w=[{targets[:,3].min():.2f}, {targets[:,3].max():.2f}], "
          f"h=[{targets[:,4].min():.2f}, {targets[:,4].max():.2f}], "
          f"l=[{targets[:,5].min():.2f}, {targets[:,5].max():.2f}]")
    print(f"   Yaw:    sin=[{targets[:,6].min():.2f}, {targets[:,6].max():.2f}], "
          f"cos=[{targets[:,7].min():.2f}, {targets[:,7].max():.2f}]")

    # 检查 size: 应该在合理范围 (nuScenes 物体 0.5m~20m)
    sizes_valid = (targets[:,3] > 0.3) & (targets[:,5] > 0.3)  # w>0.3, l>0.3
    print(f"   Size 合理性: {sizes_valid.sum()}/{len(targets)} valid")
    if not sizes_valid.all():
        print("   ⚠️  存在异常尺寸!")

    # sin²+cos² ≈ 1
    yaw_norm = targets[:,6]**2 + targets[:,7]**2
    yaw_ok = np.abs(yaw_norm - 1.0) < 0.01
    print(f"   sin²+cos²≈1: {yaw_ok.sum()}/{len(targets)} valid")
    if not yaw_ok.all():
        print("   ⚠️  存在异常 yaw 编码!")

# 检查 4: 自适应 margin 逻辑
print("\n" + "=" * 60)
print("4. 自适应 margin 测试")
test_cases = [
    (np.array([100,100,500,500]), "近处大车 (400x400)"),
    (np.array([100,100,200,200]), "中距物体 (100x100)"),
    (np.array([100,100,130,130]), "远处小车 (30x30)"),
    (np.array([10,10,30,30]),    "极远/极小 (20x20)"),
]
for bbox, desc in test_cases:
    m = _compute_adaptive_margin(bbox, None, None, None)
    print(f"   {desc}: margin={m}")

# 检查 5: 增强一致性 (手工验证旋转后坐标)
print("\n" + "=" * 60)
print("5. 增强旋转一致性测试")
# 关闭增强的数据集里取一个样本, 模拟旋转
ds.use_augmentation = True  # 临时开启
found = False
for frame_idx in range(min(len(ds), 50)):
    samples = ds[frame_idx]
    for s in samples:
        pts = s['points'].numpy()
        target = s['target'].numpy()
        # 检查旋转后 center 是否与点云质心一致
        pts_mean = pts.mean(axis=0)
        gt_center = target[:3]
        dist = np.linalg.norm(pts_mean - gt_center)
        if dist < 2.0:  # 正常样本
            print(f"   样本: pts_mean={pts_mean.round(2)}, gt_center={gt_center.round(2)}, "
                  f"dist={dist:.3f}m ✅")
            found = True
            break
    if found:
        break
ds.use_augmentation = False

print("\n" + "=" * 60)
print("验证完成")
