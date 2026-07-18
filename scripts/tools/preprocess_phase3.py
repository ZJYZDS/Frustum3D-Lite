"""
Phase 3 离线预处理: 预计算多帧聚合 + 地面去除, 存为 .npy 文件.

用法:
  python scripts/preprocess_phase3.py                          # 默认参数
  python scripts/preprocess_phase3.py --nsweeps 5 --split all  # 自定义

输出目录: data/nuscenes/preprocess_phase3/nsweeps_{N}/
每个帧保存一个 .npy 文件: {sample_token}.npy, shape (M, 3), float32.

训练时设置 Phase3Dataset(preprocess_dir="data/nuscenes/preprocess_phase3/nsweeps_10/")
即可跳过聚合+地面去除, 直接加载.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nuscenes.nuscenes import NuScenes
from src.dataset_phase3 import aggregate_sweeps, remove_ground_ransac


def build_frame_list(nusc, val_scene_ids=2, split='all'):
    """构建帧列表, 支持 train/val/all 切分."""
    scene_samples = {}
    for sample in nusc.sample:
        scene_samples.setdefault(sample['scene_token'], []).append(sample['token'])

    scenes = nusc.scene
    sorted_scenes = sorted(scenes, key=lambda s: s['name'])

    if split == 'train':
        split_scenes = sorted_scenes[:-val_scene_ids]
    elif split == 'val':
        split_scenes = sorted_scenes[-val_scene_ids:]
    else:
        split_scenes = sorted_scenes

    frames = []
    for scene in split_scenes:
        for sample_token in scene_samples.get(scene['token'], []):
            sample = nusc.get('sample', sample_token)
            if ('CAM_FRONT' in sample['data'] and
                    'LIDAR_TOP' in sample['data']):
                frames.append(sample_token)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Phase 3 offline preprocessing")
    parser.add_argument("--nusc_root", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--nsweeps", type=int, default=5)
    parser.add_argument("--split", default="all", choices=["train", "val", "all"])
    parser.add_argument("--val_scene_ids", type=int, default=2)
    parser.add_argument("--ransac_thresh", type=float, default=0.25)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root)
    frames = build_frame_list(nusc, args.val_scene_ids, args.split)

    output_dir = args.output_dir or os.path.join(
        args.nusc_root, f"preprocess_phase3/nsweeps_{args.nsweeps}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"预处理: {len(frames)} 帧, nsweeps={args.nsweeps}")
    print(f"输出: {output_dir}")
    print(f"{'='*60}")

    total_time = 0
    skipped = 0
    for i, sample_token in enumerate(frames):
        out_path = os.path.join(output_dir, f"{sample_token}.npy")
        if os.path.exists(out_path):
            skipped += 1
            continue

        t0 = time.time()

        # 多帧聚合 (aggregate_sweeps 需要 sample dict, 不是 token string)
        sample = nusc.get('sample', sample_token)
        pc = aggregate_sweeps(nusc, sample, nsweeps=args.nsweeps)
        pts = pc.points[:3, :].T  # (N, 3)

        # 地面去除
        pts = remove_ground_ransac(pts, distance_threshold=args.ransac_thresh)

        # 只保存 xyz (3 列), float32
        np.save(out_path, pts.astype(np.float32))

        elapsed = time.time() - t0
        total_time += elapsed

        if (i + 1) % 20 == 0:
            print(f"  [{i+1:4d}/{len(frames)}] "
                  f"{elapsed:.1f}s/frame, {pts.shape[0]} pts, "
                  f"avg={total_time/(i+1-skipped):.1f}s")

    if skipped:
        print(f"\n跳过 {skipped} 个已存在文件")
    print(f"完成: {len(frames)-skipped} 帧处理, "
          f"平均 {total_time/max(len(frames)-skipped,1):.1f}s/帧")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
