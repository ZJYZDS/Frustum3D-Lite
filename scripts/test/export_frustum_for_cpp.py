"""
导出视锥点云 + 2D 检测信息为 PCD 文件, 供 C++ applyPriorCrop 测试使用.

用法:
  python scripts/test/export_frustum_for_cpp.py \
    --nusc-root /media/zjy/Ventoy1/cross_atn_pointNet++/data/nuscenes \
    --preprocess-dir .../preprocess_phase3/nsweeps_5 \
    --detector /media/zjy/Ventoy1/cross_atn_pointNet++/weiTiao_pt/best.pt \
    --output-dir /tmp/frustum_test

输出 (--output-dir 下):
  frustum_000.pcd              ← LiDAR 帧视锥点云 (ASCII PCD)
  frustum_000_meta.txt         ← 检测元信息 (class_id, bbox, K, T_lidar2cam)
  ...
"""

import argparse, os, sys, math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.preprocess import filter_points_by_frustum
from pipeline.projector import LiDARProjector
from pipeline.detector import YOLOPtDetector, OBSTACLE_CLASS_IDS
from nuscenes.nuscenes import NuScenes


CLASS_NAMES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus',
    5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: 'traffic light', 9: 'traffic sign',
}


def save_pcd_ascii(filepath, pts):
    """Save (N, 3) float32 points as ASCII PCD file."""
    with open(filepath, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {len(pts)}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {len(pts)}\n")
        f.write("DATA ascii\n")
        for p in pts:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nusc-root', required=True)
    parser.add_argument('--preprocess-dir', required=True)
    parser.add_argument('--detector', required=True)
    parser.add_argument('--output-dir', default='/tmp/frustum_test')
    parser.add_argument('--sample-token', default=None,
                        help='Specific sample token; random if omitted')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.nusc_root, verbose=False)
    projector = LiDARProjector(args.nusc_root)
    detector = YOLOPtDetector(args.detector, conf_thresh=0.6)

    # Pick a test frame
    if args.sample_token:
        sample_token = args.sample_token
    else:
        import random; random.seed(42)
        frames = [(s['token'], s['scene_token'])
                  for s in nusc.sample
                  if 'CAM_FRONT' in s['data'] and 'LIDAR_TOP' in s['data']]
        # Pick one with good number of detections (scene-1094 worked well before)
        for token, scene in frames:
            s = nusc.get('scene', scene)
            if s['name'] == 'scene-1094':
                sample_token = token
                break
        else:
            sample_token = frames[0][0]

    sample = nusc.get('sample', sample_token)
    scene_name = nusc.get('scene', sample['scene_token'])['name']
    print(f"Frame: {sample_token} (scene: {scene_name})")

    # Load point cloud (preprocessed, ground-removed)
    npy_path = os.path.join(args.preprocess_dir, f'{sample_token}.npy')
    pts_lidar = np.load(npy_path).astype(np.float32)
    pts_lidar = pts_lidar[np.linalg.norm(pts_lidar[:, :2], axis=1) >= 1.5]
    print(f"LiDAR: {len(pts_lidar)} points (ground-removed, ego-filtered)")

    # Run YOLO on CAM_FRONT
    img_path = os.path.join(args.nusc_root,
                            nusc.get('sample_data',
                                     sample['data']['CAM_FRONT'])['filename'])
    import cv2
    img = cv2.imread(img_path)
    dets = detector.predict(img)
    dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]
    print(f"YOLO: {len(dets)} detections on CAM_FRONT")

    K, T_lidar2cam, _ = projector.get_transform(sample_token, 'CAM_FRONT')

    saved = 0
    for i, det in enumerate(dets):
        bbox = det['bbox'].copy()
        cls_id = det['class_id']
        cls_name = CLASS_NAMES.get(cls_id, f'cls_{cls_id}')

        # Frustum crop
        frustum_pts, margin = filter_points_by_frustum(
            pts_lidar, bbox, K, T_lidar2cam, margin='auto')
        if len(frustum_pts) < 5:
            print(f"  [{i}] {cls_name}: {len(frustum_pts)} frustum pts → SKIP")
            continue

        # Save PCD
        pcd_path = os.path.join(args.output_dir, f'frustum_{saved:03d}.pcd')
        save_pcd_ascii(pcd_path, frustum_pts)

        # Save metadata: class_id, bbox(x1,y1,x2,y2), bbox_center(u,v),
        #   bbox_h(pixels), K(3x3), T_lidar2cam(3x4)
        x1, y1, x2, y2 = bbox
        meta_path = os.path.join(args.output_dir, f'frustum_{saved:03d}_meta.txt')
        with open(meta_path, 'w') as f:
            f.write(f"class_id {cls_id}\n")
            f.write(f"class_name {cls_name}\n")
            f.write(f"bbox {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
            f.write(f"bbox_cu {(x1+x2)/2:.1f}\n")
            f.write(f"bbox_cv {(y1+y2)/2:.1f}\n")
            f.write(f"bbox_h {(y2-y1):.1f}\n")
            f.write(f"fx {K[0,0]:.4f}\n")
            f.write(f"fy {K[1,1]:.4f}\n")
            f.write(f"ppx {K[0,2]:.4f}\n")
            f.write(f"ppy {K[1,2]:.4f}\n")
            f.write(f"margin {margin}\n")

        # Also save K and T as numpy for reference
        np.savez(os.path.join(args.output_dir, f'frustum_{saved:03d}_calib.npz'),
                 K=K, T_lidar2cam=T_lidar2cam)

        print(f"  [{saved}] {cls_name}: {len(frustum_pts)} frustum pts "
              f"bbox={bbox.astype(int)} margin={margin}")

        saved += 1

    print(f"\nExported {saved} frustum clouds to {args.output_dir}/")
    print(f"\nTo run C++ test on one sample:")
    print(f"  cd build && make test_prior_crop")
    print(f"  ./bin/test_prior_crop {args.output_dir}/frustum_000.pcd "
          f"{args.output_dir}/frustum_000_meta.txt {args.output_dir}/frustum_000_cropped.pcd")


if __name__ == '__main__':
    main()
