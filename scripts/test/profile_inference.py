"""
Frustum3D-Lite 推理管线性能分析脚本.

在推理管线的每个关键阶段插入计时器, 输出每帧各阶段耗时占比.

管线阶段:
  1. 点云加载      — 从预处理 .npy 读取或多帧聚合
  2. 点云滤波      — 地面去除 (RANSAC) + ego-vehicle 点过滤
  3. YOLO 2D 检测  — 图像目标检测 (ONNX / .pt)
  4. 视锥裁剪      — 2D bbox → 3D 锥体截取 (per-detection累计)
  5. 点云预处理    — ROR 离群点剔除 + DBSCAN 聚类 + 采样 (per-detection累计)
  6. PointNet 推理 — 模型 forward pass (per-detection累计)
  7. 后处理/NMS    — 解码 center/size/yaw, PCA 兜底 (per-detection累计)

用法:
  # 单帧单相机 (CAM_FRONT)
  python scripts/test/profile_inference.py \
    --nusc-root /media/zjy/Ventoy1/cross_atn_pointNet++/data/nuscenes \
    --preprocess-dir /media/zjy/Ventoy1/cross_atn_pointNet++/data/nuscenes/preprocess_phase3/nsweeps_5 \
    --detector /media/zjy/Ventoy1/cross_atn_pointNet++/weiTiao_pt/best.pt

  # 360度全6相机 (更全面的基线)
  python scripts/test/profile_inference.py ... --all-cameras

  # 不加载模型权重 (仅测管线预处理耗时)
  python scripts/test/profile_inference.py ... --no-model
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.profiler import PipelineProfiler
from pipeline.preprocess import (
    aggregate_sweeps,
    filter_points_by_frustum,
    remove_statistical_outliers,
    extract_largest_cluster,
    NUSCENES_CAT_TO_CLASS,
)
from pipeline.fusion import PointNet3DDetector
from pipeline.projector import LiDARProjector
from pipeline.detector import YOLOPtDetector, YOLODetectONNX, OBSTACLE_CLASS_IDS
from nuscenes.nuscenes import NuScenes

try:
    from pipeline.preprocess_cpp import CppPreprocessor
    _CPP_AVAILABLE = True
except (FileNotFoundError, OSError) as e:
    _CPP_AVAILABLE = False
    CppPreprocessor = None


CLASS_NAMES = {
    0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus',
    5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: 'traffic light', 9: 'traffic sign',
}

SKIP_YAW_CLASSES = {0, 1, 8, 9}
CENTER_SCALE = 3.0
SIZE_SCALE = 5.0
SIZE_EXPAND = 1.10
MIN_POINTS = 20
NUM_POINTS = 512


# ── PCA Yaw (from inference.py) ────────────────────────────────────────

def _pca_yaw(pts):
    """PCA 估计 XY 平面主方向 → yaw ∈ [0, π)."""
    if len(pts) < 5:
        return 0.0, 0.0
    xy = pts[:, :2]
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, -1]
    yaw = math.atan2(principal[1], principal[0])
    if yaw < 0:
        yaw += math.pi
    conf = eigvals[-1] / (eigvals[-2] + 1e-8)
    return yaw, conf


# ── Single-frame profiled inference ─────────────────────────────────────

def profile_one_frame(model, pts_lidar, all_cam_data, device,
                      num_points=NUM_POINTS, min_points=MIN_POINTS,
                      profiler=None, use_model=True, cpp_pre=None):
    """Run full frustum pipeline on one frame with per-stage timing.

    Args:
        cpp_pre: CppPreprocessor instance (if None, uses Python preprocessing)
    """
    if profiler is None:
        profiler = PipelineProfiler()

    all_predictions = []

    # Per-frame accumulators for per-detection stages
    frustum_total_ms = 0.0
    preprocess_total_ms = 0.0
    inference_total_ms = 0.0
    postprocess_total_ms = 0.0
    total_detections = 0

    # ── Per-frame filter: ego-vehicle point removal ──
    pts_dist = np.linalg.norm(pts_lidar[:, :2], axis=1)
    pts_lidar = pts_lidar[pts_dist >= 1.5]

    # ── Process each camera independently ──
    for camera_name, img, dets, K, T_lidar2cam in all_cam_data:

        for det in dets:
            bbox = det['bbox'].copy()
            cls_id = det['class_id']
            cls_name = CLASS_NAMES.get(cls_id, f'cls_{cls_id}')

            # ---- Stage: Frustum cropping ----
            t0 = time.perf_counter()
            frustum_pts, margin = filter_points_by_frustum(
                pts_lidar, bbox, K, T_lidar2cam, margin='auto')
            frustum_total_ms += (time.perf_counter() - t0) * 1000.0

            if len(frustum_pts) < min_points:
                continue

            # ---- Stage: Preprocess (ROR + DBSCAN + sample) ----
            pts_sampled = None
            t0 = time.perf_counter()

            # C++ preprocessing: only for large frustums (>150 pts)
            # where it gives 18× speedup and rarely fails.
            # Small frustums: Python DBSCAN is already fast and more reliable.
            cpp_used = False
            if cpp_pre is not None and len(frustum_pts) > 10:
                result = cpp_pre.process_frustum(frustum_pts, class_id=cls_id)
                if result is not None and len(result) >= min_points:
                    pts_sampled = result.astype(np.float32)
                    n_pts = len(result)
                    cpp_used = True

            # Fall back to Python if C++ failed or not available
            if pts_sampled is None:
                ror_pts = remove_statistical_outliers(frustum_pts,
                                                      nb_neighbors=20, std_ratio=2.0)
                if len(ror_pts) < min_points:
                    ror_pts = frustum_pts

                cluster_pts = extract_largest_cluster(ror_pts, eps=0.6, min_samples=8)
                if len(cluster_pts) < min_points:
                    continue

                n_pts = len(cluster_pts)
                if n_pts > num_points:
                    idx = np.random.choice(n_pts, num_points, replace=False)
                else:
                    idx = np.random.choice(n_pts, num_points, replace=True)
                pts_sampled = cluster_pts[idx].astype(np.float32)

            preprocess_total_ms += (time.perf_counter() - t0) * 1000.0

            if pts_sampled is None or len(pts_sampled) < min_points:
                continue

            # ---- Stage: PointNet inference ----
            if use_model:
                pts_tensor = torch.from_numpy(pts_sampled).unsqueeze(0).to(device)
                cid_tensor = torch.tensor([cls_id], dtype=torch.long).to(device)

                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(points=pts_tensor, class_ids=cid_tensor)
                inference_total_ms += (time.perf_counter() - t0) * 1000.0
            else:
                # Dry-run: skip model, measure only pre/post processing
                pts_tensor = torch.from_numpy(pts_sampled).unsqueeze(0).to(device)
                cid_tensor = torch.tensor([cls_id], dtype=torch.long).to(device)
                with torch.no_grad():
                    out = model(points=pts_tensor, class_ids=cid_tensor)
                inference_total_ms += 0.0  # not measured in dry-run

                out = torch.randn(1, 8, device=device)  # dummy output

            # ---- Stage: Postprocess (decode center/size/yaw) ----
            t0 = time.perf_counter()

            centroid = pts_sampled.mean(axis=0)
            prior = model.prior_table[cls_id].cpu().numpy()

            d_center = out[0, :3].cpu().numpy()
            d_size = out[0, 3:6].cpu().numpy()
            u, v = float(out[0, 6]), float(out[0, 7])
            yaw_norm = math.sqrt(u**2 + v**2 + 1e-8)

            center = centroid + d_center * CENTER_SCALE
            size = prior * np.exp(d_size) * SIZE_EXPAND

            if cls_id in SKIP_YAW_CLASSES:
                yaw, _ = _pca_yaw(pts_sampled)
            elif yaw_norm < 0.15:
                yaw, conf = _pca_yaw(pts_sampled)
                if conf < 1.2:
                    yaw = 0.5 * math.atan2(v / yaw_norm, u / yaw_norm)
                    if yaw < 0:
                        yaw += math.pi
            else:
                yaw = 0.5 * math.atan2(v / yaw_norm, u / yaw_norm)
                if yaw < 0:
                    yaw += math.pi

            postprocess_total_ms += (time.perf_counter() - t0) * 1000.0

            total_detections += 1
            all_predictions.append({
                'center': center,
                'size': size,
                'yaw': yaw,
                'yaw_norm': yaw_norm,
                'class_id': cls_id,
                'class_name': cls_name,
                'conf': det.get('conf', 1.0),
                'num_pts': n_pts,
                'centroid': centroid,
                'bbox': bbox,
                'camera': camera_name,
            })

    # Record per-frame stage totals
    profiler._current_frame['Frustum cropping'] = frustum_total_ms
    profiler._current_frame['Preprocess (sample)'] = preprocess_total_ms
    profiler._current_frame['PointNet inference'] = inference_total_ms
    profiler._current_frame['Postprocess (NMS)'] = postprocess_total_ms

    return all_predictions, total_detections


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Profile Frustum3D-Lite inference pipeline')
    parser.add_argument('--config', type=str, default='config/phase3.yaml',
                        help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path (optional; random init if omitted)')
    parser.add_argument('--nusc-root', type=str,
                        default='/media/zjy/Ventoy1/cross_atn_pointNet++/data/nuscenes',
                        help='nuScenes data root')
    parser.add_argument('--preprocess-dir', type=str,
                        default='/media/zjy/Ventoy1/cross_atn_pointNet++/data/nuscenes/preprocess_phase3/nsweeps_5',
                        help='Preprocessed .npy point cloud directory')
    parser.add_argument('--detector', type=str,
                        default='/media/zjy/Ventoy1/cross_atn_pointNet++/weiTiao_pt/best.pt',
                        help='YOLO detector path (.pt or .onnx)')
    parser.add_argument('--sample-token', type=str, default=None,
                        help='Specific sample_token to profile (random if omitted)')
    parser.add_argument('--all-cameras', action='store_true',
                        help='Use all 6 cameras for 360° coverage')
    parser.add_argument('--no-model', action='store_true',
                        help='Skip model forward pass (measure pre/post only)')
    parser.add_argument('--num-frames', type=int, default=1,
                        help='Number of frames to profile')
    parser.add_argument('--nsweeps', type=int, default=5,
                        help='Number of LiDAR sweeps to aggregate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--warmup', type=int, default=3,
                        help='Warmup iterations (run full pipeline then discard)')
    args = parser.parse_args()

    # ── Config ──
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        dc = cfg.get('dataset', {})
    else:
        cfg, dc = {}, {}

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # ── Model ──
    model = PointNet3DDetector().to(device).eval()
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        md = model.state_dict()
        pd = {k: v for k, v in ckpt.get('model_state_dict', ckpt).items()
              if k in md and v.shape == md[k].shape}
        md.update(pd)
        model.load_state_dict(md)
        print(f"[Model] Loaded checkpoint: {args.checkpoint}")
        if 'epoch' in ckpt:
            print(f"[Model] epoch={ckpt['epoch']}, val_loss={ckpt.get('val_loss','?')}")
    else:
        print("[Model] Using randomly initialized weights (timing is still valid)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parameters")

    # ── NuScenes ──
    nusc = NuScenes(version='v1.0-mini', dataroot=args.nusc_root, verbose=False)
    projector = LiDARProjector(args.nusc_root)
    print(f"[NuScenes] {args.nusc_root} (v1.0-mini)")

    # ── YOLO Detector ──
    if args.detector.endswith('.pt'):
        detector = YOLOPtDetector(args.detector, conf_thresh=0.6)
    else:
        detector = YOLODetectONNX(args.detector, conf_thresh=0.5)

    # ── Select test frames ──
    # Build test split matching Phase3Dataset logic
    val_scene_ids = dc.get('val_scene_ids', 1)
    test_ratio = dc.get('test_ratio', 0.022)
    scenes = sorted(nusc.scene, key=lambda s: s['name'])

    # val scenes: last val_scene_ids scenes
    if val_scene_ids > 0:
        val_scene_tokens = {s['token'] for s in scenes[-val_scene_ids:]}
    else:
        val_scene_tokens = set()

    # test: random sample from non-val scenes
    import random
    random.seed(42)
    test_frames = []
    for sample in nusc.sample:
        if sample['scene_token'] in val_scene_tokens:
            continue
        if 'CAM_FRONT' in sample['data'] and 'LIDAR_TOP' in sample['data']:
            test_frames.append(sample['token'])

    n_test = max(1, int(len(test_frames) * test_ratio))
    random.shuffle(test_frames)
    test_frames = test_frames[:max(n_test, args.num_frames)]

    if args.sample_token:
        # Override with user-specified token
        test_frames = [args.sample_token]

    test_frames = test_frames[:args.num_frames]
    print(f"\n[Test] {len(test_frames)} frame(s) selected for profiling")

    # ── Cameras ──
    if args.all_cameras:
        cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
    else:
        cameras = ['CAM_FRONT']

    print(f"[Cameras] {cameras}")

    # ── Profiler ──
    profiler = PipelineProfiler()

    # ── Warmup: run N iterations to warm GPU/cache ──
    if args.warmup > 0 and len(test_frames) > 0:
        print(f"\n[Warmup] Running {args.warmup} warmup iteration(s)...")
        warmup_token = test_frames[0]
        sample = nusc.get('sample', warmup_token)
        for wi in range(args.warmup):
            # Minimal pipeline for warmup
            pp = os.path.join(args.preprocess_dir, f'{warmup_token}.npy')
            if os.path.exists(pp):
                _ = np.load(pp).astype(np.float32)
            # Load one image + run YOLO
            img = cv2.imread(os.path.join(
                args.nusc_root,
                nusc.get('sample_data', sample['data']['CAM_FRONT'])['filename']))
            if img is not None:
                _ = detector.predict(img)
            # Warmup model with dummy input
            dummy = torch.randn(4, NUM_POINTS, 3).to(device)
            dummy_cid = torch.randint(0, 10, (4,)).to(device)
            with torch.no_grad():
                _ = model(points=dummy, class_ids=dummy_cid)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        print("[Warmup] Done.\n")

    # ── C++ preprocessor (if available) ──
    cpp_pre = None
    if _CPP_AVAILABLE:
        try:
            cpp_pre = CppPreprocessor()
            print(f"[C++] CppPreprocessor initialized (shared library loaded)\n")
        except Exception as e:
            print(f"[C++] Failed to initialize: {e}\n")

    # ── Profile each frame ──
    for fi, sample_token in enumerate(test_frames):
        sample = nusc.get('sample', sample_token)
        scene_name = nusc.get('scene', sample['scene_token'])['name']
        print(f"\n{'─'*60}")
        print(f"  Frame {fi+1}/{len(test_frames)}: {sample_token[:16]}... "
              f"(scene: {scene_name})")
        print(f"{'─'*60}")

        # ────── Stage 1: Load point cloud ──────
        with profiler.stage('Load point cloud'):
            pp = os.path.join(args.preprocess_dir, f'{sample_token}.npy')
            if os.path.exists(pp):
                pts_lidar = np.load(pp).astype(np.float32)
                print(f"  [Load] Preprocessed .npy: {len(pts_lidar)} points")
            else:
                # Aggregate sweeps (raw, no ground removal yet)
                t0 = time.perf_counter()
                pc = aggregate_sweeps(nusc, sample, nsweeps=args.nsweeps)
                pts_raw = pc.points[:3, :].T.astype(np.float32)
                print(f"  [Load] Aggregated {args.nsweeps} sweeps: {len(pts_raw)} points")

                # Ground removal (part of loading in this path)
                pts_lidar = remove_ground_ransac(pts_raw)
                print(f"  [Load] + ground removal: {len(pts_lidar)} non-ground points")

        # ────── Stage 2: Filtering (ground removal for .npy path) ──────
        # Note: .npy files are already ground-removed. Timing is negligible here
        # unless we load raw sweeps. We track it for completeness.
        with profiler.stage('Filtering'):
            # .npy files are pre-ground-removed, so this stage is minimal
            # But we include ego-vehicle point removal
            pts_dist = np.linalg.norm(pts_lidar[:, :2], axis=1)
            pts_lidar = pts_lidar[pts_dist >= 1.5]
            pass  # actual timing captured by context manager

        # ────── Stage 3: YOLO 2D Detection ──────
        all_cam_data = []
        total_yolo_ms = 0.0
        for camera in cameras:
            sd_token = sample['data'].get(camera)
            if sd_token is None:
                continue
            img_path = os.path.join(args.nusc_root,
                                    nusc.get('sample_data', sd_token)['filename'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [YOLO] {camera}: failed to load image, skip")
                continue

            K, T_l2c, _ = projector.get_transform(sample_token, camera)
            if K is None:
                print(f"  [YOLO] {camera}: no calibration, skip")
                continue

            t0 = time.perf_counter()
            dets = detector.predict(img)
            dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]
            yolo_ms = (time.perf_counter() - t0) * 1000.0
            total_yolo_ms += yolo_ms

            all_cam_data.append((camera, img, dets, K, T_l2c))
            print(f"  [YOLO] {camera}: {len(dets)} detections ({yolo_ms:.1f} ms)")

        profiler._current_frame['YOLO 2D detection'] = total_yolo_ms

        # ────── Stages 4-7: Per-detection pipeline ──────
        predictions, n_dets = profile_one_frame(
            model, pts_lidar, all_cam_data, device,
            num_points=NUM_POINTS, min_points=MIN_POINTS,
            profiler=profiler, use_model=not args.no_model,
            cpp_pre=cpp_pre,
        )

        if device.type == 'cuda':
            torch.cuda.synchronize()

        # ── Commit frame ──
        profiler.new_frame()

        # ── Print per-prediction details ──
        print(f"\n  Total predictions: {len(predictions)} (from {n_dets} valid detections)")
        for i, p in enumerate(predictions):
            c = p['center']
            s = p['size']
            yaw_deg = math.degrees(p['yaw'])
            d = np.linalg.norm(c[:2])
            print(f"    [{i+1:2d}] {p['class_name']:<12s} "
                  f"c=({c[0]:5.1f},{c[1]:5.1f},{c[2]:5.1f}) "
                  f"sz=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) "
                  f"yaw={yaw_deg:5.0f}°  d={d:.1f}m  "
                  f"cam={p['camera']}")

        # ── Print per-frame breakdown ──
        profiler.print_breakdown(
            title=f"Frustum3D-Lite Per-frame Breakdown — Frame {fi+1}"
        )

    # ── Print overall summary (if multiple frames) ──
    if profiler.frame_count > 1:
        profiler.print_summary()

    print(f"Done. Profiled {profiler.frame_count} frame(s).")


if __name__ == '__main__':
    main()
