"""
Visible centroid vs Point cloud mean: 量化对比 z-bottom 可见表面中心的优势.

对比:
  Method A (OLD):  bbox 内所有点云均值 → center
  Method B (NEW):  z-bottom 35% 可见表面质心 → center (训练时使用的初始化)

关键: 两种方法使用相同的 bbox-based 点提取, 与训练完全一致.
"""
import os, sys, json, math
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_phase1 import (LiDARProjector, quaternion_to_yaw, quaternion_to_mat,
                                 rotate_points_z)
from src.detector import YOLOSegONNX, OBSTACLE_CLASS_IDS, OBSTACLE_CLASSES
from src.init_estimator import estimate_yaw_pca, filter_points_by_mask
from src.model import farthest_point_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "data/nuscenes"
CKPT_PATH = "checkpoints_phase2/lidar_only.pt"
NUM_FRAMES = 10

DEFAULT_SIZE = {
    0: (0.7, 0.7, 1.75), 1: (0.5, 1.8, 1.2), 2: (2.0, 4.5, 1.6),
    3: (0.8, 2.2, 1.5), 5: (2.8, 10.0, 3.0), 7: (2.8, 7.0, 2.5),
}
NUSC_CATS = {"vehicle.car", "vehicle.truck", "vehicle.bus",
             "vehicle.motorcycle", "vehicle.bicycle", "human.pedestrian"}
rng = np.random.default_rng(42)


def match_gt_2d(det_bbox, gt_anns, K, sample_token, frame_ego, ego_poses, projector):
    det_cx, det_cy = (det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2
    best_ann, best_dist = None, 80.0
    ego_pose_token = frame_ego.get(sample_token)
    if ego_pose_token is None: return None
    ego_pose = ego_poses[ego_pose_token]
    R_ego = quaternion_to_mat(*ego_pose["rotation"])
    t_ego = np.array(ego_pose["translation"], dtype=np.float32)
    cam_token = projector._sample_sensor_calib.get(sample_token, {}).get("CAM_FRONT")
    if cam_token is None: return None
    cam_calib = projector.calibs[cam_token]
    R_cam = quaternion_to_mat(*cam_calib["rotation"])
    t_cam = np.array(cam_calib["translation"], dtype=np.float32)
    for ann in gt_anns:
        gt_c = np.array(ann["translation"], dtype=np.float32)
        pt_ego = R_ego.T @ (gt_c - t_ego)
        pt_cam = R_cam.T @ (pt_ego - t_cam)
        if pt_cam[2] <= 0.5: continue
        uv = K @ pt_cam; u, v = uv[0] / uv[2], uv[1] / uv[2]
        if not (0 <= u < 1600 and 0 <= v < 900): continue
        d = math.sqrt((det_cx - u)**2 + (det_cy - v)**2)
        if d < best_dist: best_dist = d; best_ann = ann
    return best_ann


def load_tables(data_root):
    tables = {}
    for name in ["scene","sample","sample_annotation","sample_data","instance","category","ego_pose"]:
        with open(os.path.join(data_root, "v1.0-mini", f"{name}.json")) as f:
            tables[name] = json.load(f)
    return tables


def main():
    from src.dataset_phase2 import Phase2Dataset
    val_ds = Phase2Dataset(DATA_ROOT, split='val')
    val_frame_set = val_ds._frames
    del val_ds

    tables = load_tables(DATA_ROOT)
    ego_poses = {e["token"]: e for e in tables["ego_pose"]}
    sample_data = {s["token"]: s for s in tables["sample_data"]}
    instances = {i["token"]: i for i in tables["instance"]}
    categories = {c["token"]: c for c in tables["category"]}
    inst2cat = {tok: categories[inst["category_token"]]["name"]
                for tok, inst in instances.items()
                if inst["category_token"] in categories}

    gt_by_sample = {}
    for ann in tables["sample_annotation"]:
        if inst2cat.get(ann["instance_token"]) in NUSC_CATS:
            gt_by_sample.setdefault(ann["sample_token"], []).append(ann)

    frame_ego = {}
    for sd in tables["sample_data"]:
        if "CAM_FRONT" in sd["filename"]:
            frame_ego[sd["sample_token"]] = sd["ego_pose_token"]

    frame_sensors = {}
    for sd in tables["sample_data"]:
        sensor = sd["filename"].split("/")[1]
        frame_sensors.setdefault(sd["sample_token"], {})[sensor] = sd["token"]

    valid_frames = [tok for tok, sensors in frame_sensors.items()
                    if "CAM_FRONT" in sensors and "LIDAR_TOP" in sensors
                    and tok in val_frame_set]
    valid_frames.sort()

    print(f"Val frames: {len(valid_frames)}, testing first {NUM_FRAMES}")
    detector = YOLOSegONNX("models/yolov8s-seg.onnx", conf_thresh=0.5)
    projector = LiDARProjector(DATA_ROOT)

    stats_center = [[], []]  # [old_mean, new_visible_centroid]
    stats_offset = []        # visible_centroid → GT offset magnitude
    total_detections = 0

    for frame_idx, sample_token in enumerate(valid_frames[:NUM_FRAMES]):
        cam_token = frame_sensors[sample_token]["CAM_FRONT"]
        img = cv2.imread(os.path.join(DATA_ROOT, sample_data[cam_token]["filename"]))
        if img is None: continue

        lidar_token = frame_sensors[sample_token]["LIDAR_TOP"]
        lidar_full = np.fromfile(
            os.path.join(DATA_ROOT, sample_data[lidar_token]["filename"]),
            dtype=np.float32).reshape(-1, 5)

        K, T_lidar2cam, img_shape = projector.get_transform(sample_token)
        if K is None: continue
        uv, depth, valid_proj = projector.project(lidar_full, K, T_lidar2cam, img_shape)

        lidar_calib_token = projector._sample_sensor_calib.get(sample_token, {}).get("LIDAR_TOP")
        lidar_calib = projector.calibs[lidar_calib_token]
        R_lidar2ego = quaternion_to_mat(*lidar_calib["rotation"])
        t_lidar2ego = np.array(lidar_calib["translation"], dtype=np.float32)
        lidar_yaw = quaternion_to_yaw(*lidar_calib["rotation"])

        dets = detector.predict(img)
        dets = [d for d in dets if d["class_id"] in OBSTACLE_CLASS_IDS]
        gt_anns = gt_by_sample.get(sample_token, [])
        ego_pose_token = frame_ego.get(sample_token)
        ego_pose = ego_poses[ego_pose_token]
        R_ego = quaternion_to_mat(*ego_pose["rotation"])
        t_ego = np.array(ego_pose["translation"], dtype=np.float32)

        for det in dets:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue

            # ---- LiDAR 点: YOLO-seg mask 过滤 (dilate=10) ----
            obj_pts = filter_points_by_mask(uv, valid_proj, depth, det["mask"], lidar_full, dilate=10)
            if obj_pts is None or len(obj_pts) < 10: continue

            cls_id = det["class_id"]
            cat_name = OBSTACLE_CLASSES.get(cls_id, ("unknown",))[0]

            best_ann = match_gt_2d(det["bbox"], gt_anns, K, sample_token,
                                   frame_ego, ego_poses, projector)
            if best_ann is None: continue

            gt_center_global = np.array(best_ann["translation"], dtype=np.float32)
            gt_center_ego = R_ego.T @ (gt_center_global - t_ego)
            gt_center_lidar = R_lidar2ego.T @ (gt_center_ego - t_lidar2ego)

            obj_xyz = obj_pts[:, :3].astype(np.float32)

            # ---- Method A: OLD — full point cloud mean ----
            old_center_lidar = obj_xyz.mean(axis=0)

            # ---- Method B: NEW — visible centroid (z-bottom 35%, same as training) ----
            z_vals = obj_xyz[:, 2]
            z_cut = z_vals.min() + (z_vals.max() - z_vals.min()) * 0.35
            bottom_mask = z_vals <= z_cut
            if bottom_mask.sum() >= 3:
                new_center_lidar = obj_xyz[bottom_mask].mean(axis=0)
            else:
                new_center_lidar = old_center_lidar.copy()

            # Errors
            old_c_err = np.linalg.norm(gt_center_lidar - old_center_lidar)
            new_c_err = np.linalg.norm(gt_center_lidar - new_center_lidar)
            offset_norm = np.linalg.norm(gt_center_lidar - new_center_lidar)

            stats_center[0].append(old_c_err)
            stats_center[1].append(new_c_err)
            stats_offset.append(offset_norm)
            total_detections += 1

            if total_detections <= 8:
                print(f"  [{cat_name:20s}] "
                      f"mean={old_c_err:.2f}m → z-bottom={new_c_err:.2f}m | "
                      f"offset={offset_norm:.2f}m")

    # Summary
    print(f"\n{'='*70}")
    print(f"可见表面质心 vs 全点均值 (n={total_detections})")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Old (全点均值)':>15} {'New (Z-bottom)':>15} {'改善':>10}")
    print(f"{'-'*65}")

    old_m, new_m = np.mean(stats_center[0]), np.mean(stats_center[1])
    impr = (old_m - new_m) / max(old_m, 1e-6) * 100
    print(f"{'Center err':<25} {old_m:15.3f}m {new_m:15.3f}m {impr:9.1f}%")

    offset_mean, offset_std = np.mean(stats_offset), np.std(stats_offset)
    offset_max = np.max(stats_offset)
    print(f"\nOffset distribution (visible_centroid → GT):")
    print(f"  mean={offset_mean:.2f}m, std={offset_std:.2f}m, max={offset_max:.2f}m")
    print(f"  ≤0.5m: {sum(1 for o in stats_offset if o <= 0.5)}/{len(stats_offset)} "
          f"≤1.0m: {sum(1 for o in stats_offset if o <= 1.0)}/{len(stats_offset)} "
          f"≤2.0m: {sum(1 for o in stats_offset if o <= 2.0)}/{len(stats_offset)}")


if __name__ == "__main__":
    main()
