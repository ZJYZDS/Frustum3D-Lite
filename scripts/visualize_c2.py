"""
C2 推理可视化: 整帧 LIDAR_TOP 点云 + 所有检测物体的 3D bbox 叠加在一个 PLY 中.

每帧一个 .ply, 可用 CloudCompare / Open3D 查看:
  - 白色点: LIDAR_TOP 原始点云 (降采样到 ~20000 点)
  - 绿色线框: GT bbox
  - 红色线框: C2 预测 bbox
  - 蓝色线框: Noisy 初始 bbox
"""
import os
import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset_phase1 import (LiDARProjector, quaternion_to_yaw, quaternion_to_mat,
                                 rotate_points_z)
from src.detector import YOLODetectONNX, OBSTACLE_CLASS_IDS, OBSTACLE_CLASSES
from src.model import farthest_point_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "data/nuscenes"
CKPT_PATH = "checkpoints_phase2/lidar_only.pt"
OUT_DIR = "display"

# 无 GT 匹配时各类别默认尺寸 (宽, 长, 高) — 典型 nuScenes 均值
DEFAULT_SIZE = {
    0: (0.7, 0.7, 1.75),  # person
    1: (0.5, 1.8, 1.2),   # bicycle
    2: (2.0, 4.5, 1.6),   # car
    3: (0.8, 2.2, 1.5),   # motorcycle
    5: (2.8, 10.0, 3.0),  # bus
    7: (2.8, 7.0, 2.5),   # truck
}
NUM_FRAMES = 4
MANUAL_SEED = 42
EDGE_PTS_PER_BBOX = 1200
MAX_LIDAR_PTS = 20000   # 整帧 LIDAR_TOP 降采样点数

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(MANUAL_SEED)


def bbox_edges_as_points(center, size, yaw, n_pts=EDGE_PTS_PER_BBOX):
    # nuScenes: size=(宽,长,高), yaw=0° 时长沿 x(前), 宽沿 y(侧)
    l = size[1] / 2.0   # 长 → x
    w = size[0] / 2.0   # 宽 → y
    h = size[2] / 2.0
    corners = np.array([
        [-l, -w, -h], [ l, -w, -h], [ l,  w, -h], [-l,  w, -h],
        [-l, -w,  h], [ l, -w,  h], [ l,  w,  h], [-l,  w,  h],
    ], dtype=np.float32)
    cos, sin = math.cos(yaw), math.sin(yaw)
    R = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    corners = corners @ R.T + center.reshape(1, 3)
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pts_per_edge = n_pts // len(edges)
    all_pts = []
    for i, j in edges:
        t = np.linspace(0, 1, pts_per_edge)
        all_pts.append(corners[i] + t[:, None] * (corners[j] - corners[i]))
    return np.vstack(all_pts).astype(np.float32)


def extract_core_points(obj_pts, cls_id, bbox, uv, valid_proj, lidar_full, in_bbox_mask):
    """中点垂直切片 + DBSCAN → 提取目标核心表面点 (与训练时 _extract_core_points 一致)."""
    x1, y1, x2, y2 = bbox.astype(int)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    crop_w = int((x2 - x1) * 0.4)
    crop_h = int((y2 - y1) * 0.4)

    if crop_w < 3 or crop_h < 3:
        return None

    mid_mask = (
        (uv[:, 0] >= cx - crop_w / 2) & (uv[:, 0] <= cx + crop_w / 2) &
        (uv[:, 1] >= cy - crop_h / 2) & (uv[:, 1] <= cy + crop_h / 2)
    )
    core_pts = lidar_full[mid_mask & valid_proj]
    if len(core_pts) < 5:
        return None

    core_xyz = core_pts[:, :3].astype(np.float32)

    if cls_id == 0:
        eps, min_samples = 0.2, 3
    elif cls_id in (5, 7):
        eps, min_samples = 0.8, 12
    else:
        eps, min_samples = 0.5, 8

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(core_xyz)
    labels = clustering.labels_
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return None

    main_label = np.bincount(valid_labels).argmax()
    return core_xyz[labels == main_label]


def save_ply(path, points_list, color_list):
    all_xyz = [pts.astype(np.float32) for pts in points_list]
    xyz = np.vstack(all_xyz)
    rgb = np.vstack([np.tile(np.array(c, dtype=np.uint8), (len(pts), 1))
                     for pts, c in zip(points_list, color_list)])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(xyz)):
            f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} "
                    f"{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")


def match_gt_2d(det_bbox, gt_anns, K, sample_token, frame_ego, ego_poses, projector):
    det_cx = (det_bbox[0] + det_bbox[2]) / 2
    det_cy = (det_bbox[1] + det_bbox[3]) / 2
    best_ann, best_dist = None, 80.0
    ego_pose_token = frame_ego.get(sample_token)
    if ego_pose_token is None:
        return None
    ego_pose = ego_poses[ego_pose_token]
    R_ego = quaternion_to_mat(*ego_pose["rotation"])
    t_ego = np.array(ego_pose["translation"], dtype=np.float32)
    cam_calib_token = projector._sample_sensor_calib.get(sample_token, {}).get("CAM_FRONT")
    if cam_calib_token is None:
        return None
    cam_calib = projector.calibs[cam_calib_token]
    R_cam = quaternion_to_mat(*cam_calib["rotation"])
    t_cam = np.array(cam_calib["translation"], dtype=np.float32)
    for ann in gt_anns:
        gt_c = np.array(ann["translation"], dtype=np.float32)
        pt_ego = R_ego.T @ (gt_c - t_ego)
        pt_cam = R_cam.T @ (pt_ego - t_cam)
        if pt_cam[2] <= 0.5:
            continue
        uv = K @ pt_cam
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        if not (0 <= u < 1600 and 0 <= v < 900):
            continue
        d = math.sqrt((det_cx - u)**2 + (det_cy - v)**2)
        if d < best_dist:
            best_dist = d
            best_ann = ann
    return best_ann


def draw_2d_image(img_bgr, dets, highlighted_dets, out_path):
    img_rgb = img_bgr[:, :, ::-1].copy()
    h, w = img_rgb.shape[:2]
    scale = min(1200 / w, 800 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_rgb = cv2.resize(img_rgb, (new_w, new_h))
    for det in dets:
        x1 = int(det["bbox"][0] * scale)
        y1 = int(det["bbox"][1] * scale)
        x2 = int(det["bbox"][2] * scale)
        y2 = int(det["bbox"][3] * scale)
        cls_name = OBSTACLE_CLASSES.get(det["class_id"], ("?",))[0]
        if det in highlighted_dets:
            color, thick = (0, 255, 0), 3
        else:
            color, thick = (180, 180, 180), 1
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, thick)
        cv2.putText(img_rgb, cls_name, (x1, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path, img_rgb[:, :, ::-1])


def load_nuscenes_tables(data_root):
    tables = {}
    for name in ["scene", "sample", "sample_annotation", "sample_data",
                 "instance", "category", "ego_pose"]:
        with open(os.path.join(data_root, "v1.0-mini", f"{name}.json")) as f:
            tables[name] = json.load(f)
    return tables


def main():
    from src.fusion import LidarOnlyRefiner

    model = LidarOnlyRefiner().to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tables = load_nuscenes_tables(DATA_ROOT)
    ego_poses = {e["token"]: e for e in tables["ego_pose"]}
    sample_data = {s["token"]: s for s in tables["sample_data"]}
    instances = {i["token"]: i for i in tables["instance"]}
    categories = {c["token"]: c for c in tables["category"]}

    inst2cat = {}
    for tok, inst in instances.items():
        cat = categories.get(inst["category_token"])
        inst2cat[tok] = cat["name"] if cat else "unknown"

    NUSC_CATS = {"vehicle.car", "vehicle.truck", "vehicle.bus",
                 "vehicle.motorcycle", "vehicle.bicycle", "human.pedestrian"}
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

    valid_frames = []
    for tok, sensors in frame_sensors.items():
        if "CAM_FRONT" in sensors and "LIDAR_TOP" in sensors:
            valid_frames.append(tok)
    valid_frames.sort()

    print(f"有效帧: {len(valid_frames)}, 使用前 {NUM_FRAMES} 帧")
    detector = YOLODetectONNX("models/yolo26s.onnx", conf_thresh=0.5)
    projector = LiDARProjector(DATA_ROOT)
    total_objs = 0
    display_idx = 0

    for frame_idx, sample_token in enumerate(valid_frames[:NUM_FRAMES]):
        cam_token = frame_sensors[sample_token]["CAM_FRONT"]
        img = cv2.imread(os.path.join(DATA_ROOT, sample_data[cam_token]["filename"]))
        if img is None:
            continue

        lidar_token = frame_sensors[sample_token]["LIDAR_TOP"]
        lidar_full = np.fromfile(
            os.path.join(DATA_ROOT, sample_data[lidar_token]["filename"]),
            dtype=np.float32).reshape(-1, 5)
        lidar_xyz = lidar_full[:, :3]   # LiDAR 坐标系 (≈ego)

        K, T_lidar2cam, img_shape = projector.get_transform(sample_token)
        if K is None:
            continue

        uv, depth, valid_proj = projector.project(lidar_full, K, T_lidar2cam, img_shape)

        # ---- LiDAR → ego 坐标变换 ----
        lidar_calib_token = projector._sample_sensor_calib.get(sample_token, {}).get("LIDAR_TOP")
        if lidar_calib_token:
            lidar_calib = projector.calibs[lidar_calib_token]
            R_lidar2ego = quaternion_to_mat(*lidar_calib["rotation"])
            t_lidar2ego = np.array(lidar_calib["translation"], dtype=np.float32)
            # pt_ego = R @ pt_lidar + t
            lidar_xyz_ego = (R_lidar2ego @ lidar_xyz.T).T + t_lidar2ego
        else:
            lidar_xyz_ego = lidar_xyz

        dets = detector.predict(img)
        dets = [d for d in dets if d["class_id"] in OBSTACLE_CLASS_IDS]
        if not dets:
            continue

        gt_anns = gt_by_sample.get(sample_token, [])

        # 收集整帧所有物体的 bbox
        points_list = []
        color_list = []

        # 1. LIDAR_TOP 点云: 区分 CAM_FRONT FOV 内外
        in_fov = (valid_proj
                  & (uv[:, 0] >= 0) & (uv[:, 0] < 1600)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < 900)
                  & (depth > 0.5))

        if len(lidar_xyz_ego) > MAX_LIDAR_PTS:
            idx_lidar = np.random.choice(len(lidar_xyz_ego), MAX_LIDAR_PTS, replace=False)
        else:
            idx_lidar = np.arange(len(lidar_xyz_ego))

        mask_fov = in_fov[idx_lidar]
        pts_fov = lidar_xyz_ego[idx_lidar][mask_fov]   # FOV 内: 橙色高亮
        pts_out = lidar_xyz_ego[idx_lidar][~mask_fov]  # FOV 外: 深灰

        if len(pts_out) > 0:
            points_list.append(pts_out)
            color_list.append((60, 60, 60))           # 深灰: 非相机区域
        if len(pts_fov) > 0:
            points_list.append(pts_fov)
            color_list.append((255, 200, 100))        # 橙色: CAM_FRONT FOV 内

        highlighted = []
        results = []
        obj_idx = 0

        for det in dets:
            x1, y1, x2, y2 = det["bbox"].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            margin = 5
            in_bbox = (
                valid_proj
                & (uv[:, 0] >= x1 - margin) & (uv[:, 0] <= x2 + margin)
                & (uv[:, 1] >= y1 - margin) & (uv[:, 1] <= y2 + margin)
                & (depth > 0.5)
            )
            obj_pts = lidar_full[in_bbox]
            if len(obj_pts) < 10:
                continue

            best_ann = match_gt_2d(det["bbox"], gt_anns, K, sample_token,
                                   frame_ego, ego_poses, projector)
            has_gt = best_ann is not None
            ego_pose_token = frame_ego.get(sample_token)
            ego_pose = ego_poses[ego_pose_token]
            R_ego = quaternion_to_mat(*ego_pose["rotation"])
            t_ego = np.array(ego_pose["translation"], dtype=np.float32)

            if has_gt:
                cat_name = inst2cat.get(best_ann["instance_token"], "unknown")
                gt_center = R_ego.T @ (np.array(best_ann["translation"], dtype=np.float32) - t_ego)
                gt_size = np.array(best_ann["size"], dtype=np.float32)
                gt_yaw = quaternion_to_yaw(*best_ann["rotation"])
                gt_yaw -= quaternion_to_yaw(*ego_pose["rotation"])
                noisy_center = gt_center + rng.normal(0, 0.3, 3).astype(np.float32)
                noisy_size = gt_size + rng.normal(0, 0.15, 3).astype(np.float32)
                noisy_size = np.clip(noisy_size, 0.3, 20.0)
                noisy_yaw = gt_yaw + math.radians(rng.normal(0, 5.0))
            else:
                # 无 GT: mid-slice + DBSCAN → 核心点 → mean/PCA
                # (与训练时 _extract_core_points 逻辑一致)
                cls_id = det["class_id"]
                obj_xyz_all = obj_pts[:, :3]
                core_xyz = extract_core_points(
                    obj_pts, cls_id, det["bbox"],
                    uv, valid_proj, lidar_full, in_bbox)
                if core_xyz is not None and len(core_xyz) >= 5:
                    obj_xyz = core_xyz
                else:
                    obj_xyz = obj_xyz_all
                obj_mean = obj_xyz.mean(axis=0)
                noisy_center = (R_lidar2ego @ obj_mean.reshape(3, 1)).reshape(3) + t_lidar2ego
                noisy_size = np.array(
                    DEFAULT_SIZE.get(cls_id, (2.0, 4.5, 1.6)),
                    dtype=np.float32)

                # PCA 估计初始 yaw
                centered = obj_xyz[:, :2] - obj_mean[:2]
                cov = centered.T @ centered / len(centered)
                eigvals, eigvecs = np.linalg.eigh(cov)
                principal = eigvecs[:, -1]
                pca_yaw = math.atan2(principal[1], principal[0])
                pca_yaw = math.atan2(math.sin(pca_yaw), math.cos(pca_yaw))
                if abs(pca_yaw) > math.pi / 2:
                    pca_yaw -= math.copysign(math.pi, pca_yaw)
                noisy_yaw = float(pca_yaw)
                cat_name = OBSTACLE_CLASSES.get(cls_id, ("unknown",))[0]

            # ---- C2 推理 ----
            local_xyz = obj_pts[:, :3].astype(np.float32) - noisy_center
            local_xyz = rotate_points_z(local_xyz, -noisy_yaw)
            extent = max(float(np.ptp(local_xyz)), 0.3)
            scale = extent / 2.0
            local_xyz_norm = local_xyz / scale
            scale_feat = np.full((len(obj_pts), 1), np.log(scale), dtype=np.float32)

            n_pts = len(local_xyz_norm)
            if n_pts >= 256:
                idx = farthest_point_sample(
                    torch.from_numpy(local_xyz_norm).unsqueeze(0).float(), 256)[1][0]
                local_xyz_norm = local_xyz_norm[idx.numpy()]
                scale_feat = scale_feat[idx.numpy()]
                intensity = (obj_pts[idx.numpy(), 3:4]
                            if obj_pts.shape[1] >= 4 else np.zeros((256, 1)))
            else:
                reps = 256 // n_pts + 1
                local_xyz_norm = np.tile(local_xyz_norm, (reps, 1))[:256]
                scale_feat = np.tile(scale_feat, (reps, 1))[:256]
                intensity = np.tile(
                    obj_pts[:, 3:4] if obj_pts.shape[1] >= 4 else np.zeros((n_pts, 1)),
                    (reps, 1))[:256]

            point_feats = np.concatenate([local_xyz_norm, intensity, scale_feat], axis=1)
            inp = torch.from_numpy(point_feats).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                residual = model(None, inp)[0].cpu().numpy()

            pred_center = noisy_center + residual[:3]
            pred_size = noisy_size + residual[3:6]
            pred_yaw = noisy_yaw + math.atan2(residual[6], residual[7])

            # 添加到整帧点云: GT (绿), C2 (红), Noisy (蓝)
            if has_gt:
                points_list.append(bbox_edges_as_points(gt_center, gt_size, gt_yaw))
                color_list.append((0, 220, 0))        # 绿色 = GT

                points_list.append(bbox_edges_as_points(noisy_center, noisy_size, noisy_yaw))
                color_list.append((100, 150, 255))    # 蓝色 = Noisy

            points_list.append(bbox_edges_as_points(pred_center, pred_size, pred_yaw))
            color_list.append((255, 40, 40))          # 红色 = C2

            # debug: print pred vs GT for first few frames
            if frame_idx < 4:
                gt_str = f"GT sz=({gt_size[0]:.1f},{gt_size[1]:.1f},{gt_size[2]:.1f}) yaw={math.degrees(gt_yaw):.0f}°" if has_gt else "no GT"
                print(f"  [{obj_idx+1}] {cat_name:20s} pred: sz=({pred_size[0]:.1f},{pred_size[1]:.1f},{pred_size[2]:.1f}) yaw={math.degrees(pred_yaw):.0f}°  |  noisy: sz=({noisy_size[0]:.1f},{noisy_size[1]:.1f},{noisy_size[2]:.1f}) yaw={math.degrees(noisy_yaw):.0f}°  |  {gt_str}")

            highlighted.append(det)
            results.append((noisy_center, noisy_size, noisy_yaw,
                            pred_center, pred_size, pred_yaw,
                            gt_center if has_gt else None,
                            gt_size if has_gt else None,
                            gt_yaw if has_gt else None,
                            cat_name, has_gt))
            obj_idx += 1
            total_objs += 1

        if obj_idx == 0:
            continue

        # 保存整帧 PLY (全景)
        ply_path = os.path.join(OUT_DIR, f"frame_{display_idx+1:02d}.ply")
        save_ply(ply_path, points_list, color_list)

        # 保存 2D 参考图
        jpg_path = os.path.join(OUT_DIR, f"frame_{display_idx+1:02d}_cam.jpg")
        draw_2d_image(img, dets, highlighted, jpg_path)

        print(f"  frame_{display_idx+1:02d}: {obj_idx} objects, "
              f"{sum(len(p) for p in points_list)} pts "
              f"(FOV={len(pts_fov)}/{len(pts_fov)+len(pts_out)} cam pts) -> PLY")

        # ---- 每个物体单独的局部 PLY (自动对焦, 含完整 LIDAR_TOP) ----
        obj_dir = os.path.join(OUT_DIR, f"frame_{display_idx+1:02d}")
        display_idx += 1
        os.makedirs(obj_dir, exist_ok=True)
        all_lidar = lidar_xyz_ego[idx_lidar]  # 当前帧的完整降采样点云
        for i, (nc, ns, ny, pc, ps, py, gc, gs, gy, cat, has_gt) in enumerate(results):
            # 以预测 bbox 为中心, 裁剪周围点云 (半径 = bbox 对角线 * 2)
            radius = np.linalg.norm(ps[:2]) * 2.0 + 2.0
            dists = np.linalg.norm(all_lidar[:, :2] - pc[:2], axis=1)
            nearby = all_lidar[dists < radius]
            in_fov_nearby = in_fov[idx_lidar][dists < radius]

            obj_parts = []
            obj_colors = []
            if len(nearby) > 0:
                # FOV 外: 深灰, FOV 内: 橙色
                obj_parts.append(nearby[~in_fov_nearby])
                obj_colors.append((60, 60, 60))
                obj_parts.append(nearby[in_fov_nearby])
                obj_colors.append((255, 200, 100))

            if has_gt:
                obj_parts.append(bbox_edges_as_points(gc, gs, gy, n_pts=2000))
                obj_colors.append((0, 255, 0))
                obj_parts.append(bbox_edges_as_points(nc, ns, ny, n_pts=2000))
                obj_colors.append((100, 150, 255))
            obj_parts.append(bbox_edges_as_points(pc, ps, py, n_pts=2000))
            obj_colors.append((255, 40, 40))

            obj_fname = f"obj_{i+1:02d}_{cat.replace('.','_')}.ply"
            save_ply(os.path.join(obj_dir, obj_fname), obj_parts, obj_colors)

    print(f"\nDone. {total_objs} objects across {OUT_DIR}/")
    print("用 CloudCompare / Open3D 打开 .ply, 缩放至物体附近即可看到 bbox 线框.")


if __name__ == "__main__":
    main()
