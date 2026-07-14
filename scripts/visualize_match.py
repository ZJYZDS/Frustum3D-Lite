"""可视化: 所有 YOLO bbox + 仅有效样本的点云. 帧 15-19."""
import sys; sys.path.insert(0, '.')
import numpy as np
import cv2, os
from src.dataset_phase3 import Phase3Dataset, aggregate_sweeps, filter_points_by_frustum
from src.detector import OBSTACLE_CLASS_IDS

CLASS_NAMES = {0:'person',1:'bicycle',2:'car',3:'motorcycle',5:'bus',7:'truck'}

ds = Phase3Dataset(
    nusc_root='data/nuscenes', version='v1.0-mini', split='train',
    detector_path='models/yolo26s.onnx', nsweeps=10,
    max_dist=15.0, remove_ground=True, use_augmentation=False)

output_dir = 'display/match_check'
os.makedirs(output_dir, exist_ok=True)

for frame_idx in [15,16,17,18,19]:
    sample_token = ds.frames[frame_idx]
    sample = ds.nusc.get('sample', sample_token)
    img = ds._load_image(sample)
    if img is None: continue
    K, T_lidar2cam, _ = ds.projector.get_transform(sample_token)
    if K is None: continue

    # 加载全量点云
    pc = aggregate_sweeps(ds.nusc, sample, nsweeps=ds.nsweeps)
    pts_all = pc.points[:3, :].T

    # 获取 dataset 产出的有效样本
    valid_samples = ds[frame_idx]
    valid_points = [s['points'].numpy() for s in valid_samples]
    valid_targets = [s['target'].numpy() for s in valid_samples]

    # YOLO 所有检测
    dets = ds.detector.predict(img)
    dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]

    vis = img.copy()

    # 画所有 YOLO 检测框 + 只画有效样本的点云
    for det_idx, det in enumerate(dets):
        x1,y1,x2,y2 = det['bbox'].astype(int)
        cls_name = CLASS_NAMES.get(det['class_id'], str(det['class_id']))

        # 所有框用绿色细线
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.putText(vis, f'{cls_name}', (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1)

        # 检查这个 bbox 对应的点云是否在有效样本中
        obj_pts_test, _ = filter_points_by_frustum(pts_all, det['bbox'], K, T_lidar2cam, margin='auto')
        if len(obj_pts_test) < 5: continue
        pm_test = obj_pts_test[:,:3].mean(axis=0)

        # 找匹配的有效样本
        matched = None
        for vi, (vpts, vtgt) in enumerate(zip(valid_points, valid_targets)):
            vpm = vpts.mean(axis=0)
            if np.linalg.norm(pm_test - vpm) < 0.5:
                matched = (vpts, vtgt, vi)
                break

        if matched is not None:
            vpts, vtgt, vi = matched
            gc = vtgt[:3]
            d = np.linalg.norm(vpts.mean(axis=0) - gc)

            # 投影点云 (蓝色)
            pts_cam = (T_lidar2cam[:3,:3] @ vpts[:,:3].T).T + T_lidar2cam[:3,3]
            valid_z = pts_cam[:,2] > 0.5
            u = (K[0,0]*pts_cam[valid_z,0]/pts_cam[valid_z,2]+K[0,2]).astype(int)
            v = (K[1,1]*pts_cam[valid_z,1]/pts_cam[valid_z,2]+K[1,2]).astype(int)
            for ui, vi_px in zip(u, v):
                if 0<=ui<vis.shape[1] and 0<=vi_px<vis.shape[0]:
                    cv2.circle(vis, (ui,vi_px), 2, (255,0,0), -1)

            # GT 中心投影 (红色十字)
            gt_cam = (T_lidar2cam[:3,:3] @ gc).T + T_lidar2cam[:3,3]
            if gt_cam[2] > 0.5:
                gu = int(K[0,0]*gt_cam[0]/gt_cam[2]+K[0,2])
                gv = int(K[1,1]*gt_cam[1]/gt_cam[2]+K[1,2])
                cv2.drawMarker(vis, (gu,gv), (0,0,255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(vis, f'GT d={d:.1f}m', (gu+10,gv-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

            # 框加粗 + 标注 VALID
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,255), 3)
            cv2.putText(vis, f'VALID #{vi}', (x1, y2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    fname = f'{output_dir}/frame{frame_idx:03d}_all.png'
    cv2.imwrite(fname, vis)
    print(f'{fname}: {len(dets)} dets, {len(valid_samples)} valid')
