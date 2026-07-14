"""诊断 GT 匹配失败的具体原因."""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.dataset_phase3 import Phase3Dataset
from src.dataset_phase1 import LiDARProjector
from pyquaternion import Quaternion
import cv2

ds = Phase3Dataset(
    nusc_root='data/nuscenes', version='v1.0-mini', split='train',
    detector_path='models/yolo26s.onnx', nsweeps=3,
    max_dist=15.0, remove_ground=False, use_augmentation=False)

# 查看 frame 0 的匹配情况
sample_token = ds.frames[0]
sample = ds.nusc.get('sample', sample_token)
img = cv2.imread(ds.nusc.get_sample_data_path(sample['data']['CAM_FRONT']))
K, T, _ = ds.projector.get_transform(sample_token)
gt_anns = ds._gt_by_sample.get(sample_token, [])

dets = ds.detector.predict(img)
dets = [d for d in dets if d['class_id'] in {0,1,2,3,5,7}]

print(f"Frame 0: {len(dets)} detections, {len(gt_anns)} GTs")
print()

# 所有 GT 的投影中心
print("GT 标注 (投影到图像平面):")
for i, ann in enumerate(gt_anns):
    pt_cam = ds._global_to_camera(ann['translation'], sample)
    if pt_cam is not None and pt_cam[2] > 0.5:
        u = (K[0,0]*pt_cam[0]/pt_cam[2]) + K[0,2]
        v = (K[1,1]*pt_cam[1]/pt_cam[2]) + K[1,2]
        cat = ds.nusc.get('instance', ann['instance_token'])
        cat_name = ds.nusc.get('category', cat['category_token'])['name'] if cat else '?'
        print(f"  GT[{i}]: ({u:.0f},{v:.0f}) depth={pt_cam[2]:.1f}m, "
              f"center_global={ann['translation']}, cat={cat_name}")
    else:
        print(f"  GT[{i}]: behind camera")

print()
print("YOLO 检测 (匹配到最近 GT):")
for det_idx, det in enumerate(dets):
    x1,y1,x2,y2 = det['bbox'].astype(int)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bbox_w, bbox_h = x2-x1, y2-y1

    # 找最近的 GT
    best_i, best_dist = -1, 999
    for i, ann in enumerate(gt_anns):
        pt_cam = ds._global_to_camera(ann['translation'], sample)
        if pt_cam is None or pt_cam[2] <= 0.5:
            continue
        u = (K[0,0]*pt_cam[0]/pt_cam[2]) + K[0,2]
        v = (K[1,1]*pt_cam[1]/pt_cam[2]) + K[1,2]
        dist = np.sqrt((cx-u)**2 + (cy-v)**2)
        if dist < best_dist:
            best_dist = dist; best_i = i

    # 检查该 bbox 内是否包含 GT 投影中心
    gt_in_bbox = []
    for i, ann in enumerate(gt_anns):
        pt_cam = ds._global_to_camera(ann['translation'], sample)
        if pt_cam is None or pt_cam[2] <= 0.5:
            continue
        u = (K[0,0]*pt_cam[0]/pt_cam[2]) + K[0,2]
        v = (K[1,1]*pt_cam[1]/pt_cam[2]) + K[1,2]
        if x1 <= u <= x2 and y1 <= v <= y2:
            gt_in_bbox.append(i)

    matched_cat = '?'
    if best_i >= 0 and best_i < len(gt_anns):
        ann = gt_anns[best_i]
        cat = ds.nusc.get('instance', ann['instance_token'])
        matched_cat = ds.nusc.get('category', cat['category_token'])['name'] if cat else '?'

    print(f"  Det[{det_idx}]: bbox=({x1},{y1})-({x2},{y2}) {bbox_w}x{bbox_h}, "
          f"center=({cx:.0f},{cy:.0f}), class_id={det['class_id']}")
    print(f"    → matched GT[{best_i}] dist={best_dist:.0f}px cat={matched_cat}")
    print(f"    → GTs inside bbox: {gt_in_bbox}")
    if best_i not in gt_in_bbox and gt_in_bbox:
        print(f"    ⚠️  matched GT center is OUTSIDE bbox, but GT{gt_in_bbox} IS inside!")
    print()
