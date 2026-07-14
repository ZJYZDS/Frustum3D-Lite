"""360° visualization: full LiDAR + all 6 camera YOLO detections with 3D bbox."""
import sys, os, math, cv2, numpy as np; sys.path.insert(0, '.')
import torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.dataset_phase3 import Phase3Dataset, aggregate_sweeps, \
    filter_points_by_frustum, remove_statistical_outliers, extract_largest_cluster
from src.fusion import PointNet3DDetector
from src.dataset_phase1 import LiDARProjector
from src.detector import YOLOPtDetector, OBSTACLE_CLASS_IDS
from src.inference import pipeline_predict
from nuscenes.nuscenes import NuScenes

device = torch.device('cuda'); model = PointNet3DDetector().to(device).eval()
ckpt = torch.load('checkpoints_phase3/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Model epoch {ckpt["epoch"]}')

ds = Phase3Dataset(nusc_root='data/nuscenes', version='v1.0-mini', split='test',
    detector_path='weiTiao_pt/best.pt', nsweeps=5, num_points=512, max_dist=50.0,
    val_scene_ids=1, test_ratio=0.022, remove_ground=True, use_augmentation=False,
    preprocess_dir='data/nuscenes/preprocess_phase3/nsweeps_5')
nusc = ds.nusc; proj = LiDARProjector('data/nuscenes')
detector = YOLOPtDetector(pt_path='weiTiao_pt/best.pt'); pre = 'data/nuscenes/preprocess_phase3/nsweeps_5'
CC = {0:'#FF6B6B',1:'#FF9999',2:'#6BCB77',3:'#E67E22',4:'#9B59B6',5:'#836953',6:'#4D96FF',7:'#FFD93D',8:'#FFD700',9:'#00CED1'}
CN = {0:'ped',1:'rider',2:'car',3:'truck',4:'bus',5:'train',6:'moto',7:'bike',8:'tlight',9:'tsign'}
BE = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
CM = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
os.makedirs('display/360', exist_ok=True)

def bc(c,s,y):
    w,l,h=s; half=np.array([[l/2,w/2,h/2]]); cs,ss=math.cos(y),math.sin(y)
    cr=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=np.float32)*half
    R=np.array([[cs,-ss,0],[ss,cs,0],[0,0,1]],dtype=np.float32); return cr@R.T+c.reshape(1,3)

for di in range(min(2, len(ds.frames))):
    st = ds.frames[di]; sample = nusc.get('sample', st)
    pp = os.path.join(pre, f'{st}.npy')
    lidar = np.load(pp).astype(np.float32)
    if len(lidar) > 40000: lidar_disp = lidar[np.random.choice(len(lidar), 40000, replace=False)]
    else: lidar_disp = lidar

    # Collect all detections from all cameras
    all_dets = []
    cam_data = {}
    for cam in CM:
        img = ds._load_image(sample, cam)
        if img is None: continue
        K, T, _ = proj.get_transform(st, cam)
        if K is None: continue
        dets = detector.predict(img)
        dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]

        # Per-camera YOLO NMS: suppress overlapping bboxes before pipeline
        keep = np.ones(len(dets), dtype=bool)
        for i in range(len(dets)):
            if not keep[i]: continue
            xi1, yi1, xi2, yi2 = dets[i]['bbox']
            ai = (xi2 - xi1) * (yi2 - yi1)
            for j in range(i + 1, len(dets)):
                if not keep[j]: continue
                if dets[i]['class_id'] != dets[j]['class_id']: continue
                xj1, yj1, xj2, yj2 = dets[j]['bbox']
                ix1, iy1 = max(xi1, xj1), max(yi1, yj1)
                ix2, iy2 = min(xi2, xj2), min(yi2, yj2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                aj = (xj2 - xj1) * (yj2 - yj1)
                iou = inter / (ai + aj - inter + 1e-8)
                if iou > 0.3: keep[j] = False
        dets = [d for k, d in zip(keep, dets) if k]

        pf = pipeline_predict(model, lidar, dets, K, T, device, num_points=512, min_points=30)
        for p in pf: p['camera'] = cam
        cam_data[cam] = (K, T, img, dets, pf)
        all_dets.extend(pf)

    # Global dedup: same class + XY < 3.0m → same object (keep best pts)
    all_dets.sort(key=lambda p: -p['num_pts'])
    all_dets_dedup = []
    for p in all_dets:
        c = p['center']; cid = p['class_id']
        dup = False
        for ep in all_dets_dedup:
            if cid == ep['class_id'] and np.linalg.norm(c[:2] - ep['center'][:2]) < 3.0:
                dup = True; break
        if not dup: all_dets_dedup.append(p)

    pfx = f'display/360/frame_{di+1:02d}'
    print(f'Frame {di+1}: {len(all_dets_dedup)} unique objects from {len(CM)} cameras')

    # ── Full 360 LiDAR top view ──
    mr = 45
    fig, ax = plt.subplots(figsize=(18, 18))
    n = min(len(lidar_disp), 35000)
    idx = np.random.choice(len(lidar_disp), n, replace=False)
    m = np.linalg.norm(lidar_disp[idx, :2], axis=1) < mr
    # Three color layers: obstacle (<1.5m) > background (>1.5m)
    dist = np.linalg.norm(lidar_disp[idx], axis=1)
    near = dist < 1.5
    ax.scatter(lidar_disp[idx][m & ~near, 0], lidar_disp[idx][m & ~near, 1],
               c='#333333', s=0.25, alpha=0.35, rasterized=True)
    ax.scatter(lidar_disp[idx][m & near, 0], lidar_disp[idx][m & near, 1],
               c='#FF4444', s=1.5, alpha=0.8, rasterized=True, label='obstacle <1.5m')
    for p in all_dets_dedup:
        c, s, yw = p['center'], p['size'], p['yaw']
        if abs(c[0]) > mr or abs(c[1]) > mr: continue
        cr = bc(c, s, yw); clr = CC.get(p['class_id'], 'red')
        for i, j in BE: ax.plot([cr[i, 0], cr[j, 0]], [cr[i, 1], cr[j, 1]], color=clr, lw=3.0)
    # Draw camera FOV rays
    ego_x, ego_y = 0.9, 0.0  # approximate LiDAR position in ego
    for ci, cam in enumerate(CM):
        angle = -np.pi/3 + ci * np.pi/3  # approximate FOV centers
        ax.plot([ego_x, ego_x + mr * np.cos(angle)], [ego_y, ego_y + mr * np.sin(angle)], 'w--', lw=0.5, alpha=0.3)
        ax.text(ego_x + mr * 0.9 * np.cos(angle), ego_y + mr * 0.9 * np.sin(angle), cam.replace('CAM_', ''), fontsize=7, color='white', alpha=0.5)
    ax.set_xlim(-mr, mr); ax.set_ylim(-mr, mr); ax.set_aspect('equal')
    ax.set_title(f'360 deg LiDAR — {len(all_dets_dedup)} objects', fontsize=16, color='white')
    ax.set_facecolor('#111111'); fig.patch.set_facecolor('#111111')
    ax.tick_params(colors='white'); ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
    plt.tight_layout(); plt.savefig(f'{pfx}_360top.png', dpi=150, facecolor='#111111'); plt.close()

    # ── 6 camera views ──
    fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    for ci, cam in enumerate(CM):
        ax = axes[ci // 3, ci % 3]
        if cam not in cam_data: ax.set_title(cam); ax.axis('off'); continue
        K, T, img, dets, pf = cam_data[cam]
        out = img.copy()
        h_img, w_img = img.shape[:2]
        # Draw YOLO 2D boxes only
        for det in dets:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            clr = CC.get(det['class_id'], '#FFF')
            r, g, b = int(clr[1:3], 16), int(clr[3:5], 16), int(clr[5:7], 16)
            cv2.rectangle(out, (x1, y1), (x2, y2), (b, g, r), 2)
            cv2.putText(out, CN.get(det['class_id'], '?'), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 2)
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        cname = cam.replace('CAM_', '')
        ax.set_title(f'{cname} ({len(pf)} objs)', fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{pfx}_cams.jpg', dpi=120, bbox_inches='tight')
    plt.close()

    # ── PLY ──
    raw = aggregate_sweeps(nusc, sample, nsweeps=1)
    full = raw.points[:3, :].T.astype(np.float32)
    if len(full) > 80000: full = full[np.random.choice(len(full), 80000, replace=False)]
    # Color near-range points red (<1.5m), others gray
    near_mask = np.linalg.norm(full, axis=1) < 1.5
    pl, cl = [], []
    if near_mask.sum() > 0:
        pl.append(full[near_mask]); cl.append((255, 50, 50))
    if (~near_mask).sum() > 0:
        pl.append(full[~near_mask]); cl.append((160, 160, 160))
    def h2rgb(h): h = h.lstrip('#'); return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    for p in all_dets_dedup:
        c, s, yw = p['center'], p['size'], p['yaw']
        clr = h2rgb(CC.get(p['class_id'], '#FFF'))
        pts_per = 200; allp = []; cr = bc(c, s, yw)
        for i, j in BE: t = np.linspace(0, 1, pts_per); allp.append(cr[i] + t[:, None] * (cr[j] - cr[i]))
        pl.append(np.vstack(allp).astype(np.float32)); cl.append(clr)
    xyz = np.vstack([p.astype(np.float32) for p in pl])
    rgb = np.vstack([np.tile(np.array(c, dtype=np.uint8), (len(p), 1)) for p, c in zip(pl, cl)])
    with open(f'{pfx}.ply', 'w') as f:
        f.write('ply\nformat ascii 1.0\n'); f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n'); f.write('end_header\n')
        for ii in range(len(xyz)): f.write(f'{xyz[ii,0]:.4f} {xyz[ii,1]:.4f} {xyz[ii,2]:.4f} {rgb[ii,0]} {rgb[ii,1]} {rgb[ii,2]}\n')

print('Done -> display/360/')
