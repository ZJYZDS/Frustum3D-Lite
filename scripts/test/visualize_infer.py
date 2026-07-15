"""
Phase 3 推理可视化: frustum 管线 (YOLO→frustum→ROR→DBSCAN→Model).

与 visualize_scene.py 不同: 不依赖 GT bbox 内部点, 模拟真实部署.

输出:
  - frame_XX_infer_top.png / _front.png / _persp.png / _cam.jpg
  - frame_XX_gt_top.png ...  (GT-bbox 管线, 对比)

用法:
  python scripts/visualize_infer.py --num_frames 4
"""

import argparse, os, sys, math
from pathlib import Path
import cv2, numpy as np, torch, yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.dataset_phase3 import aggregate_sweeps, filter_points_by_frustum, \
    remove_statistical_outliers, extract_largest_cluster
from src.fusion import PointNet3DDetector
from src.dataset_phase1 import LiDARProjector
from src.inference import pipeline_predict, pipeline_predict_with_gt
from nuscenes.nuscenes import NuScenes

CLASS_NAMES = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus',
               6: 'motorcycle', 7: 'bicycle'}
CLASS_COLORS = {0: '#FF6B6B', 1: '#FF9999', 2: '#6BCB77', 3: '#E67E22',
                4: '#9B59B6', 6: '#4D96FF', 7: '#FFD93D'}
BBOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


# ── 复用 visualize_scene 的绘图函数 ──────────────────────────────────

def hex2rgb(h):
    h = h.lstrip('#'); return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def bbox_edges_as_points(center, size, yaw, n_pts=1200):
    w, l, h = size
    half = np.array([[l/2, w/2, h/2]])
    corners = np.array([
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
    ], dtype=np.float32) * half
    cos, sin = math.cos(yaw), math.sin(yaw)
    R = np.array([[cos,-sin,0],[sin,cos,0],[0,0,1]], dtype=np.float32)
    corners = corners @ R.T + center.reshape(1,3)
    pts_per = n_pts // 12
    all_pts = []
    for i,j in BBOX_EDGES:
        t = np.linspace(0,1,pts_per)
        all_pts.append(corners[i] + t[:,None]*(corners[j]-corners[i]))
    return np.vstack(all_pts).astype(np.float32)

def draw_bbox_2d(ax, corners, d1, d2, color='r', lw=1.5):
    for i,j in BBOX_EDGES:
        ax.plot([corners[i,d1],corners[j,d1]],[corners[i,d2],corners[j,d2]],
                color=color,lw=lw)

def draw_bbox_3d(ax, corners, color='r', lw=1.5):
    for i,j in BBOX_EDGES:
        ax.plot3D([corners[i,0],corners[j,0]],[corners[i,1],corners[j,1]],
                  [corners[i,2],corners[j,2]],color=color,lw=lw)

def save_ply(path, pts_list, color_list):
    xyz = np.vstack([p.astype(np.float32) for p in pts_list])
    rgb = np.vstack([np.tile(np.array(c,dtype=np.uint8),(len(p),1))
                     for p,c in zip(pts_list,color_list)])
    with open(path,'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write(f'{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} '
                    f'{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n')


# ── 视图函数 ────────────────────────────────────────────────────────

def draw_top_view(ax, pts, preds, preds_gt=None, max_r=60, pts_inside=None):
    n = min(len(pts),25000)
    idx = np.random.choice(len(pts),n,replace=False)
    m = np.linalg.norm(pts[idx,:2],axis=1)<max_r
    # Frustum 内点 (CAM_FRONT 可见) 亮色, 外点暗色
    if pts_inside is not None:
        inside_m = pts_inside[idx] & m
        outside_m = (~pts_inside[idx]) & m
        ax.scatter(pts[idx][outside_m,0],pts[idx][outside_m,1],c='darkgray',s=0.15,alpha=0.25,rasterized=True)
        ax.scatter(pts[idx][inside_m,0],pts[idx][inside_m,1],c='gold',s=0.3,alpha=0.5,rasterized=True)
    else:
        ax.scatter(pts[idx][m,0],pts[idx][m,1],c='lightgray',s=0.2,alpha=0.4,rasterized=True)
    # GT predictions (dashed)
    if preds_gt:
        for c,s,yaw,name,cid,_,_ in preds_gt:
            if np.linalg.norm(c[:2])>max_r: continue
            corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
            draw_bbox_2d(ax, corners[:8], 0, 1, color='lightgreen', lw=1.2)
    # Frustum predictions (solid)
    for c,s,yaw,name,cid,_,_ in preds:
        if np.linalg.norm(c[:2])>max_r: continue
        corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
        draw_bbox_2d(ax, corners[:8], 0, 1, color=CLASS_COLORS.get(cid,'red'), lw=1.8)
        al = s[1]*0.6
        ax.arrow(c[0],c[1],al*math.cos(yaw),al*math.sin(yaw),
                 head_width=0.3,head_length=0.3,fc=CLASS_COLORS.get(cid,'red'),
                 ec=CLASS_COLORS.get(cid,'red'))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')
    ax.grid(True,alpha=0.2); ax.set_xlim(-max_r,max_r); ax.set_ylim(-max_r,max_r)

def draw_front_view(ax, pts, preds, max_r=60):
    n = min(len(pts),25000)
    idx = np.random.choice(len(pts),n,replace=False)
    m = np.linalg.norm(pts[idx,:2],axis=1)<max_r
    ax.scatter(pts[idx][m,0],pts[idx][m,2],c='lightgray',s=0.2,alpha=0.4,rasterized=True)
    for c,s,yaw,name,cid,_,_ in preds:
        if np.linalg.norm(c[:2])>max_r: continue
        corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
        draw_bbox_2d(ax, corners[:8], 0, 2, color=CLASS_COLORS.get(cid,'red'), lw=1.5)
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_aspect('equal')
    ax.grid(True,alpha=0.2)

def draw_camera_view(img, preds, out_path):
    """Draw YOLO 2D bbox + fitted 3D params as text, no 3D wireframe."""
    out = img.copy()
    for c, s, yaw, name, cid, dist, bbox in preds:
        x1, y1, x2, y2 = bbox.astype(int)
        color_hex = CLASS_COLORS.get(cid, '#FFFFFF')
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        # YOLO 2D detection box
        cv2.rectangle(out, (x1, y1), (x2, y2), (b, g, r), 2)
        # Fitted 3D params text
        lines = [
            f'{name}',
            f'yaw={math.degrees(yaw):.0f} deg',
            f'sz=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f})m',
            f'd={dist:.1f}m',
        ]
        tx, ty = x1, y1 - 8
        for li, line in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ty_off = ty - li * (th + 4)
            if ty_off - th < 0:
                ty_off = y2 + (li + 1) * (th + 4)
            cv2.putText(out, line, (tx, ty_off), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (b, g, r), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, out)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/phase3.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_phase3/best_model.pt')
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--output_dir', type=str, default='display/infer_frustum')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f: cfg = yaml.safe_load(f)
    dc = cfg['dataset']
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ── Model ──
    model = PointNet3DDetector().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    md = model.state_dict()
    pd = {k:v for k,v in ckpt['model_state_dict'].items()
          if k in md and v.shape==md[k].shape}
    md.update(pd); model.load_state_dict(md)
    vl = ckpt.get('val_loss', None)
    if vl is not None:
        print(f"Model epoch {ckpt['epoch']}, val_loss={vl:.4f}")
    else:
        print(f"Model epoch {ckpt['epoch']}")
    print(f"  {sum(p.numel() for p in model.parameters()):,} params")

    # ── NuScenes ──
    nusc = NuScenes(version=dc.get('version','v1.0-mini'),
                    dataroot=dc['nusc_root'], verbose=True)
    projector = LiDARProjector(dc['nusc_root'])

    # YOLO detector
    from src.detector import YOLOPtDetector, OBSTACLE_CLASS_IDS
    detector = YOLOPtDetector(pt_path=cfg.get('detector_path','weiTiao_pt/best.pt'))

    # Build val frame list
    val_scene_ids = dc.get('val_scene_ids', 2)
    scenes = sorted(nusc.scene, key=lambda s: s['name'])
    val_scenes = scenes[-val_scene_ids:]
    val_scene_tokens = {s['token'] for s in val_scenes}
    val_frames = []
    for sample in nusc.sample:
        if sample['scene_token'] in val_scene_tokens:
            if 'CAM_FRONT' in sample['data'] and 'LIDAR_TOP' in sample['data']:
                val_frames.append(sample['token'])

    pre_dir = dc.get('preprocess_dir', '')
    nsweeps = dc.get('nsweeps', 5)

    print(f"\nRunning frustum inference on {min(args.num_frames, len(val_frames))} val frames...")

    for di in range(min(args.num_frames, len(val_frames))):
        sample_token = val_frames[di]
        sample = nusc.get('sample', sample_token)

        # ── LiDAR point cloud ──
        pp = os.path.join(pre_dir, f'{sample_token}.npy') if pre_dir else ''
        if pp and os.path.exists(pp):
            lidar = np.load(pp).astype(np.float32)
        else:
            raw = aggregate_sweeps(nusc, sample, nsweeps=nsweeps)
            lidar = raw.points[:3, :].T.astype(np.float32)

        # ── Camera ──
        img = cv2.imread(os.path.join(dc['nusc_root'],
                         nusc.get('sample_data', sample['data']['CAM_FRONT'])['filename']))
        if img is None:
            print(f"  Frame {di+1}: no image, skip")
            continue
        K, T_l2c, _ = projector.get_transform(sample_token)
        if K is None:
            continue

        # ── YOLO detection ──
        dets = detector.predict(img)
        dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]

        # ── Frustum pipeline predictions ──
        preds_frustum = pipeline_predict(
            model, lidar, dets, K, T_l2c, device,
            num_points=dc.get('num_points', 512), min_points=30)

        # ── GT-bbox predictions (对比) ──
        from src.dataset_phase3 import Phase3Dataset
        val_set = Phase3Dataset(nusc_root=dc['nusc_root'],
            version=dc.get('version','v1.0-mini'), split='val',
            detector_path=cfg.get('detector_path','weiTiao_pt/best.pt'),
            nsweeps=nsweeps, num_points=dc.get('num_points',512),
            max_dist=dc.get('max_dist',50.0),
            val_scene_ids=val_scene_ids,
            remove_ground=dc.get('remove_ground',True),
            use_augmentation=False, preprocess_dir=pre_dir)

        # Find matching frame index
        try:
            frame_idx = val_set.frames.index(sample_token)
            frame_samples = val_set[frame_idx]
        except (ValueError, IndexError):
            frame_samples = []

        preds_gt = pipeline_predict_with_gt(frame_samples, model, device)

        # ── Format for display ──
        def fmt_preds(pred_list, T_l2c=None):
            out = []
            for p in pred_list:
                c = p['center']; s = p['size']; y = p['yaw']
                nm = p['class_name']; cid = p['class_id']
                # Camera-frame depth (intuitive for image annotation)
                if T_l2c is not None:
                    c_cam = T_l2c[:3,:3] @ c + T_l2c[:3,3]
                    d_cam = float(c_cam[2])  # forward depth in camera frame
                else:
                    d_cam = np.linalg.norm(c[:2])
                bbox = p.get('bbox', np.array([0, 0, 0, 0]))
                out.append((c, s, y, nm, cid, d_cam, bbox))
            return out

        pf = fmt_preds(preds_frustum, T_l2c)
        pg = fmt_preds(preds_gt, T_l2c)

        # ── Frustum mask: CAM_FRONT 可视区域内的 LiDAR 点 ──
        h_img, w_img = img.shape[:2]
        pts_cam = (T_l2c[:3,:3] @ lidar[:,:3].T).T + T_l2c[:3,3]
        z_cam = pts_cam[:,2]
        u_cam = (K[0,0]*pts_cam[:,0]/z_cam.clip(0.01)+K[0,2]).astype(int)
        v_cam = (K[1,1]*pts_cam[:,1]/z_cam.clip(0.01)+K[1,2]).astype(int)
        inside_frustum = (z_cam>0.5) & (u_cam>=0) & (u_cam<w_img) & (v_cam>=0) & (v_cam<h_img)

        # ── Downsample for display ──
        if len(lidar) > 30000:
            idx_ds = np.random.choice(len(lidar), 30000, replace=False)
            lidar_disp = lidar[idx_ds]
            inside_disp = inside_frustum[idx_ds]
        else:
            lidar_disp = lidar
            inside_disp = inside_frustum

        print(f"\n  Frame {di+1:02d}: {len(pf)} frustum / {len(pg)} GT-bbox objects")

        # Print frustum predictions
        for i,(c,s,y,nm,cid,d,_) in enumerate(pf):
            print(f"    [F{i+1}] {nm:<10} c=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) "
                  f"sz=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) yaw={math.degrees(y):.0f}° d={d:.1f}m")

        pfx = f'{args.output_dir}/frame_{di+1:02d}'

        # ── Frustum views ──
        fig,ax=plt.subplots(figsize=(12,10))
        draw_top_view(ax,lidar_disp,pf,pg,pts_inside=inside_disp); plt.tight_layout()
        plt.savefig(f'{pfx}_infer_top.png',dpi=150); plt.close()

        fig,ax=plt.subplots(figsize=(12,6))
        draw_front_view(ax,lidar_disp,pf); plt.tight_layout()
        plt.savefig(f'{pfx}_infer_front.png',dpi=150); plt.close()

        # 3D 视图: frustum 内点蓝色, 外点灰色
        fig=plt.figure(figsize=(14,10)); ax=fig.add_subplot(111,projection='3d')
        n_disp = min(len(lidar_disp),15000)
        idx_d = np.random.choice(len(lidar_disp),n_disp,replace=False)
        rd = lidar_disp[idx_d]; fi = inside_disp[idx_d]
        ax.scatter(rd[~fi,0],rd[~fi,1],rd[~fi,2],c='darkgray',s=0.1,alpha=0.2,rasterized=True)
        ax.scatter(rd[fi,0],rd[fi,1],rd[fi,2],c='gold',s=0.2,alpha=0.4,rasterized=True)
        for c,s,yaw,nm,cid,_,_ in pf:
            corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
            for i,j in BBOX_EDGES:
                ax.plot3D([corners[i,0],corners[j,0]],[corners[i,1],corners[j,1]],
                          [corners[i,2],corners[j,2]],color=CLASS_COLORS.get(cid,'red'),lw=1.8)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=30,azim=-60); plt.tight_layout()
        plt.savefig(f'{pfx}_infer_persp.png',dpi=150); plt.close()

        draw_camera_view(img, pf, f'{pfx}_infer_cam.jpg')

        # ── GT-bbox comparison views ──
        if pg:
            fig,ax=plt.subplots(figsize=(12,10))
            draw_top_view(ax,lidar_disp,pg,pts_inside=inside_disp); plt.tight_layout()
            plt.savefig(f'{pfx}_gt_top.png',dpi=150); plt.close()

            fig,ax=plt.subplots(figsize=(14,10)); ax=fig.add_subplot(111,projection='3d')
            rd = lidar_disp[idx_d]; fi = inside_disp[idx_d]
            ax.scatter(rd[~fi,0],rd[~fi,1],rd[~fi,2],c='darkgray',s=0.1,alpha=0.2,rasterized=True)
            ax.scatter(rd[fi,0],rd[fi,1],rd[fi,2],c='gold',s=0.2,alpha=0.4,rasterized=True)
            for c,s,yaw,nm,cid,_,_ in pg:
                corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
                for i,j in BBOX_EDGES:
                    ax.plot3D([corners[i,0],corners[j,0]],[corners[i,1],corners[j,1]],
                              [corners[i,2],corners[j,2]],color=CLASS_COLORS.get(cid,'red'),lw=1.8)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.view_init(elev=30,azim=-60); plt.tight_layout()
            plt.savefig(f'{pfx}_gt_persp.png',dpi=150); plt.close()

        # ── PLY (frustum) ──
        raw_full = aggregate_sweeps(nusc, sample, nsweeps=1)
        full = raw_full.points[:3,:].T.astype(np.float32)
        if len(full)>80000:
            full = full[np.random.choice(len(full),80000,replace=False)]
        pl, cl = [full], [(160,160,160)]
        for c,s,y,nm,cid,_,_ in pf:
            clr = hex2rgb(CLASS_COLORS.get(cid,'#FFFFFF'))
            pl.append(bbox_edges_as_points(c,s,y,2400)); cl.append(clr)
        save_ply(f'{pfx}_infer.ply', pl, cl)
        print(f"    -> {pfx}_infer_*.png/.jpg + .ply")

    print(f"\nDone. Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
