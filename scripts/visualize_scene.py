"""
Phase 3 全场景可视化: LiDAR 点云 + 3D bbox + 相机投影.

输出每帧:
  - frame_XX_top.png / _front.png / _persp.png / _cam.jpg
  - frame_XX.ply  (完整聚合点云 + 所有 3D bbox 线框)

用法:
  python scripts/visualize_scene.py --num_frames 4
"""

import argparse, os, sys, math
from pathlib import Path
import cv2, numpy as np, torch, yaml
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset_phase3 import Phase3Dataset, aggregate_sweeps
from src.fusion import PointNet3DDetector
from src.dataset_phase1 import LiDARProjector

CLASS_NAMES = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 6: 'motorcycle', 7: 'bicycle'}
CLASS_COLORS = {0: '#FF6B6B', 1: '#FF9999', 2: '#6BCB77', 3: '#E67E22', 4: '#9B59B6', 6: '#4D96FF', 7: '#FFD93D'}
BBOX_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
SKIP_YAW_CLASSES = {0, 1}  # pedestrian & rider: no geometric yaw info

CENTER_SCALE = 3.0; SIZE_SCALE = 5.0
YAW_NORM_FALLBACK = 0.15  # below this norm, fall back to PCA or default yaw


def pca_yaw_estimate(pts):
    """Estimate yaw from point cloud PCA (principal direction in XY plane).

    Returns (yaw_rad, confidence) where 0 = unreliable, 1 = clear direction.
    confidence is the ratio of largest to second-largest eigenvalue.
    """
    if len(pts) < 5:
        return 0.0, 0.0
    xy = pts[:, :2]
    cov = np.cov(xy.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # largest eigenvector direction
    principal = eigvecs[:, -1]
    yaw = math.atan2(principal[1], principal[0])
    if yaw < 0:
        yaw += math.pi
    # confidence: how dominant is the principal direction
    confidence = eigvals[-1] / (eigvals[-2] + 1e-8)
    return yaw, confidence


def hex2rgb(h):
    h = h.lstrip('#'); return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)


def bbox_edges_as_points(center, size, yaw, n_pts=1200):
    w, l, h = size[0], size[1], size[2]
    half = np.array([[l/2, w/2, h/2]])  # use (l,w,h) = (x,y,z) extent
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


def save_ply(path, pts_list, color_list):
    xyz = np.vstack([p.astype(np.float32) for p in pts_list])
    rgb = np.vstack([np.tile(np.array(c,dtype=np.uint8),(len(p),1)) for p,c in zip(pts_list,color_list)])
    with open(path,'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write(f'{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n')


def draw_bbox_2d(ax, corners, d1, d2, color='r', lw=1.5):
    for i,j in BBOX_EDGES:
        ax.plot([corners[i,d1],corners[j,d1]],[corners[i,d2],corners[j,d2]],color=color,lw=lw)


def draw_bbox_3d(ax, corners, color='r', lw=1.5):
    for i,j in BBOX_EDGES:
        ax.plot3D([corners[i,0],corners[j,0]],[corners[i,1],corners[j,1]],[corners[i,2],corners[j,2]],color=color,lw=lw)


def project_bbox(img_shape, corners, K, T_l2c):
    h0,w0 = img_shape[:2]
    pts_cam = (T_l2c[:3,:3] @ corners.T).T + T_l2c[:3,3]
    z = pts_cam[:,2]; v = z > 0.5
    uv = np.zeros((8,2),dtype=np.float32)
    uv[:,0] = K[0,0]*pts_cam[:,0]/z.clip(0.01) + K[0,2]
    uv[:,1] = K[1,1]*pts_cam[:,1]/z.clip(0.01) + K[1,2]
    vb = v & (uv[:,0]>=0)&(uv[:,0]<w0)&(uv[:,1]>=0)&(uv[:,1]<h0)
    return uv, vb


def draw_camera_view(img, preds, K, T_l2c, out_path):
    out = img.copy()
    for c,s,yaw,name,cid,_ in preds:
        corners = bbox_edges_as_points(c,s,yaw,n_pts=600)[:8]  # just the 8 corners
        uv, vb = project_bbox(img.shape, corners, K, T_l2c)
        color = CLASS_COLORS.get(cid,'#FFFFFF')
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        for i,j in BBOX_EDGES:
            if vb[i] and vb[j]:
                cv2.line(out, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)), (b,g,r), 2)
        if vb.any():
            m = uv[vb].mean(0).astype(int)
            cv2.putText(out, name, tuple(m), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (b,g,r), 2)
    cv2.imwrite(out_path, out)


def draw_top_view(ax, pts, preds, max_r=60, pts_inside=None):
    n = min(len(pts),25000); idx = np.random.choice(len(pts),n,replace=False)
    m = np.linalg.norm(pts[idx,:2],axis=1)<max_r
    if pts_inside is not None:
        im = pts_inside[idx] & m; om = (~pts_inside[idx]) & m
        ax.scatter(pts[idx][om,0], pts[idx][om,1], c='darkgray', s=0.15, alpha=0.25, rasterized=True)
        ax.scatter(pts[idx][im,0], pts[idx][im,1], c='gold', s=0.3, alpha=0.5, rasterized=True)
    else:
        ax.scatter(pts[idx][m,0], pts[idx][m,1], c='lightgray', s=0.2, alpha=0.4, rasterized=True)
    for c,s,yaw,name,cid,_ in preds:
        if np.linalg.norm(c[:2])>max_r: continue
        corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
        draw_bbox_2d(ax, corners[:8], 0, 1, color=CLASS_COLORS.get(cid,'red'), lw=1.8)
        al = s[1]*0.6; ax.arrow(c[0],c[1],al*math.cos(yaw),al*math.sin(yaw),head_width=0.3,head_length=0.3,fc=CLASS_COLORS.get(cid,'red'),ec=CLASS_COLORS.get(cid,'red'))
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal'); ax.grid(True,alpha=0.2)
    ax.set_xlim(-max_r,max_r); ax.set_ylim(-max_r,max_r); ax.set_title('Top View (XY)')


def draw_front_view(ax, pts, preds, max_r=60, inside=None):
    n = min(len(pts),25000); idx = np.random.choice(len(pts),n,replace=False)
    m = np.linalg.norm(pts[idx,:2],axis=1)<max_r
    if inside is not None:
        im = inside[idx] & m; om = (~inside[idx]) & m
        ax.scatter(pts[idx][om,0], pts[idx][om,2], c='darkgray', s=0.15, alpha=0.25, rasterized=True)
        ax.scatter(pts[idx][im,0], pts[idx][im,2], c='gold', s=0.3, alpha=0.5, rasterized=True)
    else:
        ax.scatter(pts[idx][m,0], pts[idx][m,2], c='lightgray', s=0.2, alpha=0.4, rasterized=True)
    for c,s,yaw,name,cid,_ in preds:
        if np.linalg.norm(c[:2])>max_r: continue
        corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
        draw_bbox_2d(ax, corners[:8], 0, 2, color=CLASS_COLORS.get(cid,'red'), lw=1.5)
    ax.set_xlabel('X'); ax.set_ylabel('Z'); ax.set_aspect('equal'); ax.grid(True,alpha=0.2); ax.set_title('Front View (XZ)')


def draw_persp_view(ax, pts, preds, max_r=60, inside=None):
    n = min(len(pts),15000); idx = np.random.choice(len(pts),n,replace=False)
    m = np.linalg.norm(pts[idx,:2],axis=1)<max_r
    if inside is not None:
        im = inside[idx] & m; om = (~inside[idx]) & m
        ax.scatter(pts[idx][om,0],pts[idx][om,1],pts[idx][om,2],c='darkgray',s=0.1,alpha=0.2,rasterized=True)
        ax.scatter(pts[idx][im,0],pts[idx][im,1],pts[idx][im,2],c='gold',s=0.2,alpha=0.4,rasterized=True)
    else:
        ax.scatter(pts[idx][m,0],pts[idx][m,1],pts[idx][m,2],c='lightgray',s=0.15,alpha=0.35,rasterized=True)
    for c,s,yaw,name,cid,_ in preds:
        if np.linalg.norm(c[:2])>max_r: continue
        corners = bbox_edges_as_points(c,s,yaw,n_pts=1200)
        draw_bbox_3d(ax, corners[:8], color=CLASS_COLORS.get(cid,'red'), lw=1.8)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.view_init(elev=30,azim=-60); ax.set_title('Perspective (3D)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/phase3.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints_phase3/best_model.pt')
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='display')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f: cfg = yaml.safe_load(f)
    dc = cfg['dataset']; device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Model
    model = PointNet3DDetector().to(device).eval()
    ckpt = torch.load(args.checkpoint, map_location=device)
    md = model.state_dict()
    pd = {k:v for k,v in ckpt['model_state_dict'].items() if k in md and v.shape==md[k].shape}
    md.update(pd); model.load_state_dict(md)
    vl = ckpt.get('val_loss', None)
    print(f"Model epoch {ckpt['epoch']}, val_loss={vl:.4f}" if vl is not None else f"Model epoch {ckpt['epoch']}")
    print(f"  Loaded {len(pd)}/{len(md)} matching params, {sum(p.numel() for p in model.parameters()):,} total")

    # Dataset
    val_set = Phase3Dataset(nusc_root=dc['nusc_root'], version=dc.get('version','v1.0-mini'),
        split='val', detector_path=cfg.get('detector_path','models/yolo26s.onnx'),
        nsweeps=dc.get('nsweeps',5), num_points=dc.get('num_points',512),
        max_dist=dc.get('max_dist',50.0), val_scene_ids=dc.get('val_scene_ids',2),
        remove_ground=dc.get('remove_ground',True), use_augmentation=False,
        preprocess_dir=dc.get('preprocess_dir',None))
    nusc = val_set.nusc
    projector = LiDARProjector(dc['nusc_root'])
    pre_dir = dc.get('preprocess_dir','')
    nsweeps = dc.get('nsweeps',5)

    print(f"Running inference on {len(val_set)} val frames...")

    for di in range(min(args.num_frames, len(val_set))):
        sample_token = val_set.frames[di]
        frame_samples = val_set[di]
        if not frame_samples: continue

        # Build predictions
        preds = []
        for s in frame_samples:
            pts = s['points'].unsqueeze(0).to(device)
            cid = torch.tensor([s['class_id']], dtype=torch.long).to(device)
            with torch.no_grad():
                out = model(points=pts, class_ids=cid)

            centroid = pts[0].mean(dim=0).cpu().numpy()
            prior = model.prior_table[cid[0]].cpu().numpy()
            d_center = out[0,:3].cpu().numpy()
            d_size = out[0,3:6].cpu().numpy()
            u, v = float(out[0,6]), float(out[0,7])
            yaw_norm = math.sqrt(u**2+v**2+1e-8)

            center = centroid + d_center * CENTER_SCALE
            size = prior * np.exp(d_size) * 1.12  # 膨胀确保包住点云

            cls_id = s['class_id']
            if cls_id in SKIP_YAW_CLASSES:
                # Pedestrian/rider: use PCA or default
                pca_yaw, conf = pca_yaw_estimate(pts.cpu().numpy()[0])
                yaw = pca_yaw if conf > 1.2 else 0.0
            elif yaw_norm < YAW_NORM_FALLBACK:
                # Low-confidence car: fall back to PCA
                pca_yaw, conf = pca_yaw_estimate(pts.cpu().numpy()[0])
                yaw = pca_yaw if conf > 1.2 else 0.5 * math.atan2(v/yaw_norm, u/yaw_norm)
                if yaw < 0: yaw += math.pi
            else:
                yaw = 0.5 * math.atan2(v/yaw_norm, u/yaw_norm)
                if yaw < 0: yaw += math.pi  # [0, π)

            name = CLASS_NAMES.get(s['class_id'], '?')
            dist = np.linalg.norm(center[:2])
            preds.append((center, size, yaw, name, s['class_id'], dist))

        # Load point cloud for display
        sample = nusc.get('sample', sample_token)
        pp = os.path.join(pre_dir, f'{sample_token}.npy') if pre_dir else ''
        if pp and os.path.exists(pp):
            lidar = np.load(pp).astype(np.float32)
        else:
            raw = aggregate_sweeps(nusc, sample, nsweeps=nsweeps)
            lidar = raw.points[:3,:].T.astype(np.float32)
        if len(lidar) > 30000:
            lidar = lidar[np.random.choice(len(lidar), 30000, replace=False)]

        # Load camera
        cam_tok = sample['data']['CAM_FRONT']
        img = cv2.imread(os.path.join(dc['nusc_root'], nusc.get('sample_data',cam_tok)['filename']))
        K, T_l2c, _ = projector.get_transform(sample_token)

        # Frustum mask for LiDAR point coloring
        h_img, w_img = img.shape[:2]
        pts_cam_f = (T_l2c[:3,:3] @ lidar[:,:3].T).T + T_l2c[:3,3]
        z_f = pts_cam_f[:,2]
        u_f = (K[0,0]*pts_cam_f[:,0]/z_f.clip(0.01)+K[0,2]).astype(int)
        v_f = (K[1,1]*pts_cam_f[:,1]/z_f.clip(0.01)+K[1,2]).astype(int)
        inside_frustum = (z_f>0.5) & (u_f>=0) & (u_f<w_img) & (v_f>=0) & (v_f<h_img)

        print(f"\n  Frame {di+1:02d}: {len(preds)} objects, {len(lidar)} pts")
        for i,(c,s,y,nm,cid,d) in enumerate(preds):
            print(f"    [{i+1}] {nm:<10} c=({c[0]:.1f},{c[1]:.1f},{c[2]:.1f}) sz=({s[0]:.1f},{s[1]:.1f},{s[2]:.1f}) yaw={math.degrees(y):.0f}° d={d:.1f}m")

        pfx = f'{args.output_dir}/frame_{di+1:02d}'

        # PNG 3-view
        fig,ax=plt.subplots(figsize=(12,10))
        draw_top_view(ax,lidar,preds,pts_inside=inside_frustum)
        plt.tight_layout(); plt.savefig(f'{pfx}_top.png',dpi=150); plt.close()
        fig,ax=plt.subplots(figsize=(12,6))
        draw_front_view(ax,lidar,preds,inside=inside_frustum)
        plt.tight_layout(); plt.savefig(f'{pfx}_front.png',dpi=150); plt.close()
        fig=plt.figure(figsize=(14,10)); ax=fig.add_subplot(111,projection='3d')
        draw_persp_view(ax,lidar,preds,inside=inside_frustum)
        plt.tight_layout(); plt.savefig(f'{pfx}_persp.png',dpi=150); plt.close()
        if img is not None:
            cv2.imwrite(f'{pfx}_cam.jpg', img)

        # PLY
        raw = aggregate_sweeps(nusc, sample, nsweeps=1)
        full = raw.points[:3,:].T.astype(np.float32)
        if len(full) > 80000:
            full = full[np.random.choice(len(full),80000,replace=False)]
        pl, cl = [full], [(160,160,160)]
        for c,s,y,nm,cid,_ in preds:
            clr = hex2rgb(CLASS_COLORS.get(cid,'#FFFFFF'))
            pl.append(bbox_edges_as_points(c,s,y,2400)); cl.append(clr)
        save_ply(f'{pfx}.ply', pl, cl)
        print(f"    -> {pfx}_*.png/.jpg + .ply")

    print(f"\nDone. Saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
