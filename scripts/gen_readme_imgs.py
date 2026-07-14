"""Generate README visualization images from GT-bbox pipeline (Frame 01)."""
import sys, os, math, cv2, numpy as np
sys.path.insert(0, '.')
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.dataset_phase3 import Phase3Dataset, aggregate_sweeps
from src.fusion import PointNet3DDetector
from src.dataset_phase1 import LiDARProjector
from nuscenes.nuscenes import NuScenes

device = torch.device('cuda')
model = PointNet3DDetector().to(device).eval()
ckpt = torch.load('checkpoints_phase3/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])

val_set = Phase3Dataset(nusc_root='data/nuscenes', version='v1.0-mini',
    split='val', detector_path='weiTiao_pt/best.pt', nsweeps=5,
    num_points=512, max_dist=50.0, val_scene_ids=2,
    remove_ground=True, use_augmentation=False,
    preprocess_dir='data/nuscenes/preprocess_phase3/nsweeps_5')

nusc = val_set.nusc
pre_dir = 'data/nuscenes/preprocess_phase3/nsweeps_5'
CLASS_NAMES = {0: 'ped', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 6: 'moto', 7: 'bike'}
CLASS_COLORS = {0: '#FF6B6B', 1: '#FF9999', 2: '#6BCB77', 3: '#E67E22',
                4: '#9B59B6', 6: '#4D96FF', 7: '#FFD93D'}
BBOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7),
              (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
CENTER_SCALE = 3.0
SKIP_YAW = {0, 1}

projector = LiDARProjector('data/nuscenes')

def hex2rgb(h):
    h = h.lstrip('#')
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def bbox_corners(c, s, y):
    w, l, h = s
    half = np.array([[l / 2, w / 2, h / 2]])
    cs, ss = math.cos(y), math.sin(y)
    cr = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
                  dtype=np.float32) * half
    R = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]], dtype=np.float32)
    return cr @ R.T + c.reshape(1, 3)

def pca_yaw(pts):
    if len(pts) < 5:
        return 0.0
    xy = pts[:, :2]
    cov = np.cov(xy.T)
    e, ev = np.linalg.eigh(cov)
    y = math.atan2(ev[1, -1], ev[0, -1])
    if y < 0:
        y += math.pi
    return y

def decode_yaw(u, v, cls_id, pts_np):
    yn = math.sqrt(u**2 + v**2 + 1e-8)
    if cls_id in SKIP_YAW:
        return pca_yaw(pts_np)
    if yn < 0.15:
        return pca_yaw(pts_np)
    y = 0.5 * math.atan2(v / yn, u / yn)
    if y < 0:
        y += math.pi
    return y

def save_ply(path, pts_list, color_list):
    xyz = np.vstack([p.astype(np.float32) for p in pts_list])
    rgb = np.vstack([np.tile(np.array(c, dtype=np.uint8), (len(p), 1))
                     for p, c in zip(pts_list, color_list)])
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(xyz)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write(f'{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} '
                    f'{rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n')

# Frame 01
sample_token = val_set.frames[0]
sample = nusc.get('sample', sample_token)
frame_samples = val_set[0]

pp = os.path.join(pre_dir, f'{sample_token}.npy')
lidar = np.load(pp).astype(np.float32)
if len(lidar) > 30000:
    lidar_disp = lidar[np.random.choice(len(lidar), 30000, replace=False)]
else:
    lidar_disp = lidar

cam_tok = sample['data']['CAM_FRONT']
img = cv2.imread(os.path.join('data/nuscenes',
                 nusc.get('sample_data', cam_tok)['filename']))
K, T_l2c, _ = projector.get_transform(sample_token)

h_img, w_img = img.shape[:2]
pts_cam = (T_l2c[:3, :3] @ lidar_disp[:, :3].T).T + T_l2c[:3, 3]
z_f = pts_cam[:, 2]
u_f = (K[0, 0] * pts_cam[:, 0] / z_f.clip(0.01) + K[0, 2]).astype(int)
v_f = (K[1, 1] * pts_cam[:, 1] / z_f.clip(0.01) + K[1, 2]).astype(int)
inside = (z_f > 0.5) & (u_f >= 0) & (u_f < w_img) & (v_f >= 0) & (v_f < h_img)

# Build predictions
preds = []
for s in frame_samples:
    pts = s['points'].unsqueeze(0).to(device)
    cid = torch.tensor([s['class_id']], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model(points=pts, class_ids=cid)
    centroid = pts[0].mean(dim=0).cpu().numpy()
    prior = model.prior_table[cid[0]].cpu().numpy()
    d_center = out[0, :3].cpu().numpy()
    d_size = out[0, 3:6].cpu().numpy()
    u, v = float(out[0, 6]), float(out[0, 7])
    cls_id = s['class_id']

    center = centroid + d_center * CENTER_SCALE
    size = prior * np.exp(d_size) * 1.12
    yaw = decode_yaw(u, v, cls_id, pts[0].cpu().numpy())

    preds.append((center, size, yaw, CLASS_NAMES.get(cls_id, '?'),
                  cls_id, np.linalg.norm(center[:2])))

max_r = 55
print(f'Frame 01: {len(preds)} objects')

# ── TOP VIEW ──
fig, ax = plt.subplots(figsize=(16, 14))
n = min(len(lidar_disp), 25000)
idx = np.random.choice(len(lidar_disp), n, replace=False)
m = np.linalg.norm(lidar_disp[idx, :2], axis=1) < max_r
im = inside[idx] & m
om = (~inside[idx]) & m
ax.scatter(lidar_disp[idx][om, 0], lidar_disp[idx][om, 1],
           c='#333333', s=0.3, alpha=0.3, rasterized=True)
ax.scatter(lidar_disp[idx][im, 0], lidar_disp[idx][im, 1],
           c='#FFD700', s=0.5, alpha=0.55, rasterized=True)
for c, s, yaw, name, cid, d in preds:
    if np.linalg.norm(c[:2]) > max_r:
        continue
    corners = bbox_corners(c, s, yaw)
    clr = CLASS_COLORS.get(cid, 'red')
    for i, j in BBOX_EDGES:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]], color=clr, lw=3.5)
    al = s[1] * 0.7
    ax.arrow(c[0], c[1], al * math.cos(yaw), al * math.sin(yaw),
             head_width=0.6, head_length=0.6, fc=clr, ec=clr, lw=2.5)
    ax.annotate(name, (c[0], c[1]), fontsize=11, color=clr, weight='bold',
                xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('X (m)', fontsize=14)
ax.set_ylabel('Y (m)', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15)
ax.set_xlim(-max_r, max_r)
ax.set_ylim(-max_r, max_r)
ax.set_title('LiDAR Top View -- All 3D BBoxes', fontsize=16)
plt.tight_layout()
plt.savefig('docs/images/frame01_top.png', dpi=200)
plt.close()
print('  top.png done')

# ── FRONT VIEW ──
fig, ax = plt.subplots(figsize=(16, 8))
n = min(len(lidar_disp), 25000)
idx = np.random.choice(len(lidar_disp), n, replace=False)
im = inside[idx] & (np.linalg.norm(lidar_disp[idx, :2], axis=1) < max_r)
om = (~inside[idx]) & (np.linalg.norm(lidar_disp[idx, :2], axis=1) < max_r)
ax.scatter(lidar_disp[idx][om, 0], lidar_disp[idx][om, 2],
           c='#333333', s=0.3, alpha=0.3, rasterized=True)
ax.scatter(lidar_disp[idx][im, 0], lidar_disp[idx][im, 2],
           c='#FFD700', s=0.5, alpha=0.55, rasterized=True)
for c, s, yaw, name, cid, d in preds:
    if np.linalg.norm(c[:2]) > max_r:
        continue
    corners = bbox_corners(c, s, yaw)
    clr = CLASS_COLORS.get(cid, 'red')
    for i, j in BBOX_EDGES:
        ax.plot([corners[i, 0], corners[j, 0]],
                [corners[i, 2], corners[j, 2]], color=clr, lw=3.5)
ax.set_xlabel('X (m)', fontsize=14)
ax.set_ylabel('Z (m)', fontsize=14)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15)
ax.set_title('LiDAR Front View (XZ)', fontsize=16)
plt.tight_layout()
plt.savefig('docs/images/frame01_front.png', dpi=200)
plt.close()
print('  front.png done')

# ── 3D PERSP ──
fig = plt.figure(figsize=(16, 14))
ax = fig.add_subplot(111, projection='3d')
n3 = min(len(lidar_disp), 12000)
idx3 = np.random.choice(len(lidar_disp), n3, replace=False)
m3 = np.linalg.norm(lidar_disp[idx3, :2], axis=1) < max_r
im3 = inside[idx3] & m3
om3 = (~inside[idx3]) & m3
ax.scatter(lidar_disp[idx3][om3, 0], lidar_disp[idx3][om3, 1],
           lidar_disp[idx3][om3, 2], c='#333333', s=0.15, alpha=0.2, rasterized=True)
ax.scatter(lidar_disp[idx3][im3, 0], lidar_disp[idx3][im3, 1],
           lidar_disp[idx3][im3, 2], c='#FFD700', s=0.3, alpha=0.4, rasterized=True)
for c, s, yaw, name, cid, d in preds:
    if np.linalg.norm(c[:2]) > max_r:
        continue
    corners = bbox_corners(c, s, yaw)
    clr = CLASS_COLORS.get(cid, 'red')
    for i, j in BBOX_EDGES:
        ax.plot3D([corners[i, 0], corners[j, 0]],
                  [corners[i, 1], corners[j, 1]],
                  [corners[i, 2], corners[j, 2]], color=clr, lw=3.0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=-60)
plt.tight_layout()
plt.savefig('docs/images/frame01_persp.png', dpi=200)
plt.close()
print('  persp.png done')

# ── CAMERA VIEW ──
out = img.copy()
for c, s, yaw, name, cid, d in preds:
    corners = bbox_corners(c, s, yaw)[:8]
    pts_c = (T_l2c[:3, :3] @ corners.T).T + T_l2c[:3, 3]
    zc = pts_c[:, 2]
    vz = zc > 0.5
    uv = np.zeros((8, 2), dtype=np.float32)
    uv[:, 0] = K[0, 0] * pts_c[:, 0] / zc.clip(0.01) + K[0, 2]
    uv[:, 1] = K[1, 1] * pts_c[:, 1] / zc.clip(0.01) + K[1, 2]
    vb = vz & (uv[:, 0] >= 0) & (uv[:, 0] < w_img) & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
    ch = CLASS_COLORS.get(cid, '#FFFFFF')
    r, g, b = int(ch[1:3], 16), int(ch[3:5], 16), int(ch[5:7], 16)
    for i, j in BBOX_EDGES:
        if vb[i] and vb[j]:
            cv2.line(out, tuple(uv[i].astype(int)), tuple(uv[j].astype(int)),
                     (b, g, r), 3)
    if vb.any():
        m_uv = uv[vb].mean(0).astype(int)
        label = f'{name} y={math.degrees(yaw):.0f}deg d={d:.1f}m'
        cv2.putText(out, label, tuple(m_uv), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (b, g, r), 2)
cv2.imwrite('docs/images/frame01_cam.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 95])
print('  cam.jpg done')

# ── PLY ──
raw = aggregate_sweeps(nusc, sample, nsweeps=1)
full = raw.points[:3, :].T.astype(np.float32)
if len(full) > 80000:
    full = full[np.random.choice(len(full), 80000, replace=False)]
pl, cl = [full], [(160, 160, 160)]
for c, s, yaw, name, cid, d in preds:
    clr = hex2rgb(CLASS_COLORS.get(cid, '#FFFFFF'))
    pts_per = 200
    allp = []
    crs = bbox_corners(c, s, yaw)
    for i, j in BBOX_EDGES:
        t = np.linspace(0, 1, pts_per)
        allp.append(crs[i] + t[:, None] * (crs[j] - crs[i]))
    pl.append(np.vstack(allp).astype(np.float32))
    cl.append(clr)
save_ply('docs/images/frame01.ply', pl, cl)
print('  ply done')
print('All docs/images/ regenerated')
