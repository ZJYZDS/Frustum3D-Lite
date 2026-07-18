"""Real-time 360° detection video: frame-by-frame processing + MP4 output."""
import sys, os, math, time, cv2, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import torch, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.dataset_phase3 import Phase3Dataset, aggregate_sweeps
from src.fusion import PointNet3DDetector
from src.dataset_phase1 import LiDARProjector
from src.detector import YOLOPtDetector, OBSTACLE_CLASS_IDS
from src.inference import pipeline_predict
from src.tracker import Tracker
from nuscenes.nuscenes import NuScenes

device = torch.device('cuda')
model = PointNet3DDetector().to(device).eval()
ckpt = torch.load('checkpoints_phase3/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
print(f'Model epoch {ckpt["epoch"]}')

nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
proj = LiDARProjector('data/nuscenes')
detector = YOLOPtDetector(pt_path='weiTiao_pt/best.pt')
pre = 'data/nuscenes/preprocess_phase3/nsweeps_5'

CC = {0:'#FF6B6B',1:'#FF9999',2:'#6BCB77',3:'#E67E22',4:'#9B59B6',5:'#836953',6:'#4D96FF',7:'#FFD93D',8:'#FFD700',9:'#00CED1'}
BE = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
CM = ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']

# Pick first val scene
val_scene = sorted(nusc.scene, key=lambda s: s['name'])[-1]
scene_samples = []
for s in nusc.sample:
    if s['scene_token'] == val_scene['token']:
        scene_samples.append(s)
scene_samples.sort(key=lambda s: s['timestamp'])
print(f'Scene: {val_scene["name"]}, {len(scene_samples)} frames')

os.makedirs('display/video', exist_ok=True)
tracker = Tracker(max_dist=5.0, min_history=3)
VIDEO_W, VIDEO_H = 1920, 1080
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('display/video/realtime_360.mp4', fourcc, 4, (VIDEO_W, VIDEO_H))

def bc(c,s,y):
    w,l,h=s; half=np.array([[l/2,w/2,h/2]]); cs,ss=math.cos(y),math.sin(y)
    cr=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=np.float32)*half
    R=np.array([[cs,-ss,0],[ss,cs,0],[0,0,1]],dtype=np.float32); return cr@R.T+c.reshape(1,3)

times = []
for fi, sample in enumerate(scene_samples):
    t0 = time.time()
    st = sample['token']

    pp = os.path.join(pre, f'{st}.npy')
    if not os.path.exists(pp): continue
    lidar = np.load(pp).astype(np.float32)
    if len(lidar) > 25000: lidar = lidar[np.random.choice(len(lidar), 25000, replace=False)]

    # Per-camera YOLO detection
    all_preds = []
    cam_frames = {}
    cam_imgs = {}
    for cam in CM:
        img_path = nusc.get_sample_data_path(sample['data'][cam]) if cam in sample['data'] else None
        if img_path is None: continue
        img = cv2.imread(img_path)
        if img is None: continue
        cam_imgs[cam] = img
        K, T, _ = proj.get_transform(st, cam)
        if K is None: continue
        dets = detector.predict(img)
        dets = [d for d in dets if d['class_id'] in OBSTACLE_CLASS_IDS]
        # Per-camera NMS
        keep = np.ones(len(dets), dtype=bool)
        for i in range(len(dets)):
            if not keep[i]: continue
            xi1,yi1,xi2,yi2=dets[i]['bbox']; ai=(xi2-xi1)*(yi2-yi1)
            for j in range(i+1,len(dets)):
                if not keep[j] or dets[i]['class_id']!=dets[j]['class_id']: continue
                xj1,yj1,xj2,yj2=dets[j]['bbox']; ix1,iy1=max(xi1,xj1),max(yi1,yj1); ix2,iy2=min(xi2,xj2),min(yi2,yj2)
                iw,ih=max(0,ix2-ix1),max(0,iy2-iy1); inter=iw*ih; aj=(xj2-xj1)*(yj2-yj1)
                if inter/(ai+aj-inter+1e-8)>0.3: keep[j]=False
        dets = [d for k,d in zip(keep,dets) if k]
        pf = pipeline_predict(model, lidar, dets, K, T, device, num_points=512, min_points=30)
        for p in pf: p['camera'] = cam
        all_preds.extend(pf)
        cam_frames[cam] = cv2.resize(img, (280, 160))

    # Dedup
    all_preds.sort(key=lambda p: -p['num_pts'])
    dedup = []
    for p in all_preds:
        c=p['center']; cid=p['class_id']
        dup=False
        for ep in dedup:
            if cid==ep['class_id'] and np.linalg.norm(c[:2]-ep['center'][:2])<3.0: dup=True; break
            if np.linalg.norm(c[:2]-ep['center'][:2])<1.0: dup=True; break
        if not dup: dedup.append(p)

    # Tracker update → get motion-fitted tracks
    dt = time.time() - t0
    tracks = tracker.update(dedup, sample['timestamp'], proc_latency=dt)
    times.append(dt)

    # ── Render frame ──
    fig = plt.figure(figsize=(VIDEO_W/100, VIDEO_H/100), dpi=100, facecolor='#111111')
    # Left: 360 top view
    ax_lidar = fig.add_axes([0.02, 0.05, 0.52, 0.90])
    ax_lidar.set_facecolor('#111111')
    mr = 45
    n = min(len(lidar), 20000)
    idx = np.random.choice(len(lidar), n, replace=False)
    m = np.linalg.norm(lidar[idx,:2], axis=1) < mr
    near = np.linalg.norm(lidar[idx], axis=1) < 1.5
    ax_lidar.scatter(lidar[idx][m & ~near,0], lidar[idx][m & ~near,1], c='#444', s=0.3, alpha=0.4)
    ax_lidar.scatter(lidar[idx][m & near,0], lidar[idx][m & near,1], c='#FF4444', s=1.2, alpha=0.8)
    ax_lidar.set_xlim(-mr,mr); ax_lidar.set_ylim(-mr,mr); ax_lidar.set_aspect('equal')
    ax_lidar.tick_params(colors='white', labelsize=8)
    ax_lidar.set_title(f'360 deg LiDAR — {len(tracks)} objects', color='white', fontsize=12)
    for spine in ax_lidar.spines.values(): spine.set_color('#555')

    # Draw tracks: bbox uses model yaw, velocity arrow from (vx, vy) directly
    for trk in tracks:
        c = trk['center']; s = trk['size']
        model_yaw = trk.get('model_yaw', trk['yaw'])
        v = trk['v']; tid = trk['track_id']
        vx = trk.get('vx', 0.0); vy = trk.get('vy', 0.0)
        if abs(c[0]) > mr or abs(c[1]) > mr: continue
        cr = bc(c, s, model_yaw); clr = CC.get(trk['class_id'], 'red')
        for i, j in BE: ax_lidar.plot([cr[i, 0], cr[j, 0]], [cr[i, 1], cr[j, 1]], color=clr, lw=2.5)
        if v > 0.5:
            ax_lidar.arrow(c[0], c[1], vx, vy,
                           head_width=0.4, head_length=0.4, fc='cyan', ec='cyan', lw=1.5, alpha=0.8)
        ax_lidar.text(c[0] + 0.5, c[1] + 0.5,
                      f'{trk["class_name"]}#{tid} {v:.1f}m/s',
                      fontsize=7, color=clr, weight='bold')

    # Right: 6 camera thumbnails in 2x3 grid
    for ci, cam in enumerate(CM):
        ax_cam = fig.add_axes([0.57 + (ci%3)*0.14, 0.55 if ci<3 else 0.08, 0.13, 0.40])
        if cam in cam_frames:
            ax_cam.imshow(cv2.cvtColor(cam_frames[cam], cv2.COLOR_BGR2RGB))
        ax_cam.set_title(cam.replace('CAM_',''), color='white', fontsize=8)
        ax_cam.axis('off')

    # Footer: status bar
    fps = 1.0 / (np.mean(times[-20:]) + 1e-8) if times else 0
    fig.text(0.02, 0.01, f'Frame {fi+1}/{len(scene_samples)} | '
             f'{sample["timestamp"]/1e6:.0f}s | {dt*1000:.0f}ms | {fps:.1f} FPS | '
             f'{len(dedup)} objects',
             color='#aaa', fontsize=9, family='monospace')

    # Render to buffer via savefig
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='#111111', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    frame_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame_bgr = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
    if frame_bgr is not None:
        frame_bgr = cv2.resize(frame_bgr, (VIDEO_W, VIDEO_H))
        out_video.write(frame_bgr)
    plt.close(fig)
    buf.close()

    print(f'  Frame {fi+1:3d}: {len(dedup):2d} objs, {dt*1000:4.0f}ms')

out_video.release()
avg_time = np.mean(times) if times else 0
print(f'\nDone: display/video/realtime_360.mp4')
print(f'Frames: {len(times)}, Avg: {avg_time*1000:.0f}ms, FPS: {1/avg_time:.1f}')
