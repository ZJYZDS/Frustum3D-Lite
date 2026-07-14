"""
Phase 3 训练: PointNet3DDetector — center残差 + size对数残差 + yaw 2θ.
"""

import argparse, os, sys, time, math
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset_phase3 import Phase3Dataset, phase3_collate
from src.fusion import PointNet3DDetector
from src.loss import PointNet3DLoss
from src.metrics import compute_metrics_absolute

CLASS_NAMES = {0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus', 6: 'motorcycle', 7: 'bicycle'}


def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


def build_model(cfg=None):
    return PointNet3DDetector()


@torch.no_grad()
def compute_metrics_from_pred(pred, target, priors, centroids, criterion):
    c, s, y = criterion.denormalize(pred, centroids, priors)
    gc, gs, gy = criterion.denormalize(target, centroids, priors)
    # Center error
    ce = torch.norm(c - gc, dim=1).mean()
    se = torch.abs(s - gs).mean()
    # Yaw axis error (mod π, handle 180° ambiguity in [0,π) range)
    ye = torch.abs(y - gy)
    ye = torch.min(ye, math.pi - ye).mean() * (180.0 / math.pi)
    return {'center_err': ce.item(), 'size_err': se.item(), 'yaw_deg': ye.item()}


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total = {"loss": 0.0, "center": 0.0, "size": 0.0, "yaw": 0.0}
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        if batch is None: continue
        points = batch['points'].to(device)
        target = batch['target'].to(device)
        cids = batch.get('class_ids')
        if cids is not None: cids = cids.to(device)

        optimizer.zero_grad()
        pred = model(points=points, class_ids=cids,
                     face_cov=batch.get('face_cov'),
                     max_face_idx=batch.get('max_face_idx'))
        loss, d = criterion(pred, target, class_ids=cids)
        loss.backward(); optimizer.step()

        for k in total:
            if k in d: total[k] += d[k]

        if batch_idx % 5 == 0:
            cents = points.mean(dim=1)
            pr = model.prior_table[cids] if cids is not None else \
                 model.prior_table[2].unsqueeze(0).expand(points.shape[0], -1)
            m = compute_metrics_from_pred(pred.detach(), target, pr, cents, criterion)
            print(f"  Ep {epoch:3d} B{batch_idx:3d} | loss={d['loss']:.4f} "
                  f"c={d['center']:.4f} s={d['size']:.4f} y={d['yaw']:.4f} | "
                  f"c_err={m['center_err']:.2f}m s_err={m['size_err']:.2f}m yaw={m['yaw_deg']:.0f}°")

    elapsed = time.time() - start
    avg = {k: v/max(len(loader),1) for k,v in total.items()}
    print(f"  [Train] Ep {epoch:3d} | loss={avg['loss']:.4f} c={avg['center']:.4f} "
          f"s={avg['size']:.4f} y={avg['yaw']:.4f} | {elapsed:.1f}s")
    return avg


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, detail=False):
    model.eval()
    total_loss = {"loss": 0.0, "center": 0.0, "size": 0.0, "yaw": 0.0}
    total_metrics = {"center_err": 0.0, "size_err": 0.0, "yaw_deg": 0.0}
    n_b = 0
    all_cid, all_ce, all_se, all_ye = [], [], [], []

    for batch in loader:
        if batch is None: continue
        n_b += 1
        points = batch['points'].to(device)
        target = batch['target'].to(device)
        cids = batch.get('class_ids')
        if cids is not None: cids = cids.to(device)

        pred = model(points=points, class_ids=cids,
                     face_cov=batch.get('face_cov'),
                     max_face_idx=batch.get('max_face_idx'))
        _, d = criterion(pred, target, class_ids=cids)
        cents = points.mean(dim=1)
        pr = model.prior_table[cids] if cids is not None else \
             model.prior_table[2].unsqueeze(0).expand(points.shape[0], -1)
        m = compute_metrics_from_pred(pred, target, pr, cents, criterion)

        for k in total_loss:
            if k in d: total_loss[k] += d[k]
        for k in total_metrics: total_metrics[k] += m[k]

        if detail and cids is not None:
            c, s, y = criterion.denormalize(pred, cents, pr)
            gc, gs, gy = criterion.denormalize(target, cents, pr)
            all_cid.extend(cids.cpu().tolist())
            all_ce.extend(torch.norm(c-gc, dim=1).cpu().tolist())
            all_se.extend(torch.abs(s-gs).mean(dim=1).cpu().tolist())
            yd = torch.abs(y - gy)
            ye = torch.min(yd, math.pi - yd) * (180.0 / math.pi)  # axis error
            all_ye.extend(ye.cpu().tolist())

    n = max(n_b, 1)
    avg_loss = {k: v/n for k,v in total_loss.items()}
    avg_metrics = {k: v/n for k,v in total_metrics.items()}
    print(f"  [Val]   Ep {epoch:3d} | loss={avg_loss['loss']:.4f} "
          f"c={avg_loss['center']:.4f} s={avg_loss['size']:.4f} y={avg_loss['yaw']:.4f} | "
          f"c_err={avg_metrics['center_err']:.2f}m s_err={avg_metrics['size_err']:.2f}m "
          f"yaw={avg_metrics['yaw_deg']:.0f}°")
    dd = None
    if detail: dd = {'class_ids': all_cid, 'center_err': all_ce, 'size_err': all_se, 'yaw_err': all_ye}
    return avg_loss, avg_metrics, dd


def print_detail_report(dd):
    class_ids=np.array(dd['class_ids']); ce=np.array(dd['center_err']); se=np.array(dd['size_err']); ye=np.array(dd['yaw_err'])
    print(f"  [Detail] {'Class':<16} {'Cnt':>5} {'Center':>8} {'Size':>8} {'Yaw':>8}")
    print(f"  {'-'*50}")
    for cid in sorted(set(class_ids)):
        mask=class_ids==cid; n=mask.sum()
        print(f"  {'':>8}{CLASS_NAMES.get(cid,f'cls{cid}'):<16} {n:>5} {ce[mask].mean():>7.3f}m {se[mask].mean():>7.3f}m {ye[mask].mean():>7.1f}°")
    print(f"  {'':>8}{'ALL':<16} {len(class_ids):>5} {ce.mean():>7.3f}m {se.mean():>7.3f}m {ye.mean():>7.1f}°")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/phase3.yaml')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs: cfg['training']['epochs'] = args.epochs
    if args.batch_size: cfg['training']['batch_size'] = args.batch_size
    if args.lr: cfg['training']['lr'] = args.lr

    data_cfg = cfg.get('dataset', {})
    nsweeps = data_cfg.get('nsweeps', 5)
    train_cfg = cfg['training']
    device_str = train_cfg.get('device', 'auto')
    device = torch.device('cuda' if device_str == 'auto' and torch.cuda.is_available() else device_str)
    print(f"Device: {device}")

    # Data
    data_root = data_cfg.get('nusc_root', 'data/nuscenes')
    detector_path = cfg.get('detector_path', 'models/yolo26s.onnx')
    pre_dir = data_cfg.get('preprocess_dir', None)

    print(f"[Phase3] Building datasets... preprocess_dir={pre_dir}")
    f_mix = data_cfg.get('frustum_mix_ratio', 0.3)
    train_set = Phase3Dataset(nusc_root=data_root, version=data_cfg.get('version','v1.0-mini'),
        split='train', detector_path=detector_path, nsweeps=nsweeps,
        num_points=data_cfg.get('num_points',512), crop_size=data_cfg.get('crop_size',128),
        max_dist=data_cfg.get('max_dist',50.0), min_points=data_cfg.get('min_points',5),
        val_scene_ids=data_cfg.get('val_scene_ids',2),
        remove_ground=data_cfg.get('remove_ground',True),
        use_augmentation=data_cfg.get('use_augmentation',True), preprocess_dir=pre_dir,
        frustum_mix_ratio=f_mix)
    val_set = Phase3Dataset(nusc_root=data_root, version=data_cfg.get('version','v1.0-mini'),
        split='val', detector_path=detector_path, nsweeps=nsweeps,
        num_points=data_cfg.get('num_points',512), crop_size=data_cfg.get('crop_size',128),
        max_dist=data_cfg.get('max_dist',50.0), min_points=data_cfg.get('min_points',5),
        val_scene_ids=data_cfg.get('val_scene_ids',2),
        remove_ground=data_cfg.get('remove_ground',True),
        use_augmentation=False, preprocess_dir=pre_dir,
        frustum_mix_ratio=0.0)  # val 纯 GT-bbox 评估

    train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=True,
        collate_fn=phase3_collate, num_workers=train_cfg.get('num_workers',0), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg['batch_size'], shuffle=False,
        collate_fn=phase3_collate, num_workers=0, drop_last=True)

    # Model
    model = build_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Loss
    loss_cfg = cfg.get('loss', {})
    criterion = PointNet3DLoss(
        center_w=4.0, size_w=0.3, yaw_w=0.3,
        center_scale=loss_cfg.get('center_scale', 3.0),
        size_scale=loss_cfg.get('size_scale', 5.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'],
                                  weight_decay=train_cfg.get('weight_decay',1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['epochs'])

    save_dir = Path(train_cfg.get('save_dir','checkpoints_phase3'))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss, history, start_epoch = float('inf'), [], 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('val_loss', float('inf'))
        history = ckpt.get('history', [])
        remaining = train_cfg['epochs'] - ckpt['epoch']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    print(f"\n{'='*60}")
    print(f"Phase 3 Training: epochs {start_epoch}→{train_cfg['epochs']}, "
          f"{len(train_set)} train / {len(val_set)} val frames")
    print(f"Model: PointNet3DDetector ({n_params:,} params)")
    print(f"Loss: center_residual + size_log + yaw_2theta")
    print(f"Frustum mix: {f_mix*100:.0f}% | Face coverage encoder: 12→16")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, train_cfg['epochs'] + 1):
        train_avg = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        need_detail = (epoch % 4 == 0)
        val_avg, val_metrics, dd = validate(model, val_loader, criterion, device, epoch, detail=need_detail)
        if need_detail and dd: print_detail_report(dd)
        scheduler.step()

        history.append({'epoch': epoch, 'train': train_avg, 'val': val_avg, 'val_metrics': val_metrics})

        if val_avg['loss'] < best_loss:
            best_loss = val_avg['loss']
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_loss, 'history': history}, save_dir / 'best_model.pt')
            print(f"  -> Saved best (val_loss={best_loss:.4f})")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history}, save_dir / 'latest.pt')

    print(f"\n{'='*60}")
    if history:
        best = min(history, key=lambda h: h['val']['loss'])
        print(f"Best: epoch={best['epoch']}, val_loss={best['val']['loss']:.4f}")
    print(f"Checkpoints: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
